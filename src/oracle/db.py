"""ORACLE domain database — schema, connection management, init.

This is the DOMAIN database for ORACLE: signals collected from sources, user
feedback on ideas/signals, topic timeline tracking, and user-added sources.

It is INTENTIONALLY SEPARATE from the LangGraph checkpoint database
(`oracle.db` from Step 1). Reasons:
  - Domain data must persist; checkpoint data can be safely truncated
  - Backup / restore lifecycles differ
  - Schema migrations on one don't risk the other
  - LangGraph owns its own table layout and we shouldn't share writes

The four producer/consumer relationships from the user spec:

  signals          producer: scout/market/trend/custom (Steps 4-7)
                   consumer: synthesizer (Step 9), topic_timeline writer

  feedback         producer: Telegram bot button callbacks (Step 3)
                   consumer: learning system (Step 14) — recalibrates every 30

  topic_timeline   producer: synthesizer extracts topics from signals (Step 9)
                   consumer: idea_generator lifecycle stage; /history command

  user_sources     producer: /add_source bot flow (Step 8)
                   consumer: custom collector node (Step 7)

  portfolio_holdings  producer: /add_holding bot flow (Step 20)
                      consumer: investment_analyzer (personalized advice based on
                                user's actual positions + live market prices)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

log = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Default location — can be overridden per-call (e.g. for tests).
# Step 4 will move this to Settings.data_db_path when config arrives.
DB_PATH = Path("oracle_data.db")

# Bump this when schema changes; init_db() applies missing migrations idempotently.
SCHEMA_VERSION = 4

# Connection-level pragmas — applied on every aiosqlite.connect.
# - foreign_keys: enforce FK constraints (off by default in SQLite, surprising)
# - journal_mode WAL: Write-Ahead Logging — concurrent reads while a writer is active
# - synchronous NORMAL: balanced durability/speed (safe with WAL)
# - busy_timeout 5000ms: wait up to 5s for locks instead of failing immediately
PRAGMAS = (
    "PRAGMA foreign_keys = ON",
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA busy_timeout = 5000",
)


# ============================================================================
# Schema (DDL)
# ============================================================================

SCHEMA_SQL = """
-- ----------------------------------------------------------------------------
-- schema_version — migration tracking
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- ----------------------------------------------------------------------------
-- signals — every piece of evidence collected from any source
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type     TEXT NOT NULL,
        -- 'rss','reddit','hn','producthunt','telegram','youtube',
        -- 'website','twitter_x','market','fred','crunchbase','github_trending'
    source_id       TEXT NOT NULL,
        -- e.g. 'reuters/businessNews', 'r/SaaS', '@bensbites'
    external_id     TEXT,
        -- ID at source (HN item ID, Reddit post ID, TG msg ID) — for dedup
    title           TEXT,
    content         TEXT NOT NULL,
    url             TEXT,
    published_at    TEXT NOT NULL,              -- ISO 8601 UTC
    collected_at    TEXT NOT NULL,              -- ISO 8601 UTC
    freshness_score INTEGER NOT NULL,           -- 0..100 (100=<2h, 75=<24h, 50=<72h, 0=stale)
    is_breaking     INTEGER NOT NULL DEFAULT 0, -- 1 if <2h old at collection
    topic_tags      TEXT,                       -- JSON array: ["agentic-rag","openai"]
    raw_metadata    TEXT,                       -- JSON blob: source-specific extras
    UNIQUE(source_type, source_id, external_id)
);
CREATE INDEX IF NOT EXISTS idx_signals_published  ON signals(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_collected  ON signals(collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_source     ON signals(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_signals_freshness  ON signals(freshness_score DESC);
CREATE INDEX IF NOT EXISTS idx_signals_breaking   ON signals(is_breaking) WHERE is_breaking = 1;

-- ----------------------------------------------------------------------------
-- feedback — Maksim's likes/dislikes/saves on ideas and investment signals
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    item_type     TEXT NOT NULL,                -- 'idea' or 'investment_signal'
    item_id       TEXT NOT NULL,                -- UUID of the idea/signal as shown
    digest_id     TEXT,                          -- which digest run produced it (no FK — runs may be purged)
    feedback_type TEXT NOT NULL,                -- 'like','dislike','save','deep_dive'
    reason        TEXT,
        -- only set for dislikes on ideas:
        -- 'competitors','customer','revenue','complex','market','timing'
    item_snapshot TEXT NOT NULL,                -- JSON of the full item AT TIME OF FEEDBACK
                                                 -- (so weights work even if the item evolves)
    created_at    TEXT NOT NULL                 -- ISO 8601 UTC
);
CREATE INDEX IF NOT EXISTS idx_feedback_created    ON feedback(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_item_type  ON feedback(item_type, feedback_type);
CREATE INDEX IF NOT EXISTS idx_feedback_digest     ON feedback(digest_id);

-- ----------------------------------------------------------------------------
-- topic_timeline — temporal tracking of topics (e.g. "agentic-rag")
-- Used for lifecycle stage detection (EMERGING/GROWING/PEAK/DECLINING)
-- and the /history Telegram command.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS topic_timeline (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    topic            TEXT NOT NULL UNIQUE,        -- normalized: lowercase-hyphenated, e.g. "agentic-rag"
    display_name     TEXT NOT NULL,               -- human form: "Agentic RAG"
    first_seen_at    TEXT NOT NULL,               -- when ORACLE first noticed it
    last_seen_at     TEXT NOT NULL,               -- most recent signal mentioning it
    total_mentions   INTEGER NOT NULL DEFAULT 0,
    weekly_mentions  TEXT,                         -- JSON: {"2026-W14": 12, "2026-W15": 35}
    lifecycle_stage  TEXT NOT NULL DEFAULT 'EMERGING',
        -- 'EMERGING','GROWING','PEAK','DECLINING'
    velocity         REAL NOT NULL DEFAULT 0.0,   -- this_week_mentions / max(last_week_mentions, 1)
    notes            TEXT                          -- optional: synthesizer's running notes
);
CREATE INDEX IF NOT EXISTS idx_topic_last_seen  ON topic_timeline(last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_topic_lifecycle  ON topic_timeline(lifecycle_stage);
CREATE INDEX IF NOT EXISTS idx_topic_velocity   ON topic_timeline(velocity DESC);

-- ----------------------------------------------------------------------------
-- user_sources — custom sources Maksim adds via /add_source
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_sources (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type     TEXT NOT NULL,
        -- 'telegram_channel','youtube_channel','website_blog','rss_custom',
        -- 'twitter_x','reddit_custom'
    source_url      TEXT NOT NULL,                -- URL or identifier ('@bensbites', 'https://example.com/feed')
    display_name    TEXT NOT NULL,                -- human-friendly
    category        TEXT NOT NULL,                -- 'business_ideas','investments','geopolitics','ai_tech','other'
    added_at        TEXT NOT NULL,
    last_fetched_at TEXT,                          -- ISO 8601 UTC of last successful fetch
    last_seen_id    TEXT,                          -- incremental fetching cursor (TG msg ID, RSS GUID, etc)
    is_active       INTEGER NOT NULL DEFAULT 1,
    fetch_config    TEXT,                          -- JSON: source-specific config (max_items, filters, etc)
    quality_score   REAL NOT NULL DEFAULT 0.0,    -- learning system: useful signals per fetch
    UNIQUE(source_type, source_url)
);
CREATE INDEX IF NOT EXISTS idx_sources_active    ON user_sources(is_active, source_type);
CREATE INDEX IF NOT EXISTS idx_sources_category  ON user_sources(category);
CREATE INDEX IF NOT EXISTS idx_sources_quality   ON user_sources(quality_score DESC);

-- ----------------------------------------------------------------------------
-- learning_weights — calibration outputs for the personalization loop (Step 14)
-- Single row per "key". Updated by oracle.learning every N feedbacks (default 30).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS learning_weights (
    key         TEXT PRIMARY KEY,
        -- 'prompt_injection_ideas'      → text injected into idea_generator system prompt
        -- 'prompt_injection_critic'     → text injected into critic system prompt
        -- 'last_calibration_at'         → ISO timestamp of most recent calibration
        -- 'calibrations_run'            → count of calibrations ever run (number)
        -- 'feedbacks_at_last_calibration' → snapshot of total feedback count at calibration time
        -- 'weekly_summary'              → most recent weekly summary text for /stats
    value       REAL,                          -- numeric weight, NULL for text-only keys
    text_value  TEXT,                          -- text content (for prompts, summaries)
    rationale   TEXT,                          -- why this value was chosen
    updated_at  TEXT NOT NULL                  -- ISO 8601 UTC
);
CREATE INDEX IF NOT EXISTS idx_learning_updated ON learning_weights(updated_at DESC);

-- ----------------------------------------------------------------------------
-- llm_cost_log — per-call token + cost audit trail (Step 17)
-- Written by oracle.observability.log_llm_usage after every LLM response.
-- Read by /stats and /cost bot commands. Complements Langfuse tracing with
-- local, query-able cost data that survives without an API call.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_cost_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT,                     -- LangGraph checkpoint thread_id
    agent          TEXT NOT NULL,            -- synthesizer | idea_generator | critic | ...
    model          TEXT NOT NULL,            -- gpt-4o | gpt-4o-mini | ...
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    cost_usd       REAL    NOT NULL DEFAULT 0.0,
    latency_ms     INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_cost_created ON llm_cost_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_cost_agent   ON llm_cost_log(agent);
CREATE INDEX IF NOT EXISTS idx_llm_cost_run     ON llm_cost_log(run_id);

-- ----------------------------------------------------------------------------
-- portfolio_holdings — Maksim's actual positions (Step 20)
-- Read by investment_analyzer to give PERSONALIZED advice based on what he
-- actually owns + live prices. P&L computed on-the-fly from market_data.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_label     TEXT NOT NULL,                 -- must match a label in market.py
                                                    -- catalogs (e.g. 'BTC','NVDA','GOLD','EURPLN')
    asset_class     TEXT NOT NULL,                 -- 'crypto','stock','etf','commodity','forex','cash'
    quantity        REAL NOT NULL,                 -- units owned (BTC, shares, oz, ...)
    avg_buy_price   REAL NOT NULL,                 -- average entry in USD
    currency        TEXT NOT NULL DEFAULT 'USD',   -- entry currency (USD/PLN/EUR/...)
    notes           TEXT,                          -- free-form ('long-term hold','swing','staking',...)
    added_at        TEXT NOT NULL,                 -- ISO 8601 UTC of first add
    updated_at      TEXT NOT NULL,                 -- ISO 8601 UTC of last edit
    UNIQUE(asset_label)                            -- one row per asset; re-add updates avg
);
CREATE INDEX IF NOT EXISTS idx_portfolio_class ON portfolio_holdings(asset_class);
"""


# ============================================================================
# Connection helpers
# ============================================================================


async def _apply_pragmas(conn: aiosqlite.Connection) -> None:
    for pragma in PRAGMAS:
        await conn.execute(pragma)


@asynccontextmanager
async def get_db(db_path: Path | str = DB_PATH) -> AsyncIterator[aiosqlite.Connection]:
    """Async context manager yielding a configured aiosqlite connection.

    Use from agent nodes to read/write domain tables. The connection has:
      - Row factory set to aiosqlite.Row (dict-like access by column name)
      - Foreign keys enforced
      - WAL journaling for concurrent reads
      - 5s busy timeout

    Call init_db() at app startup once before using this.
    """
    conn = await aiosqlite.connect(str(db_path))
    try:
        conn.row_factory = aiosqlite.Row
        await _apply_pragmas(conn)
        yield conn
    finally:
        await conn.close()


async def init_db(db_path: Path | str = DB_PATH) -> None:
    """Create schema if missing, record schema version. Idempotent.

    Safe to call on every startup. Future migrations will go in here too —
    bump SCHEMA_VERSION and add migration blocks per version.
    """
    log.info("init_db: ensuring schema at %s (version %d)", db_path, SCHEMA_VERSION)
    async with get_db(db_path) as conn:
        await conn.executescript(SCHEMA_SQL)
        await conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
        )
        await conn.commit()
    log.info("init_db: schema ready")
