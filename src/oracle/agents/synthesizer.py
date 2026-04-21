"""Synthesizer agent — Step 9.

This is ORACLE's first LLM-using agent. It reads all collected signals from
state (~700 per run), pre-filters to high-signal items, sends them to gpt-4o
with a structured-output schema, and produces 5-10 cross-signal INSIGHT
CLUSTERS for the idea_generator and investment_analyzer downstream.

It also writes to the `topic_timeline` table (Step 2 schema) so we can
track topic velocity (this week vs last week) and assign objective
lifecycle stages (EMERGING / GROWING / PEAK / DECLINING).

Cost discipline:
  - Pre-filter to ~150 best signals (top 15 fresh per source, deduped by URL)
  - Compress each signal to ~80 tokens (source + title + url + 100ch excerpt)
  - Single LLM call per run (~8-12k input tokens, ~$0.05-0.10 with gpt-4o)
  - Output capped at 10 clusters

Per the user spec:
  - 70/30 priority: business ideas first, investments secondary
  - "3 great clusters beat 10 mediocre ones — be selective"
  - Each cluster must cite ≥3 distinct signals (otherwise it's just news)

Gracefully no-ops without OPENAI_API_KEY — returns empty synthesized list,
graph still runs, downstream nodes still placeholders or skip.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..config import get_settings
from ..db import get_db
from ..state import OracleState

log = logging.getLogger(__name__)


# ============================================================================
# Output schema (Pydantic) — what the LLM must return
# ============================================================================

LifecycleStage = Literal["EMERGING", "GROWING", "PEAK", "DECLINING"]
ClusterCategory = Literal["business_idea", "investment", "geopolitics", "ai_tech", "other"]


class InsightCluster(BaseModel):
    """One synthesized cross-signal pattern."""

    topic: str = Field(
        description="kebab-case key, e.g. 'agentic-rag', 'voice-saas-vertical', 'fed-pause'"
    )
    display_name: str = Field(description="human-readable name, e.g. 'Agentic RAG'")
    story: str = Field(
        description="1-2 sentences explaining the cross-signal pattern, citing what fed it"
    )
    lifecycle_stage: LifecycleStage = Field(
        description="EMERGING (fresh, few know) | GROWING (best time to build) | "
                    "PEAK (everyone knows) | DECLINING (fading)"
    )
    confidence: int = Field(
        ge=0, le=100,
        description="0-100; clusters under 40 should be killed unless very high upside"
    )
    related_signal_titles: list[str] = Field(
        min_length=1, max_length=10,
        description="3-10 signal titles that fed this insight"
    )
    related_signal_sources: list[str] = Field(
        min_length=1, max_length=10,
        description="source IDs of contributing signals (e.g. 'hn', 'reddit/r/SaaS')"
    )
    category: ClusterCategory = Field(
        description="business_idea (primary, 70%) | investment (30%) | other"
    )


class SynthesisOutput(BaseModel):
    """Top-level LLM response — list of clusters, sorted by importance."""

    insights: list[InsightCluster] = Field(
        max_length=10,
        description="5-10 high-quality clusters, sorted by importance (most important first)"
    )


# ============================================================================
# Signal selection — pre-filter for cost discipline
# ============================================================================

MAX_SIGNALS_TO_LLM = 150
TOP_PER_SOURCE = 15


def select_signals_for_synthesis(state: OracleState) -> list[dict]:
    """Pick the highest-signal subset of all collected signals.

    Strategy:
      1. Concatenate scout + trend + custom signals (skip market — structured)
      2. Group by source, take top 15 freshest per source (variety constraint)
      3. Dedupe by URL (cross-source reposts)
      4. Sort by (is_breaking desc, freshness_score desc)
      5. Cap at 150 total
    """
    all_signals = (
        list(state.get("scout_signals", []) or [])
        + list(state.get("trend_signals", []) or [])
        + list(state.get("custom_signals", []) or [])
    )

    # Group by source, take top N per source (no source dominates)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for s in all_signals:
        by_source[s.get("source", "unknown")].append(s)

    selected: list[dict] = []
    for src, sigs in by_source.items():
        sigs.sort(key=lambda x: x.get("freshness_score", 0), reverse=True)
        selected.extend(sigs[:TOP_PER_SOURCE])

    # Dedupe by URL (or title if no URL)
    seen: set[str] = set()
    deduped: list[dict] = []
    for s in selected:
        key = (s.get("url") or s.get("title") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    # Sort by importance: breaking first, then by freshness desc
    deduped.sort(
        key=lambda s: (1 if s.get("is_breaking") else 0, s.get("freshness_score", 0)),
        reverse=True,
    )
    return deduped[:MAX_SIGNALS_TO_LLM]


def format_signals_for_llm(signals: list[dict]) -> str:
    """Compress signals into a token-efficient text block.

    Format per signal (~80 tokens):
        [<source>] <title>
          url: <url>
          excerpt: <first 100 chars>...
    """
    lines: list[str] = []
    for i, s in enumerate(signals, start=1):
        source = s.get("source", "unknown")
        title = (s.get("title") or "").strip().replace("\n", " ")[:200]
        url = s.get("url", "")
        excerpt = (s.get("content") or "").strip().replace("\n", " ")[:120]
        breaking = " 🔴" if s.get("is_breaking") else ""
        lines.append(f"#{i} [{source}]{breaking} {title}")
        if url:
            lines.append(f"  url: {url}")
        if excerpt and excerpt != title:
            lines.append(f"  ...{excerpt}")
    return "\n".join(lines)


def format_market_for_llm(market_data: dict) -> str:
    """One-line market context."""
    parts: list[str] = []
    if (btc := market_data.get("crypto", {}).get("BTC")):
        parts.append(f"BTC ${btc['price']:,.0f} ({btc['change_24h']:+.1f}%)")
    if (eth := market_data.get("crypto", {}).get("ETH")):
        parts.append(f"ETH ${eth['price']:,.0f} ({eth['change_24h']:+.1f}%)")
    if (spy := market_data.get("equities", {}).get("SPY")):
        parts.append(f"SPY ${spy['price']:.2f} ({spy['change_24h']:+.1f}%)")
    if (qqq := market_data.get("equities", {}).get("QQQ")):
        parts.append(f"QQQ ${qqq['price']:.2f} ({qqq['change_24h']:+.1f}%)")
    if (gold := market_data.get("commodities", {}).get("GOLD")):
        parts.append(f"Gold ${gold['price']:.0f} ({gold['change_24h']:+.1f}%)")
    if (oil := market_data.get("commodities", {}).get("OIL_WTI")):
        parts.append(f"Oil ${oil['price']:.0f} ({oil['change_24h']:+.1f}%)")
    if (vix := market_data.get("indices", {}).get("VIX")):
        parts.append(f"VIX {vix['price']:.1f}")
    if (dxy := market_data.get("forex", {}).get("DXY")):
        parts.append(f"DXY {dxy['price']:.1f}")
    if (tnx := market_data.get("indices", {}).get("TNX_10Y")):
        parts.append(f"10Y {tnx['price']:.2f}%")
    return ("Markets: " + " · ".join(parts)) if parts else "Markets: (no data this run)"


# ============================================================================
# Prompts
# ============================================================================

SYNTHESIZER_SYSTEM = """\
You are ORACLE's signal synthesizer for Maksim, a Python AI engineer who can \
build solo MVPs in 2-6 weeks using Python / FastAPI / LangGraph / RAG / LLMs.

You receive a batch of signals collected this run from news (Reuters, FT, \
Bloomberg, BBC, Guardian via RSS), Reddit (r/SaaS, r/indiehackers, r/startups, \
r/microsaas, r/MachineLearning, r/LocalLLaMA, r/artificial), HackerNews top \
stories, ProductHunt, GitHub trending repos, VC deal RSS (TechCrunch, \
Crunchbase), and the user's custom Telegram / YouTube / blog sources. \
Plus a one-line market data summary.

Your job: find 5-10 CROSS-SIGNAL PATTERNS — moments where multiple \
INDEPENDENT signals point to the same emerging trend. These are the seeds \
the idea_generator and investment_analyzer downstream will turn into \
business ideas and trade scenarios.

PRIORITY (this is critical):
- 70% of clusters should target BUSINESS IDEAS — SaaS / AI tools / micro-SaaS
  / agentic apps / dev tools / API-first products. Maksim wants to spot
  ideas BEFORE they hype, with enough lead time to ship an MVP first.
- 30% of clusters should target INVESTMENT signals — macro, stocks, crypto,
  commodities, geopolitics that affects markets.
- Skip clusters with confidence under 40 unless the upside is exceptional.

QUALITY RULES (per user spec — "3 great beat 10 mediocre"):
1. Each cluster MUST cite ≥3 DISTINCT signals (otherwise it's just one piece
   of news, not a pattern). Cluster signals from DIFFERENT sources when
   possible — that's the cross-signal value.
2. Skip generic noise like "AI is changing everything" — find SPECIFIC
   opportunities. Name the niche, the buyer, the tech enabler.
3. For BUSINESS IDEA clusters, every story MUST hint at: what changed
   recently (the "why now"), who would pay (target customer), and what
   makes it timely (market timing).
4. For INVESTMENT clusters, link to specific assets and reference current
   market context when relevant. Educational analysis only — never give
   actual buy/sell recommendations.
5. Prefer EMERGING and GROWING lifecycle stages — those are where Maksim
   has time to act. PEAK is acceptable for niche plays. DECLINING means
   skip unless the cluster is about a contrarian opportunity.
6. SKIP obvious spam, low-quality reposts, or content with no substance.
7. Every cluster's `related_signal_titles` and `related_signal_sources`
   MUST come VERBATIM from the input signals. Do not invent signals.

OUTPUT: respond ONLY with valid JSON matching the schema. Sort clusters by
importance, most important first.
"""


# ============================================================================
# LLM call
# ============================================================================


async def call_synthesizer_llm(
    signals: list[dict],
    market_data: dict,
) -> SynthesisOutput | None:
    """Single OpenAI call. Returns None on error (graceful degradation)."""
    settings = get_settings()
    from ..observability import has_llm_credentials  # noqa: PLC0415
    if not has_llm_credentials():
        log.info("synthesizer: no LLM credentials (set OPENAI_API_KEY or AZURE_OPENAI_*) — skipping")
        return None

    try:
        from ..observability import get_openai_client, log_llm_usage  # noqa: PLC0415
    except ImportError:
        log.warning("synthesizer: observability module unavailable")
        return None

    try:
        client = get_openai_client(agent="synthesizer")
    except RuntimeError as e:
        log.error("synthesizer: %s", e)
        return None

    signal_blob = format_signals_for_llm(signals)
    market_line = format_market_for_llm(market_data)
    user_msg = f"{market_line}\n\n=== {len(signals)} signals ===\n{signal_blob}"

    log.info(
        "synthesizer: calling %s with %d signals (~%d KB user msg)",
        settings.openai_model_heavy,
        len(signals),
        len(user_msg) // 1024,
    )

    try:
        response = await client.beta.chat.completions.parse(
            model=settings.openai_model_heavy,
            messages=[
                {"role": "system", "content": SYNTHESIZER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=SynthesisOutput,
            temperature=0.4,  # some creativity, mostly grounded
        )
    except Exception as e:  # noqa: BLE001
        log.error("synthesizer: LLM call failed: %s", e)
        return None

    await log_llm_usage("synthesizer", response)

    parsed = response.choices[0].message.parsed
    if not parsed:
        log.error("synthesizer: LLM returned no parsed output")
        return None

    usage = response.usage
    if usage:
        log.info(
            "synthesizer: %d clusters · in=%d out=%d tokens",
            len(parsed.insights),
            usage.prompt_tokens,
            usage.completion_tokens,
        )
    return parsed


# ============================================================================
# topic_timeline updater
# ============================================================================


def _compute_velocity_and_lifecycle(
    weekly_mentions: dict[str, int],
) -> tuple[float, str]:
    """Compute velocity (this week / last week) and derive lifecycle stage.

    Stages per user spec:
      EMERGING  — brand new (no last-week data) and small total
      GROWING   — velocity > 1.5  (best time to build)
      PEAK      — 0.5 ≤ velocity ≤ 1.5  (saturated)
      DECLINING — velocity < 0.5
    """
    weeks_sorted = sorted(weekly_mentions.keys(), reverse=True)
    if not weeks_sorted:
        return 1.0, "EMERGING"
    this_week = weekly_mentions.get(weeks_sorted[0], 0)
    last_week = weekly_mentions.get(weeks_sorted[1], 0) if len(weeks_sorted) >= 2 else 0

    if last_week == 0:
        # New topic — call it EMERGING regardless of size this week
        return (3.0 if this_week > 0 else 1.0), "EMERGING"

    velocity = this_week / last_week
    if velocity > 1.5:
        stage = "GROWING"
    elif velocity < 0.5:
        stage = "DECLINING"
    else:
        stage = "PEAK"
    return velocity, stage


async def upsert_topic_timeline(insights: list[InsightCluster]) -> None:
    """Write/update topic_timeline rows for each insight cluster.

    Uses week key 'YYYY-Www' (ISO week). Read-modify-write per topic to merge
    weekly_mentions JSON. All updates wrapped in a single transaction.
    """
    if not insights:
        return

    now = datetime.now(timezone.utc)
    week_key = now.strftime("%Y-W%V")
    now_iso = now.isoformat()

    async with get_db() as conn:
        for insight in insights:
            topic = insight.topic.strip().lower().replace(" ", "-")
            if not topic:
                continue
            mention_count = len(insight.related_signal_titles)

            async with conn.execute(
                "SELECT * FROM topic_timeline WHERE topic = ?", (topic,)
            ) as cur:
                existing = await cur.fetchone()

            if existing:
                weekly = json.loads(existing["weekly_mentions"] or "{}")
                weekly[week_key] = weekly.get(week_key, 0) + mention_count
                velocity, lifecycle = _compute_velocity_and_lifecycle(weekly)
                await conn.execute(
                    """UPDATE topic_timeline SET
                        last_seen_at = ?,
                        total_mentions = total_mentions + ?,
                        weekly_mentions = ?,
                        lifecycle_stage = ?,
                        velocity = ?
                       WHERE id = ?""",
                    (
                        now_iso, mention_count, json.dumps(weekly),
                        lifecycle, velocity, existing["id"],
                    ),
                )
            else:
                weekly = {week_key: mention_count}
                # New topic — EMERGING by default
                await conn.execute(
                    """INSERT INTO topic_timeline
                        (topic, display_name, first_seen_at, last_seen_at,
                         total_mentions, weekly_mentions, lifecycle_stage, velocity)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        topic, insight.display_name, now_iso, now_iso,
                        mention_count, json.dumps(weekly), "EMERGING", 1.0,
                    ),
                )
        await conn.commit()


# ============================================================================
# LangGraph node
# ============================================================================


async def synthesizer_node(state: OracleState) -> dict:
    """Step 9 real implementation — replaces the Step 1 placeholder.

    Returns a state patch with `synthesized` populated. Graceful empty
    list when OpenAI key is missing or LLM call fails.
    """
    selected = select_signals_for_synthesis(state)
    log.info("synthesizer: selected %d signals for LLM (from total state)", len(selected))

    if not selected:
        log.info("synthesizer: no signals to synthesize — empty result")
        return {"synthesized": []}

    market_data = state.get("market_data", {}) or {}
    parsed = await call_synthesizer_llm(selected, market_data)

    if parsed is None:
        return {"synthesized": []}

    # Persist to topic_timeline
    try:
        await upsert_topic_timeline(parsed.insights)
    except Exception as e:  # noqa: BLE001
        log.error("synthesizer: topic_timeline upsert failed: %s", e)
        # Don't fail the run — synthesized output is still valid

    # Compact log of cluster headlines
    for i, c in enumerate(parsed.insights[:5], start=1):
        log.info(
            "  cluster #%d [%s] %s — %s (conf=%d)",
            i, c.lifecycle_stage, c.display_name, c.category, c.confidence,
        )

    return {
        "synthesized": [c.model_dump() for c in parsed.insights],
    }


# ============================================================================
# Standalone CLI: `uv run python -m oracle.agents.synthesizer`
# ============================================================================


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    async def _main() -> None:
        from ..db import init_db
        await init_db()

        # Build a fake state by reading recent signals from DB
        async with get_db() as conn:
            async with conn.execute(
                """SELECT source_id AS source, title, content, url, published_at,
                          freshness_score, is_breaking
                   FROM signals
                   WHERE freshness_score >= 50
                   ORDER BY published_at DESC
                   LIMIT 800"""
            ) as cur:
                rows = await cur.fetchall()

        signals = [dict(r) for r in rows]
        if not signals:
            print("No fresh signals in DB — run `uv run python -m oracle.main` first.")
            return

        # Bucket the signals so the existing selector logic can process them
        scout, trend, custom = [], [], []
        for s in signals:
            src = s["source"]
            if src.startswith(("rss/", "rss")) and not src.startswith("rss_custom"):
                scout.append(s)
            elif src.startswith(("hn", "reddit/", "github", "producthunt")):
                trend.append(s)
            else:
                custom.append(s)

        fake_state: OracleState = {
            "scout_signals": scout,
            "market_data": {},
            "trend_signals": trend,
            "custom_signals": custom,
            "synthesized": [],
            "raw_ideas": [],
            "investment_scenarios": [],
            "reflexion_round": 0,
            "surviving_ideas": [],
            "validated": [],
            "final_digest": {},
            "errors": [],
        }

        result = await synthesizer_node(fake_state)
        clusters = result.get("synthesized", [])
        print()
        print("=" * 70)
        print(f"Synthesizer output: {len(clusters)} clusters")
        print("=" * 70)
        for i, c in enumerate(clusters, start=1):
            print()
            print(f"#{i} [{c['lifecycle_stage']}] {c['display_name']}  ({c['category']}, conf={c['confidence']})")
            print(f"   {c['story']}")
            print(f"   Sources: {', '.join(c['related_signal_sources'][:5])}")

    asyncio.run(_main())
