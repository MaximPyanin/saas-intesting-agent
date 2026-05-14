"""Feedback learning module — Step 14.

Closes the personalization loop. Every N feedbacks (default 30), runs a
calibration LLM call that reads recent rows from oracle_data.db.feedback
and produces:

  - `prompt_injection_ideas`  — text injected into idea_generator's system
                                prompt on the next graph run, steering future
                                generation toward what Maksim actually likes
  - `weekly_summary`          — Telegram-friendly summary text shown by the
                                bot's /stats command
  - calibration counters and timestamps in `learning_weights` table

Trigger paths:
  - Auto: when total feedbacks % 30 == 0, the bot's `_save_feedback` fires
    a background calibration task (fire-and-forget; non-blocking for the
    button click)
  - Manual: `/calibrate` slash command in the bot
  - CLI:    `uv run python -m oracle.learning` (read-only status)
            `uv run python -m oracle.learning --force` (force-run now)

Read by:
  - `agents/idea_generator.py` — prepends prompt_injection to system prompt
  - `bot/handlers.py` `/stats` — shows weekly_summary

Cost: ~3-5k tokens in / ~1.5k out per calibration on gpt-4o-mini ≈ $0.001.
At 1 calibration per ~30 feedbacks ≈ once a week → ~$0.05/year. Negligible.

Without OPENAI_API_KEY: gracefully no-op. Counters still update so the bot
knows when to retry after a key is added. Idea generator falls back to its
default system prompt (no injection).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .config import get_settings
from .db import get_db

log = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Recalibrate every N new feedbacks since the last calibration.
# Lowering this means fresher personalization but more LLM calls.
CALIBRATION_THRESHOLD = 20

# Cap how many recent feedbacks the LLM sees per calibration.
# 100 keeps the prompt small and weights recent preferences higher.
RECENT_FEEDBACK_LIMIT = 100


# ============================================================================
# Output schema (Pydantic)
# ============================================================================


class LearningCalibration(BaseModel):
    """LLM-generated analysis of Maksim's feedback patterns."""

    total_analyzed: int = Field(description="how many feedback rows were analyzed")
    likes: int = Field(description="count of 'like' feedback in the analyzed batch")
    dislikes: int = Field(description="count of 'dislike' feedback")
    saves: int = Field(description="count of 'save' feedback")

    top_liked_patterns: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="kebab-case patterns Maksim likes "
                    "(e.g. 'dev-tools', 'agentic-rag', 'short-mvps')",
    )
    top_disliked_patterns: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="kebab-case patterns to avoid "
                    "(e.g. 'crypto', 'regulated-vertical', 'long-mvps')",
    )

    prompt_injection_ideas: str = Field(
        max_length=700,
        description=(
            "Concrete instruction for the next idea_generator call. "
            "150-400 chars typical. Example: 'Maksim's preferences (last 30 "
            "feedbacks): strongly likes dev-tools, AI infra, RAG/LangGraph "
            "stack. Prefers MVPs ≤4 weeks. Avoids crypto-related ideas "
            "(rejected 4/4) and regulated verticals. Top-quality signal "
            "sources: HackerNews, r/MachineLearning.'"
        ),
    )

    weekly_summary: str = Field(
        max_length=800,
        description=(
            "3-5 line Telegram summary for the /stats command. Use emojis "
            "and concrete numbers. Example: '📊 Last 30 feedbacks: 12 liked, "
            "14 rejected, 4 saved.\n💡 Top liked: dev-tools (5/6), "
            "agentic-rag (3/3).\n❌ Rejected most: crypto (0/4).\n"
            "🛠 Stack preference: RAG + LangGraph get +20% love.'"
        ),
    )


# ============================================================================
# Counter / DB helpers
# ============================================================================


async def total_feedback_count() -> int:
    """How many feedback rows ever recorded."""
    async with get_db() as conn:
        async with conn.execute("SELECT COUNT(*) FROM feedback") as cur:
            row = await cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


async def feedbacks_at_last_calibration() -> int:
    """Snapshot of total count at the last successful calibration. 0 if never."""
    async with get_db() as conn:
        async with conn.execute(
            "SELECT value FROM learning_weights WHERE key = 'feedbacks_at_last_calibration'"
        ) as cur:
            row = await cur.fetchone()
    if not row or row[0] is None:
        return 0
    return int(row[0])


async def calibrations_run_count() -> int:
    async with get_db() as conn:
        async with conn.execute(
            "SELECT value FROM learning_weights WHERE key = 'calibrations_run'"
        ) as cur:
            row = await cur.fetchone()
    if not row or row[0] is None:
        return 0
    return int(row[0])


async def last_calibration_at() -> str:
    async with get_db() as conn:
        async with conn.execute(
            "SELECT text_value FROM learning_weights WHERE key = 'last_calibration_at'"
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row and row[0] else ""


async def should_recalibrate() -> bool:
    """True if (total - last) >= threshold AND total > 0."""
    total = await total_feedback_count()
    if total == 0:
        return False
    last = await feedbacks_at_last_calibration()
    return (total - last) >= CALIBRATION_THRESHOLD


async def recent_feedbacks(limit: int = RECENT_FEEDBACK_LIMIT) -> list[dict]:
    """Most recent feedback rows for calibration input."""
    async with get_db() as conn:
        async with conn.execute(
            """SELECT item_type, item_id, feedback_type, reason, item_snapshot, created_at
               FROM feedback
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_prompt_injection_ideas() -> str:
    """Read current prompt_injection text. Empty string if never calibrated."""
    async with get_db() as conn:
        async with conn.execute(
            "SELECT text_value FROM learning_weights WHERE key = 'prompt_injection_ideas'"
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row and row[0] else ""


async def get_weekly_summary() -> str:
    """Read current weekly summary text for /stats."""
    async with get_db() as conn:
        async with conn.execute(
            "SELECT text_value FROM learning_weights WHERE key = 'weekly_summary'"
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row and row[0] else ""


async def get_manual_preferences() -> str:
    """Manual preference text set by Maksim via /preferences command.

    Coexists with the auto-learned `prompt_injection_ideas` — both are
    appended to the idea_generator system prompt. Manual is read first so
    explicit user wishes always win over inferred patterns.

    Returns empty string if never set.
    """
    async with get_db() as conn:
        async with conn.execute(
            "SELECT text_value FROM learning_weights WHERE key = 'manual_preferences'"
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row and row[0] else ""


async def set_manual_preferences(text: str) -> None:
    """Persist Maksim's manual preference text (or empty to clear)."""
    cleaned = (text or "").strip()
    await _upsert_weight(
        "manual_preferences",
        None,
        cleaned if cleaned else None,
        rationale="manual /preferences command",
    )


async def _upsert_weight(
    key: str, value: float | None, text_value: str | None, rationale: str
) -> None:
    """Single-row upsert helper."""
    now = datetime.now(timezone.utc).isoformat()
    async with get_db() as conn:
        await conn.execute(
            """INSERT INTO learning_weights (key, value, text_value, rationale, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                   value = excluded.value,
                   text_value = excluded.text_value,
                   rationale = excluded.rationale,
                   updated_at = excluded.updated_at""",
            (key, value, text_value, rationale, now),
        )
        await conn.commit()


async def save_calibration(cal: LearningCalibration) -> None:
    """Persist the calibration result atomically (single transaction)."""
    now = datetime.now(timezone.utc).isoformat()
    total = await total_feedback_count()

    async with get_db() as conn:
        upserts = [
            ("prompt_injection_ideas", None, cal.prompt_injection_ideas,
             f"calibration LLM, {cal.total_analyzed} feedbacks"),
            ("weekly_summary", None, cal.weekly_summary,
             f"calibration LLM, {cal.total_analyzed} feedbacks"),
            ("feedbacks_at_last_calibration", float(total), str(total),
             f"snapshot at {now}"),
            ("last_calibration_at", None, now, "ISO timestamp"),
        ]
        for key, value, text_value, rationale in upserts:
            await conn.execute(
                """INSERT INTO learning_weights (key, value, text_value, rationale, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                       value = excluded.value,
                       text_value = excluded.text_value,
                       rationale = excluded.rationale,
                       updated_at = excluded.updated_at""",
                (key, value, text_value, rationale, now),
            )

        # Increment calibrations_run counter via UPSERT
        await conn.execute(
            """INSERT INTO learning_weights (key, value, text_value, rationale, updated_at)
               VALUES ('calibrations_run', 1, '1', 'first calibration', ?)
               ON CONFLICT(key) DO UPDATE SET
                   value = COALESCE(learning_weights.value, 0) + 1,
                   text_value = CAST(CAST(COALESCE(learning_weights.value, 0) + 1 AS INTEGER) AS TEXT),
                   updated_at = excluded.updated_at""",
            (now,),
        )
        await conn.commit()


# ============================================================================
# LLM calibration call
# ============================================================================


CALIBRATION_SYSTEM = """\
You are ORACLE's learning calibrator for Maksim, a Python AI engineer building
solo MVPs (2-6 weeks each, RAG/LangGraph stack).

You receive Maksim's recent feedback on business ideas and investment signals
from his Telegram bot. Each feedback row contains:
  - item_type: 'idea' or 'investment_signal'
  - feedback_type: 'like' / 'dislike' / 'save' / 'build' / 'deep_dive'
  - reason: dislike reason (competitors / customer / revenue / complex /
            market / timing) — only set for dislikes on ideas
  - item_snapshot: JSON of the full BusinessIdea or InvestmentSignal at the
                   time of feedback (so weights work even if the item has
                   since evolved)

WEIGHT BY FEEDBACK TYPE (Maksim's signal strength):
  - 'build'   = 5x  (highest — he's actually going to ship this)
  - 'save'    = 3x  (saved for later — strong interest)
  - 'like'    = 2x  (positive reaction)
  - 'deep_dive' = 1.5x  (curiosity, not commitment)
  - 'dislike' = 2x negative (with reason → tells us what to avoid)
When inferring preferences, weight clusters of 'build' or 'save' MUCH
heavier than 'like'. A single 'build' is worth ~3 'like's of evidence.

YOUR JOB: extract patterns and produce a `prompt_injection_ideas` text that
will steer FUTURE idea generation toward what Maksim actually likes.

ANALYZE for ideas:
1. CATEGORIES — dev-tools, AI infra, agentic apps, vertical SaaS, dev-tools,
   consumer SaaS, etc.
2. TECH STACKS — RAG, LangGraph, Python web stack, embeddings, voice agents
3. LIFECYCLE STAGES — does he prefer EMERGING (high upside) or GROWING (best
   time to build) or PEAK (late but possible)?
4. mvp_weeks RANGES — short (≤4w), medium (5-8w), long (9-12w)
5. DISLIKE REASONS — these are the most informative signals. If someone
   rejects 5 ideas with reason='competitors', Maksim cares about defensibility.
6. SIGNAL SOURCES — which sources fed liked vs disliked ideas

ANALYZE for investments (secondary):
1. Asset classes (crypto, equities, commodities, forex, macro)
2. Signal types (Macro Momentum, Defensive Hedge, etc.)
3. Timeframes

OUTPUT REQUIREMENTS:

`prompt_injection_ideas` (~150-500 chars) — a CONCRETE instruction for the
next idea_generator. Frame it as "Maksim's preferences (last N feedbacks):
strongly likes X, prefers Y, avoids Z." Be specific. Cite numbers where
possible.

`weekly_summary` (~3-5 lines) — Telegram-friendly summary for the /stats
command. Use emojis (📊 💡 ❌ 🛠 📡) and concrete numbers. Be honest about
small samples.

CRITICAL — small-sample honesty:
- If total_analyzed < 10, explicitly write "insufficient data — preferences
  unclear yet" in the weekly_summary, and produce a MINIMAL prompt injection
  like "Insufficient feedback data so far. Continue ORACLE's default profile."
- If a category has only 1-2 data points, do NOT make strong claims about it.
  Use language like "preliminary signal" or "early lean toward".
- Don't invent patterns from coincidences. Only call out preferences with at
  least 3 data points reinforcing them.

Respond ONLY with valid JSON matching the schema.
"""


def format_feedbacks_for_llm(feedbacks: list[dict]) -> str:
    """Compact representation of feedback rows for the calibration LLM."""
    lines: list[str] = []
    for i, fb in enumerate(feedbacks, start=1):
        try:
            snapshot = json.loads(fb["item_snapshot"]) if fb["item_snapshot"] else {}
        except Exception:
            snapshot = {}

        item_type = fb.get("item_type", "?")
        ftype = (fb.get("feedback_type") or "?").upper()
        reason = fb.get("reason") or ""
        marker = f"{ftype}:{reason}" if reason else ftype

        if item_type == "idea":
            title = snapshot.get("title", "(no title)")[:80]
            stage = snapshot.get("lifecycle_stage", "?")
            weeks = snapshot.get("mvp_weeks", "?")
            stack = ", ".join((snapshot.get("mvp_stack") or [])[:5])
            sources = ", ".join((snapshot.get("signal_sources") or [])[:4])
            confidence = snapshot.get("confidence", "?")
            lines.append(f"#{i} [{marker}] IDEA: {title}")
            lines.append(f"    stage={stage} weeks={weeks} conf={confidence}")
            if stack:
                lines.append(f"    stack={stack}")
            if sources:
                lines.append(f"    sources={sources}")
        else:  # investment_signal
            asset = snapshot.get("asset", "?")
            sig_type = snapshot.get("signal_type", "?")
            timeframe = snapshot.get("timeframe", "?")
            strength = snapshot.get("strength", "?")
            lines.append(f"#{i} [{marker}] INVESTMENT: {asset}")
            lines.append(f"    type={sig_type} timeframe={timeframe} strength={strength}")
    return "\n".join(lines)


async def calibrate_from_feedback() -> LearningCalibration | None:
    """Run one calibration LLM call. Returns None on error or no key."""
    settings = get_settings()
    from .observability import has_llm_credentials  # noqa: PLC0415
    if not has_llm_credentials():
        log.info("learning: no LLM credentials — calibration skipped")
        return None

    feedbacks = await recent_feedbacks(RECENT_FEEDBACK_LIMIT)
    if not feedbacks:
        log.info("learning: no feedback to analyze")
        return None

    try:
        from .observability import get_openai_client, log_llm_usage  # noqa: PLC0415
    except ImportError:
        log.warning("learning: observability module unavailable")
        return None

    try:
        client = get_openai_client(agent="learning_calibration")
    except RuntimeError as e:
        log.error("learning: %s", e)
        return None

    blob = format_feedbacks_for_llm(feedbacks)
    user_msg = (
        f"=== {len(feedbacks)} most recent feedbacks ===\n"
        f"{blob}\n\n"
        f"Analyze the patterns and produce prompt_injection_ideas + "
        f"weekly_summary. If sample is small (<10), say so."
    )

    log.info(
        "learning: calling %s with %d feedbacks (~%d KB)",
        settings.openai_model_light,
        len(feedbacks),
        len(user_msg) // 1024,
    )

    try:
        response = await client.beta.chat.completions.parse(
            model=settings.openai_model_light,  # gpt-4o-mini — analysis is cheap
            messages=[
                {"role": "system", "content": CALIBRATION_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=LearningCalibration,
            temperature=0.2,  # grounded analysis, not creative
        )
    except Exception as e:  # noqa: BLE001
        log.error("learning: LLM call failed: %s", e)
        return None

    await log_llm_usage("learning_calibration", response)

    parsed = response.choices[0].message.parsed
    if not parsed:
        log.error("learning: LLM returned no parsed output")
        return None

    await save_calibration(parsed)

    usage = response.usage
    if usage:
        log.info(
            "learning: calibration done · %d analyzed (%d like / %d dislike / %d save) · "
            "in=%d out=%d tokens (~$0.001)",
            parsed.total_analyzed, parsed.likes, parsed.dislikes, parsed.saves,
            usage.prompt_tokens, usage.completion_tokens,
        )
    return parsed


async def maybe_calibrate() -> LearningCalibration | None:
    """Threshold-gated calibration. Called from bot's _save_feedback.

    Returns the calibration result if it ran, None otherwise. Safe to await
    in fire-and-forget tasks because it never raises (catches all errors).
    """
    try:
        if not await should_recalibrate():
            return None
        log.info("learning: threshold reached, running calibration")
        return await calibrate_from_feedback()
    except Exception as e:  # noqa: BLE001
        log.error("learning: maybe_calibrate failed: %s", e)
        return None


# ============================================================================
# Standalone CLI: `uv run python -m oracle.learning [--force]`
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
        from .db import init_db
        await init_db()

        total = await total_feedback_count()
        last = await feedbacks_at_last_calibration()
        runs = await calibrations_run_count()
        last_at = await last_calibration_at()
        pending = max(0, total - last)

        print()
        print("=" * 70)
        print("ORACLE learning system status")
        print("=" * 70)
        print(f"  Total feedbacks ever:          {total}")
        print(f"  At last calibration:           {last}")
        print(f"  New since last calibration:    {pending}")
        print(f"  Threshold for auto-calibrate:  {CALIBRATION_THRESHOLD}")
        print(f"  Calibrations run ever:         {runs}")
        print(f"  Last calibration timestamp:    {last_at or '(never)'}")
        print(f"  Should recalibrate now:        {await should_recalibrate()}")

        injection = await get_prompt_injection_ideas()
        summary = await get_weekly_summary()
        print()
        print("=" * 70)
        print("Current prompt_injection_ideas")
        print("=" * 70)
        print(f"  {injection or '(empty — never calibrated)'}")
        print()
        print("=" * 70)
        print("Current weekly_summary")
        print("=" * 70)
        print(summary or "(empty — never calibrated)")

        if "--force" in sys.argv:
            print()
            print("=" * 70)
            print("Force-running calibration NOW")
            print("=" * 70)
            result = await calibrate_from_feedback()
            if result:
                print(f"\n✓ Calibration complete:")
                print(f"  Analyzed: {result.total_analyzed}")
                print(f"  Likes: {result.likes} · Dislikes: {result.dislikes} · Saves: {result.saves}")
                print(f"\nNew prompt_injection_ideas:")
                print(f"  {result.prompt_injection_ideas}")
                print(f"\nNew weekly_summary:")
                print(result.weekly_summary)
            else:
                print("\n✗ Calibration did not run (no key, no feedback, or LLM error)")

    asyncio.run(_main())
