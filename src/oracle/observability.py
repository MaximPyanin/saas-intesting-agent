"""Observability + cost tracking — Step 17.

Wraps OpenAI client with optional Langfuse instrumentation for per-call
traces, token counts, costs, and latency. Also maintains a local
`llm_cost_log` DB table (Step 2 schema v3) so the bot can show cumulative
cost in /stats and /cost without hitting the Langfuse API.

Langfuse integration is LAZY + OPTIONAL:
  - Without LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY:
    returns vanilla openai.AsyncOpenAI, no tracing
  - With keys + langfuse package installed:
    returns langfuse.openai.AsyncOpenAI drop-in replacement, all calls
    auto-traced to Langfuse cloud (free tier) or self-hosted instance

Local cost logging ALWAYS runs regardless of Langfuse state — so /stats
and /cost commands work even without Langfuse.

Usage in an agent:

    from ..observability import get_openai_client, log_llm_usage

    client = get_openai_client(agent="synthesizer")
    response = await client.beta.chat.completions.parse(...)
    await log_llm_usage("synthesizer", response)
    # ... parse response

The agent name threads through as a Langfuse `metadata.agent` tag so
traces are grouped by agent in the Langfuse UI. Local DB rows are also
tagged for SQL-based aggregation.

Run ID (LangGraph thread_id from main.run_once) is injected via a
contextvar so agents don't need to thread it manually through their
signatures.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from .config import get_settings
from .db import get_db

log = logging.getLogger(__name__)


# ============================================================================
# Run ID contextvar — set by main.run_once() at the start of each graph run
# ============================================================================

_current_run_id: ContextVar[str] = ContextVar("_current_run_id", default="")


def set_run_id(run_id: str) -> None:
    """Called by main.run_once() before graph invocation."""
    _current_run_id.set(run_id)


def get_run_id() -> str:
    """Agents use this to tag their LLM call logs with the current run."""
    return _current_run_id.get()


# ============================================================================
# Model pricing (USD per 1M tokens) — April 2026 approximations
# ============================================================================

# Update these as OpenAI changes pricing. Langfuse has its own authoritative
# model catalog for Langfuse-side cost display; this dict is for LOCAL DB
# logging that powers /stats and /cost.
#
# Any model not listed here gets cost=0.0 with a WARNING log — just add it.

MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1m_usd, output_per_1m_usd)
    "gpt-4o":             (2.50, 10.00),
    "gpt-4o-2024-08-06":  (2.50, 10.00),
    "gpt-4o-2024-11-20":  (2.50, 10.00),
    "gpt-4o-mini":        (0.15,  0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4-turbo":        (10.00, 30.00),
    "gpt-4-turbo-2024-04-09": (10.00, 30.00),
    "gpt-3.5-turbo":      (0.50,  1.50),
    "gpt-5":              (5.00, 20.00),
    "gpt-5-mini":         (0.25,  1.00),
    "gpt-5-chat":             (1.25, 10.00),
    "gpt-5-chat-2025-10-03":  (1.25, 10.00),
    "gpt-4.1":                (2.00,  8.00),
    "gpt-4.1-2025-04-14":     (2.00,  8.00),
    "gpt-4.1-mini":              (0.40, 1.60),
    "gpt-4.1-mini-2025-04-14":   (0.40, 1.60),
    "gpt-4.1-nano":              (0.10, 0.40),
    "gpt-4.1-nano-2025-04-14":   (0.10, 0.40),
    # GPT-5.4 family (Maksim's current Azure deployments)
    "gpt-5.4":                   (3.00, 12.00),
    "gpt-5.4-2026-03-05":        (3.00, 12.00),
    "gpt-5.4-mini":              (0.20,  0.80),
    "gpt-5.4-mini-2026-03-17":   (0.20,  0.80),
    "gpt-5.4-nano":              (0.05,  0.20),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute approximate USD cost for a single LLM call.

    Returns 0.0 if model isn't in the pricing table. Logs a WARNING in
    that case so you know to add it.
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        # Fallback: strip date suffix if present (e.g. "gpt-4o-2024-08-06" → "gpt-4o")
        base = model.rsplit("-", 3)[0] if model.count("-") >= 3 else model
        pricing = MODEL_PRICING.get(base)
        if not pricing:
            log.warning("observability: unknown model %r for cost estimation", model)
            return 0.0
    input_cost = (input_tokens / 1_000_000) * pricing[0]
    output_cost = (output_tokens / 1_000_000) * pricing[1]
    return round(input_cost + output_cost, 6)


# ============================================================================
# OpenAI client getter — returns Langfuse-wrapped or vanilla
# ============================================================================

def has_llm_credentials() -> bool:
    """True if either OpenAI direct OR Azure OpenAI is configured.

    Used by every agent at the top of its LLM call function to decide
    whether to graceful no-op (no key) or proceed. Replaces the older
    `if not settings.openai_api_key` checks so the same agents work
    against either backend without modification.
    """
    settings = get_settings()
    if settings.azure_openai_endpoint and settings.azure_openai_api_key:
        return True
    if settings.openai_api_key:
        return True
    return False


# Cache the Langfuse availability check so we don't re-probe on every call.
_langfuse_available: bool | None = None


def _try_langfuse_available() -> bool:
    """Check once whether Langfuse is installed AND configured. Cached."""
    global _langfuse_available
    if _langfuse_available is not None:
        return _langfuse_available

    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        _langfuse_available = False
        return False

    try:
        # Test import — doesn't make any API calls
        from langfuse.openai import AsyncOpenAI as _  # noqa: F401
        _langfuse_available = True
        log.info("observability: Langfuse OpenAI wrapper enabled")
        return True
    except ImportError:
        log.info("observability: langfuse package not installed, using vanilla OpenAI")
        _langfuse_available = False
        return False


def get_openai_client(agent: str = "") -> Any:
    """Return an OpenAI-compatible AsyncClient — Azure or OpenAI direct.

    Backend selection (auto-detect):
      - If `azure_openai_endpoint` is set in settings → Azure OpenAI / Azure
        AI Foundry, returns AsyncAzureOpenAI. The model name in
        `parse(model=...)` calls is interpreted as the Azure DEPLOYMENT NAME.
      - Else if `openai_api_key` is set → OpenAI direct, returns AsyncOpenAI.
      - Else → RuntimeError (agent should check this BEFORE calling here).

    Both AsyncOpenAI and AsyncAzureOpenAI expose IDENTICAL interfaces, so
    agent code is the same regardless of backend.

    Langfuse: drop-in `langfuse.openai.AsyncOpenAI` works with the OpenAI
    direct path. For Azure path, Langfuse instrumentation is currently
    skipped — local cost logging in `llm_cost_log` table still runs and
    powers `/cost` and `/stats`.

    Raises:
        RuntimeError: if neither key is set.
    """
    settings = get_settings()

    # Hard timeout for every LLM call — prevents the bot from hanging when
    # Azure / OpenAI lazes on a complex structured-output request. 120 s is
    # generous (most calls finish in 5-15 s) but short enough to fail fast.
    # On timeout the SDK raises APITimeoutError which agent code catches and
    # gracefully degrades (e.g. returns empty scenarios list).
    LLM_TIMEOUT_SECONDS = 120.0

    # ---------- Azure OpenAI / Azure AI Foundry path ----------
    if settings.azure_openai_endpoint:
        if not settings.azure_openai_api_key:
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT is set but AZURE_OPENAI_API_KEY is missing"
            )
        from openai import AsyncAzureOpenAI  # noqa: PLC0415
        return AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=1,  # don't retry hung Azure requests for 5 minutes
        )

    # ---------- OpenAI direct path ----------
    if not settings.openai_api_key:
        raise RuntimeError(
            "Neither AZURE_OPENAI_ENDPOINT nor OPENAI_API_KEY is set — "
            "cannot build LLM client"
        )

    if _try_langfuse_available():
        from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI  # noqa: PLC0415
        return LangfuseAsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=1,
        )

    from openai import AsyncOpenAI  # noqa: PLC0415
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=1,
    )


# ============================================================================
# LLM call logging — persists to llm_cost_log table
# ============================================================================


async def log_llm_usage(
    agent: str,
    response: Any,
    latency_ms: int = 0,
) -> None:
    """Persist one LLM call's usage to llm_cost_log. Never raises.

    Args:
        agent: agent name — e.g. "synthesizer", "idea_generator",
            "idea_generator_improve", "critic", "validator",
            "investment_analyzer", "learning_calibration"
        response: OpenAI response object (must have .usage and .model)
        latency_ms: optional — call latency in ms if the caller measured it

    This function catches ALL exceptions internally so a logging failure
    never crashes an agent. Observability should not break production.
    """
    try:
        model = getattr(response, "model", "unknown")
        usage = getattr(response, "usage", None)
        if not usage:
            return
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = estimate_cost_usd(model, input_tokens, output_tokens)

        async with get_db() as conn:
            await conn.execute(
                """INSERT INTO llm_cost_log
                   (run_id, agent, model, input_tokens, output_tokens, cost_usd,
                    latency_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    get_run_id(),
                    agent,
                    model,
                    input_tokens,
                    output_tokens,
                    cost,
                    latency_ms,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await conn.commit()

        log.debug(
            "observability: logged %s · %s · in=%d out=%d · $%.5f",
            agent, model, input_tokens, output_tokens, cost,
        )
    except Exception as e:  # noqa: BLE001 — observability never crashes agents
        log.debug("observability: log_llm_usage failed: %s", e)


# ============================================================================
# Cost query helpers — read by /stats and /cost bot commands
# ============================================================================


async def get_cost_summary(days: int = 30) -> dict:
    """Return cumulative cost breakdown for the last N days.

    Returns a dict:
        {
            "days": 30,
            "total_cost_usd": 12.45,
            "total_calls": 120,
            "total_input_tokens": 450_000,
            "total_output_tokens": 80_000,
            "by_agent": [(name, calls, cost), ...],    # sorted by cost desc
            "by_day": [(YYYY-MM-DD, cost), ...],       # most recent first
        }
    """
    async with get_db() as conn:
        async with conn.execute(
            f"""SELECT COUNT(*), COALESCE(SUM(cost_usd), 0),
                       COALESCE(SUM(input_tokens), 0),
                       COALESCE(SUM(output_tokens), 0)
                FROM llm_cost_log
                WHERE created_at >= datetime('now', '-{int(days)} days')"""
        ) as cur:
            row = await cur.fetchone()
        total_calls = int(row[0] or 0)
        total_cost = float(row[1] or 0.0)
        total_in = int(row[2] or 0)
        total_out = int(row[3] or 0)

        async with conn.execute(
            f"""SELECT agent, COUNT(*), COALESCE(SUM(cost_usd), 0)
                FROM llm_cost_log
                WHERE created_at >= datetime('now', '-{int(days)} days')
                GROUP BY agent
                ORDER BY 3 DESC"""
        ) as cur:
            by_agent = [(r[0], int(r[1] or 0), float(r[2] or 0.0))
                        for r in await cur.fetchall()]

        async with conn.execute(
            f"""SELECT substr(created_at, 1, 10) AS day, COALESCE(SUM(cost_usd), 0)
                FROM llm_cost_log
                WHERE created_at >= datetime('now', '-{int(days)} days')
                GROUP BY day
                ORDER BY day DESC LIMIT 10"""
        ) as cur:
            by_day = [(r[0], float(r[1] or 0.0)) for r in await cur.fetchall()]

    return {
        "days": days,
        "total_cost_usd": round(total_cost, 4),
        "total_calls": total_calls,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "by_agent": by_agent,
        "by_day": by_day,
    }


async def get_last_run_cost(run_id: str) -> dict:
    """Cost summary for one specific graph run (by thread_id)."""
    async with get_db() as conn:
        async with conn.execute(
            """SELECT COUNT(*), COALESCE(SUM(cost_usd), 0),
                      COALESCE(SUM(input_tokens), 0),
                      COALESCE(SUM(output_tokens), 0)
               FROM llm_cost_log
               WHERE run_id = ?""",
            (run_id,),
        ) as cur:
            row = await cur.fetchone()
        async with conn.execute(
            """SELECT agent, COUNT(*), COALESCE(SUM(cost_usd), 0)
               FROM llm_cost_log
               WHERE run_id = ?
               GROUP BY agent
               ORDER BY 3 DESC""",
            (run_id,),
        ) as cur:
            by_agent = [(r[0], int(r[1] or 0), float(r[2] or 0.0))
                        for r in await cur.fetchall()]

    return {
        "run_id": run_id,
        "total_calls": int(row[0] or 0),
        "total_cost_usd": round(float(row[1] or 0.0), 4),
        "total_input_tokens": int(row[2] or 0),
        "total_output_tokens": int(row[3] or 0),
        "by_agent": by_agent,
    }


# ============================================================================
# Langfuse flush — called at end of each graph run to flush buffered events
# ============================================================================


async def flush_langfuse() -> None:
    """Flush any buffered Langfuse events. No-op without Langfuse.

    Langfuse buffers events in-memory and sends them async in background.
    Calling flush() at the end of a graph run ensures traces are written
    before the process exits (critical for short-lived container runs).
    """
    if not _try_langfuse_available():
        return
    try:
        from langfuse import Langfuse  # noqa: PLC0415
        lf = Langfuse()
        lf.flush()
        log.debug("observability: langfuse flushed")
    except Exception as e:  # noqa: BLE001
        log.debug("observability: langfuse flush failed: %s", e)


# ============================================================================
# Standalone CLI: `uv run python -m oracle.observability`
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

        print()
        print("=" * 70)
        print("ORACLE observability status")
        print("=" * 70)
        print(f"Langfuse available:  {_try_langfuse_available()}")
        print(f"Current run_id:      {get_run_id() or '(none)'}")

        print()
        print("=" * 70)
        print("Cost summary — last 30 days")
        print("=" * 70)
        summary = await get_cost_summary(days=30)
        print(f"Total calls:    {summary['total_calls']}")
        print(f"Total cost:     ${summary['total_cost_usd']:.4f}")
        print(f"Input tokens:   {summary['total_input_tokens']:,}")
        print(f"Output tokens:  {summary['total_output_tokens']:,}")
        print()
        if summary["by_agent"]:
            print("By agent:")
            for agent, calls, cost in summary["by_agent"]:
                print(f"  {agent:25s} {calls:4d} calls · ${cost:.4f}")
        else:
            print("(no LLM calls logged yet)")
        print()
        if summary["by_day"]:
            print("By day (last 10):")
            for day, cost in summary["by_day"]:
                print(f"  {day}  ${cost:.4f}")

        print()
        print("=" * 70)
        print("Cost estimation test")
        print("=" * 70)
        print(f"gpt-4o: 10k in + 2k out = ${estimate_cost_usd('gpt-4o', 10000, 2000):.5f}")
        print(f"gpt-4o-mini: 5k in + 1k out = ${estimate_cost_usd('gpt-4o-mini', 5000, 1000):.5f}")

    asyncio.run(_main())
