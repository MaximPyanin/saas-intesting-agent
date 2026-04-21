"""ORACLE entry point — runs the graph end-to-end.

Usage:
    uv run python -m oracle.main

Step 1: every node is a no-op placeholder. Verifies that the topology compiles,
runs in the right order, and checkpoints to oracle.db. Real agents land in
Steps 4–13.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .db import init_db
from .graph import build_graph
from .observability import flush_langfuse, get_last_run_cost, set_run_id
from .state import OracleState

# Windows console (cp1251) can't print emoji/em dashes — force UTF-8.
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
log = logging.getLogger("oracle")


def initial_state() -> OracleState:
    """Empty starting state — every key explicitly populated."""
    return {
        "scout_signals": [],
        "market_data": {},
        "trend_signals": [],
        "custom_signals": [],
        "synthesized": [],
        "raw_ideas": [],
        "investment_scenarios": [],
        "reflexion_round": 0,
        "investment_reflexion_round": 0,
        "surviving_ideas": [],
        "validated": [],
        "final_digest": {},
        "errors": [],
    }


async def run_once(thread_id: str | None = None) -> dict:
    """Single end-to-end execution of the ORACLE graph.

    Args:
        thread_id: optional LangGraph checkpoint thread ID. If None, a fresh
            timestamp-based ID is used (so each run starts clean). The
            scheduler (Step 15) calls this with a timestamp per run so the
            checkpointer doesn't try to resume a completed prior run.
    """
    # Ensure domain DB schema exists before any agent tries to write to it.
    # Idempotent — safe to call on every run.
    await init_db()

    if not thread_id:
        from datetime import datetime, timezone
        thread_id = "oracle-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Step 17: seed run_id contextvar so observability.log_llm_usage can
    # tag every LLM call with the current graph run_id without threading
    # it through every agent signature.
    set_run_id(thread_id)

    workflow = build_graph()

    try:
        async with AsyncSqliteSaver.from_conn_string("oracle.db") as cp:
            graph = workflow.compile(checkpointer=cp)
            config = {"configurable": {"thread_id": thread_id}}
            result = await graph.ainvoke(initial_state(), config=config)
            log.info("ORACLE run finished (thread=%s). final_digest=%s",
                     thread_id, result["final_digest"])

            # Run cost summary — logs cumulative cost for this single run
            try:
                cost = await get_last_run_cost(thread_id)
                if cost["total_calls"]:
                    log.info(
                        "ORACLE run cost: %d LLM calls · $%.4f · "
                        "in=%d out=%d tokens",
                        cost["total_calls"],
                        cost["total_cost_usd"],
                        cost["total_input_tokens"],
                        cost["total_output_tokens"],
                    )
            except Exception as e:  # noqa: BLE001
                log.debug("main: cost summary failed: %s", e)

            return result
    finally:
        # Flush Langfuse traces before process exits (critical for containers)
        await flush_langfuse()


def main() -> None:
    """Sync wrapper for `python -m oracle.main`."""
    asyncio.run(run_once())


if __name__ == "__main__":
    main()
