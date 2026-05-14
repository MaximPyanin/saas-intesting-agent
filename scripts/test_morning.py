"""Quick local smoke test: simulate the 07:00 morning brief WITHOUT Telegram.

Runs the same code path as scheduler.morning_brief_job but prints the two
sections to stdout instead of sending them to Telegram. Useful for testing
the new urgent + portfolio-advice flow locally before deploying.

Usage:
    py -3 scripts/test_morning.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Windows UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


async def _main() -> None:
    from datetime import datetime, timezone

    from oracle.agents.market import collect_market_data
    from oracle.agents.portfolio_advisor import generate_morning_portfolio_advice
    from oracle.alerts import get_active_alerts_no_dedup, get_recently_fired_alerts
    from oracle.bot.views import (
        render_portfolio_morning_advice,
        render_urgent_section,
    )
    from oracle.db import init_db
    from oracle.portfolio import get_portfolio_with_pnl

    await init_db()  # ensures v5 schema
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("=" * 70)
    print("  Fetching market data ...")
    print("=" * 70)
    market_data, errs = await collect_market_data()
    if errs:
        print(f"  market errors (non-fatal): {len(errs)}")

    print()
    print("=" * 70)
    print("  SECTION A: 🚨 СРОЧНО")
    print("=" * 70)
    active = await get_active_alerts_no_dedup(market_data)
    recent = await get_recently_fired_alerts(hours=12)
    urgent = render_urgent_section(active, recent, date=today)
    print(urgent)

    print()
    print("=" * 70)
    print("  SECTION B: 💼 СОВЕТ ПО ПОРТФЕЛЮ")
    print("=" * 70)
    portfolio = await get_portfolio_with_pnl(market_data)
    advice = await generate_morning_portfolio_advice(portfolio)
    out = render_portfolio_morning_advice(advice, portfolio, date=today)
    print(out)

    print()
    print("=" * 70)
    print(f"  Done. {len(active)} active alerts · {len(recent)} overnight fires · "
          f"{len(portfolio.get('holdings') or [])} holdings")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(_main())
