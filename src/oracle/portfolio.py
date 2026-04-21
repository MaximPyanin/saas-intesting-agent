"""Portfolio module — Maksim's actual holdings + live P&L (Step 20).

Stores his real positions in oracle_data.db.portfolio_holdings and computes
unrealized P&L on demand using the live market_data dict from the market
collector (Step 4). Read by:

  - investment_analyzer (Step 13): injects portfolio context into the LLM
    prompt so scenarios are PERSONALIZED (e.g. "you hold 0.5 BTC at $52k,
    current $63k → +21% unrealized; here's what the news means for YOUR
    position specifically")

  - Telegram bot (Step 8): /portfolio command renders the table with P&L,
    /add_holding wizard adds new positions, /remove_holding deletes them.

DESIGN NOTES:
- One row per asset_label (no FIFO lots) — when re-buying, the avg_buy_price
  is recomputed weighted-average style. Simple, fits a personal portfolio
  better than a brokerage ledger.
- All prices internally normalized to USD because that's how market_data
  publishes everything. Currency hint is stored only for display.
- "asset_label" MUST match a label in market.py catalogs (e.g. "BTC", "NVDA",
  "GOLD", "EURPLN"). validate_asset_label() helps the bot reject typos.
- Cash positions (asset_class='cash') get price=1.0, P&L always 0 — they're
  shown for completeness but don't affect signals.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .db import get_db

log = logging.getLogger(__name__)


# ============================================================================
# Asset universe (single source of truth — mirrors market.py catalogs)
# ============================================================================

# Lazy import to avoid yfinance import cost when only the bot is running
def _get_known_assets() -> dict[str, str]:
    """Return {asset_label: asset_class} for every tracked symbol.

    Uses the catalogs from market.py so adding a new ticker there auto-extends
    the portfolio's allowed asset universe.
    """
    from .agents.market import (  # noqa: PLC0415
        COMMODITY_SYMBOLS,
        CRYPTO_COINGECKO,
        DRONE_DEFENSE_SYMBOLS,
        EQUITY_SYMBOLS,
        FOREX_SYMBOLS,
        INDEX_SYMBOLS,
        NUCLEAR_SYMBOLS,
        STOCK_SYMBOLS,
        TRUMP_POLITICAL_SYMBOLS,
    )

    universe: dict[str, str] = {}
    for label in EQUITY_SYMBOLS:
        universe[label] = "etf"
    for label in STOCK_SYMBOLS:
        universe[label] = "stock"
    for label in NUCLEAR_SYMBOLS:
        universe[label] = "nuclear"
    for label in DRONE_DEFENSE_SYMBOLS:
        universe[label] = "drones_defense"
    for label in TRUMP_POLITICAL_SYMBOLS:
        universe[label] = "trump_political"
    for label in COMMODITY_SYMBOLS:
        universe[label] = "commodity"
    for label in FOREX_SYMBOLS:
        universe[label] = "forex"
    for label in INDEX_SYMBOLS:
        universe[label] = "index"
    for label in CRYPTO_COINGECKO:
        universe[label] = "crypto"
    universe["CASH_USD"] = "cash"
    universe["CASH_EUR"] = "cash"
    universe["CASH_PLN"] = "cash"
    universe["CASH_BYN"] = "cash"
    return universe


def validate_asset_label(label: str) -> tuple[bool, str]:
    """Return (is_valid, asset_class_or_error_message)."""
    universe = _get_known_assets()
    if label in universe:
        return True, universe[label]
    return False, (
        f"Unknown asset '{label}'. Tracked assets: "
        + ", ".join(sorted(universe.keys()))
    )


# ============================================================================
# CRUD
# ============================================================================


async def add_or_update_holding(
    asset_label: str,
    quantity: float,
    buy_price_usd: float,
    *,
    notes: str | None = None,
    currency: str = "USD",
) -> dict[str, Any]:
    """Insert a new holding OR weighted-average merge into an existing one.

    Returns the resulting row as a dict.

    Re-add semantics (weighted average):
        existing: 0.5 BTC @ $50k
        add:      0.2 BTC @ $60k
        result:   0.7 BTC @ ((0.5*50k + 0.2*60k) / 0.7) = $52,857
    """
    ok, asset_class_or_err = validate_asset_label(asset_label)
    if not ok:
        raise ValueError(asset_class_or_err)
    asset_class = asset_class_or_err
    if quantity <= 0:
        raise ValueError("quantity must be > 0")
    if buy_price_usd < 0:
        raise ValueError("buy_price_usd must be >= 0")

    now = datetime.now(timezone.utc).isoformat()

    async with get_db() as conn:
        cur = await conn.execute(
            "SELECT id, quantity, avg_buy_price FROM portfolio_holdings WHERE asset_label = ?",
            (asset_label,),
        )
        row = await cur.fetchone()

        if row is None:
            await conn.execute(
                """
                INSERT INTO portfolio_holdings
                  (asset_label, asset_class, quantity, avg_buy_price,
                   currency, notes, added_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_label,
                    asset_class,
                    quantity,
                    buy_price_usd,
                    currency,
                    notes,
                    now,
                    now,
                ),
            )
            await conn.commit()
            log.info(
                "portfolio: added %s — %.6f units @ $%.2f",
                asset_label, quantity, buy_price_usd,
            )
        else:
            old_qty = float(row["quantity"])
            old_price = float(row["avg_buy_price"])
            new_qty = old_qty + quantity
            new_avg = (
                (old_qty * old_price + quantity * buy_price_usd) / new_qty
                if new_qty > 0
                else buy_price_usd
            )
            await conn.execute(
                """
                UPDATE portfolio_holdings
                   SET quantity = ?,
                       avg_buy_price = ?,
                       notes = COALESCE(?, notes),
                       updated_at = ?
                 WHERE asset_label = ?
                """,
                (new_qty, new_avg, notes, now, asset_label),
            )
            await conn.commit()
            log.info(
                "portfolio: merged %s — was %.6f@$%.2f, now %.6f@$%.2f",
                asset_label, old_qty, old_price, new_qty, new_avg,
            )

    return await get_holding(asset_label) or {}


async def get_holding(asset_label: str) -> dict[str, Any] | None:
    async with get_db() as conn:
        cur = await conn.execute(
            "SELECT * FROM portfolio_holdings WHERE asset_label = ?",
            (asset_label,),
        )
        row = await cur.fetchone()
        return dict(row) if row else None


async def list_holdings() -> list[dict[str, Any]]:
    async with get_db() as conn:
        cur = await conn.execute(
            "SELECT * FROM portfolio_holdings ORDER BY asset_class, asset_label"
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


async def remove_holding(asset_label: str) -> bool:
    """Delete a holding by label. Returns True if a row was removed."""
    async with get_db() as conn:
        cur = await conn.execute(
            "DELETE FROM portfolio_holdings WHERE asset_label = ?",
            (asset_label,),
        )
        await conn.commit()
        removed = cur.rowcount > 0
    if removed:
        log.info("portfolio: removed %s", asset_label)
    return removed


# ============================================================================
# Live P&L computation (uses market_data from the market collector)
# ============================================================================


def _lookup_current_price(asset_label: str, market_data: dict) -> float | None:
    """Find the live USD price for asset_label across all market_data buckets."""
    for bucket_key in (
        "equities",
        "stocks",
        "nuclear",
        "drones_defense",
        "trump_political",
        "commodities",
        "forex",
        "indices",
        "crypto",
    ):
        bucket = market_data.get(bucket_key) or {}
        if asset_label in bucket:
            try:
                return float(bucket[asset_label].get("price", 0)) or None
            except (TypeError, ValueError):
                return None
    return None


def compute_pnl(
    holding: dict[str, Any],
    market_data: dict,
) -> dict[str, Any]:
    """Enrich a holding row with live market_value, unrealized_pnl, pnl_pct.

    Returns a NEW dict (does not mutate the input).
    """
    enriched = dict(holding)
    qty = float(holding.get("quantity", 0) or 0)
    avg = float(holding.get("avg_buy_price", 0) or 0)
    asset_class = holding.get("asset_class", "")
    label = holding.get("asset_label", "")

    if asset_class == "cash":
        enriched["current_price"] = 1.0
        enriched["market_value_usd"] = qty
        enriched["cost_basis_usd"] = qty
        enriched["unrealized_pnl_usd"] = 0.0
        enriched["pnl_pct"] = 0.0
        enriched["price_status"] = "cash"
        return enriched

    current = _lookup_current_price(label, market_data)
    cost_basis = qty * avg

    if current is None:
        enriched["current_price"] = None
        enriched["market_value_usd"] = None
        enriched["cost_basis_usd"] = cost_basis
        enriched["unrealized_pnl_usd"] = None
        enriched["pnl_pct"] = None
        enriched["price_status"] = "stale"
        return enriched

    market_value = qty * current
    pnl = market_value - cost_basis
    pnl_pct = ((current / avg - 1) * 100) if avg > 0 else 0.0

    enriched["current_price"] = current
    enriched["market_value_usd"] = market_value
    enriched["cost_basis_usd"] = cost_basis
    enriched["unrealized_pnl_usd"] = pnl
    enriched["pnl_pct"] = pnl_pct
    enriched["price_status"] = "live"
    return enriched


async def get_portfolio_with_pnl(market_data: dict) -> dict[str, Any]:
    """Return the full portfolio enriched with live P&L + totals.

    Shape:
        {
          "holdings": [ {asset_label, ..., market_value_usd, pnl_pct, ...}, ... ],
          "totals": {
            "cost_basis_usd":   ...,
            "market_value_usd": ...,
            "unrealized_pnl_usd": ...,
            "pnl_pct":          ...,
          },
          "by_class": {"crypto": {...}, "stock": {...}, ...},
        }
    """
    rows = await list_holdings()
    enriched = [compute_pnl(r, market_data) for r in rows]

    total_cost = 0.0
    total_value = 0.0
    by_class: dict[str, dict[str, float]] = {}

    for h in enriched:
        cls = h.get("asset_class", "other")
        cost = float(h.get("cost_basis_usd") or 0)
        mv = h.get("market_value_usd")
        mv_f = float(mv) if mv is not None else cost  # fall back so totals stay sane
        total_cost += cost
        total_value += mv_f
        bucket = by_class.setdefault(
            cls,
            {"cost_basis_usd": 0.0, "market_value_usd": 0.0, "count": 0},
        )
        bucket["cost_basis_usd"] += cost
        bucket["market_value_usd"] += mv_f
        bucket["count"] += 1

    total_pnl = total_value - total_cost
    total_pct = ((total_value / total_cost - 1) * 100) if total_cost > 0 else 0.0

    return {
        "holdings": enriched,
        "totals": {
            "cost_basis_usd": total_cost,
            "market_value_usd": total_value,
            "unrealized_pnl_usd": total_pnl,
            "pnl_pct": total_pct,
        },
        "by_class": by_class,
    }


# ============================================================================
# LLM-friendly formatter (used by investment_analyzer prompt injection)
# ============================================================================


def format_portfolio_for_llm(portfolio: dict[str, Any]) -> str:
    """Compact human/LLM-readable portfolio summary.

    Used by investment_analyzer to inject Maksim's positions into the prompt
    so scenarios are personalized to what he actually holds.
    """
    holdings = portfolio.get("holdings") or []
    if not holdings:
        return "(no portfolio holdings tracked yet — generic market analysis only)"

    lines: list[str] = []
    totals = portfolio.get("totals") or {}
    tot_cost = totals.get("cost_basis_usd", 0.0)
    tot_mv = totals.get("market_value_usd", 0.0)
    tot_pnl = totals.get("unrealized_pnl_usd", 0.0)
    tot_pct = totals.get("pnl_pct", 0.0)

    lines.append(
        f"PORTFOLIO TOTALS: ${tot_mv:,.0f} value (cost ${tot_cost:,.0f}, "
        f"unrealized {'+' if tot_pnl >= 0 else ''}{tot_pnl:,.0f} = "
        f"{'+' if tot_pct >= 0 else ''}{tot_pct:.1f}%)"
    )
    lines.append("")
    lines.append("POSITIONS:")
    for h in holdings:
        label = h["asset_label"]
        cls = h["asset_class"]
        qty = h["quantity"]
        avg = h["avg_buy_price"]
        cur = h.get("current_price")
        mv = h.get("market_value_usd")
        pnl_pct = h.get("pnl_pct")
        status = h.get("price_status", "?")
        notes = h.get("notes") or ""

        if status == "live" and pnl_pct is not None:
            sign = "+" if pnl_pct >= 0 else ""
            lines.append(
                f"  {label} [{cls}]: {qty:g} @ ${avg:,.2f} avg → "
                f"now ${cur:,.2f} · MV ${mv:,.0f} · {sign}{pnl_pct:.1f}%"
                + (f" ({notes})" if notes else "")
            )
        elif status == "cash":
            lines.append(f"  {label} [cash]: ${qty:,.2f}" + (f" ({notes})" if notes else ""))
        else:
            lines.append(
                f"  {label} [{cls}]: {qty:g} @ ${avg:,.2f} avg "
                f"(no live price this run)"
                + (f" ({notes})" if notes else "")
            )

    return "\n".join(lines)


# ============================================================================
# Standalone CLI: `uv run python -m oracle.portfolio`
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
        from .agents.market import collect_market_data
        from .db import init_db

        await init_db()

        # Demo: add a couple of toy positions, fetch live prices, render
        await add_or_update_holding("BTC",  0.15, 52_000.0, notes="long-term")
        await add_or_update_holding("NVDA", 12,    480.0,    notes="AI exposure")
        await add_or_update_holding("GOLD", 5,     2_100.0,  notes="hedge")
        await add_or_update_holding("CASH_USD", 5_000, 1.0)

        log.info("collecting live market data...")
        market_data, _ = await collect_market_data()

        portfolio = await get_portfolio_with_pnl(market_data)
        print()
        print("=" * 70)
        print(format_portfolio_for_llm(portfolio))
        print("=" * 70)

    asyncio.run(_main())
