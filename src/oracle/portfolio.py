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

# Custom European-listed ETFs / ETPs that Maksim holds but which aren't in
# market.py's yfinance catalogs (those track US-listed tickers only).
# We accept them as valid portfolio entries; live prices may be unavailable
# (price_status='stale'), which the LLM handles gracefully.
CUSTOM_PORTFOLIO_ASSETS: dict[str, str] = {
    "CSPX":     "etf",        # iShares Core S&P 500 UCITS (IE00B5BMR087)
    "SMH":      "etf",        # VanEck Semiconductor UCITS (IE00BMC38736) — overrides US SMH
    "NATO":     "etf",        # HANetf Future of Defence UCITS (IE000OJ5TQP4)
    "NUCL":     "etf",        # VanEck Uranium & Nuclear UCITS (IE000M7V94E1)
    "EXH1":     "etf",        # iShares STOXX Europe 600 Oil & Gas (DE000A0H08M3)
    "IB1T":     "crypto",     # iShares Bitcoin ETP (XS2940466316)
    "ETH-CORE": "crypto",     # 21Shares Ethereum Core
    "IB01":     "bond",       # iShares $ Treasury 0-1 yr UCITS (IE00B3VWN518)
    "CASH-USD": "cash",       # Cash USD on broker (XTB)
    "GOLD-PHYS": "commodity", # Physical gold (off-broker)
}


# Human-readable display names for the Telegram cards. Keep the ticker in
# parentheses so Maksim can still grep by code. Used by render_investment_card
# and any other surface that shows asset_label to a human.
ASSET_DISPLAY_NAMES: dict[str, str] = {
    "CSPX":      "S&P 500 (CSPX)",
    "SMH":       "Чипы AI (SMH)",
    "NATO":      "Оборонка (NATO)",
    "NUCL":      "Уран/Ядерка (NUCL)",
    "EXH1":      "Нефть/Газ EU (EXH1)",
    "IB1T":      "Bitcoin (IB1T)",
    "ETH-CORE":  "Ethereum (ETH-CORE)",
    "IB01":      "Короткие трежери (IB01)",
    "CASH-USD":  "Кэш USD",
    "GOLD-PHYS": "Золото физ.",
    "NVDA":      "NVIDIA (NVDA)",
    "BTC":       "Bitcoin (BTC)",
    "ETH":       "Ethereum (ETH)",
    "SOL":       "Solana (SOL)",
    "XRP":       "Ripple (XRP)",
    "GOLD":      "Золото (GOLD)",
    "SILVER":    "Серебро (SILVER)",
    "SPY":       "S&P 500 (SPY)",
    "QQQ":       "NASDAQ-100 (QQQ)",
    "XLK":       "Tech-сектор (XLK)",
    "XLE":       "Энерго-сектор (XLE)",
    "XLF":       "Финансы-сектор (XLF)",
    "VNQ":       "REITs (VNQ)",
    "OIL_WTI":   "Нефть WTI",
    "OIL_BRENT": "Нефть Brent",
    "NATGAS":    "Газ (NATGAS)",
    "COPPER":    "Медь (COPPER)",
    "WHEAT":     "Пшеница (WHEAT)",
    "VIX":       "Индекс страха (VIX)",
    "DXY":       "Индекс доллара (DXY)",
    "TNX_10Y":   "Доходность 10Y трежерис",
    "EURUSD":    "EUR/USD",
    "USDPLN":    "USD/PLN",
    "EURPLN":    "EUR/PLN",
    # Stocks Maksim sees most often
    "MSFT":      "Microsoft (MSFT)",
    "GOOGL":     "Google (GOOGL)",
    "AAPL":      "Apple (AAPL)",
    "TSLA":      "Tesla (TSLA)",
    "AMD":       "AMD",
    "META":      "Meta (META)",
    "AMZN":      "Amazon (AMZN)",
    "TSM":       "TSMC (TSM)",
    "AVGO":      "Broadcom (AVGO)",
    "NFLX":      "Netflix (NFLX)",
    "PLTR":      "Palantir (PLTR)",
    "SMCI":      "Super Micro (SMCI)",
    "ARM":       "ARM Holdings (ARM)",
    "ASML":      "ASML (литограф)",
    "MU":        "Micron (MU)",
    "VRT":       "Vertiv (VRT) — DC-охлаждение",
    "SOUN":      "SoundHound AI (SOUN)",
    "AI":        "C3.ai (AI)",
    # Nuclear / uranium
    "SMR":       "NuScale Power (SMR)",
    "OKLO":      "Oklo (OKLO)",
    "NNE":       "Nano Nuclear (NNE)",
    "LEU":       "Centrus Energy (LEU) — обогащение",
    "CCJ":       "Cameco (CCJ) — уран",
    "BWXT":      "BWX Tech (BWXT)",
    "UEC":       "Uranium Energy (UEC)",
    "URA":       "Global X Uranium ETF",
    "VST":       "Vistra (VST) — энергия для AI",
    "CEG":       "Constellation Energy (CEG)",
    # Drones / defense
    "AVAV":      "AeroVironment (AVAV) — Switchblade",
    "KTOS":      "Kratos Defense (KTOS)",
    "RCAT":      "Red Cat (RCAT) — дроны",
    "ONDS":      "Ondas (ONDS) — дрон-сети",
    "RKLB":      "Rocket Lab (RKLB) — космос/оборонка",
    "EH":        "EHang (EH) — eVTOL",
    "UMAC":      "Unusual Machines (UMAC)",
    # Trump-narrative
    "DJT":       "Trump Media (DJT)",
    "RUM":       "Rumble (RUM)",
    "PSQH":      "PublicSquare (PSQH)",
    "PHUN":      "Phunware (PHUN)",
}


def display_name(asset_label: str) -> str:
    """Friendly Russian/English name for an asset, with ticker for grep-ability.

    Falls back to the raw ticker if no mapping exists.
    """
    return ASSET_DISPLAY_NAMES.get(asset_label, asset_label)


# Tickers Maksim can add to his portfolio via bot UI.
# Includes his core ETFs + any individual stock / commodity / crypto / forex
# that ORACLE actually tracks. Removed earlier restriction: Maksim wanted
# the ➕ Add button to always show options.
TRACKABLE_PORTFOLIO_TICKERS: list[str] = [
    # Core UCITS portfolio (existing)
    "CSPX", "SMH", "NATO", "NUCL", "EXH1",
    "IB1T", "ETH-CORE", "IB01",
    "CASH-USD", "GOLD-PHYS",
    # Individual stocks tracked in market.py
    "NVDA", "MSFT", "GOOGL", "AAPL", "TSLA", "AMD", "META", "AMZN",
    "TSM", "AVGO", "NFLX", "PLTR", "SMCI", "ARM", "ASML", "MU",
    "VRT", "SOUN", "AI",
    # Nuclear / uranium
    "SMR", "OKLO", "NNE", "LEU", "CCJ", "BWXT", "UEC", "URA", "VST", "CEG",
    # Drones / defense
    "AVAV", "KTOS", "RCAT", "ONDS", "RKLB", "EH", "UMAC",
    # Trump-narrative
    "DJT", "RUM", "PSQH", "PHUN",
    # ETFs
    "SPY", "QQQ", "XLK", "XLE", "XLF", "VNQ",
    # Commodities (paper exposure)
    "GOLD", "SILVER", "OIL_WTI", "OIL_BRENT", "NATGAS", "COPPER", "WHEAT",
    # Crypto (spot)
    "BTC", "ETH", "SOL", "XRP",
    # Forex
    "EURUSD", "USDPLN", "EURPLN",
    # Cash buckets (extra currencies)
    "CASH_USD", "CASH_EUR", "CASH_PLN", "CASH_BYN",
]


def is_trackable_for_portfolio(asset_label: str) -> bool:
    """True if this ticker can be added to portfolio_holdings via the bot UI."""
    return asset_label.upper() in {t.upper() for t in TRACKABLE_PORTFOLIO_TICKERS}


# Lazy import to avoid yfinance import cost when only the bot is running
def _get_known_assets() -> dict[str, str]:
    """Return {asset_label: asset_class} for every tracked symbol.

    Uses the catalogs from market.py so adding a new ticker there auto-extends
    the portfolio's allowed asset universe. Custom ETFs (CSPX, NATO, ...) are
    merged in from CUSTOM_PORTFOLIO_ASSETS — they may not have live prices but
    are still valid portfolio entries.
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
    # Merge custom assets last; they override market.py defaults if labels collide
    # (e.g. SMH is both a US ETF and a European UCITS — Maksim's is the European one).
    for label, cls in CUSTOM_PORTFOLIO_ASSETS.items():
        universe[label] = cls
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
    isin: str | None = None,
    price_at_add: float | None = None,
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
                   currency, notes, added_at, updated_at, isin, price_at_add)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    isin,
                    price_at_add,
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
                       updated_at = ?,
                       isin = COALESCE(?, isin)
                 WHERE asset_label = ?
                """,
                (new_qty, new_avg, notes, now, isin, asset_label),
            )
            await conn.commit()
            log.info(
                "portfolio: merged %s — was %.6f@$%.2f, now %.6f@$%.2f",
                asset_label, old_qty, old_price, new_qty, new_avg,
            )

    return await get_holding(asset_label) or {}


async def add_usd_holding(
    asset_label: str,
    usd_invested: float,
    *,
    asset_class: str | None = None,
    isin: str | None = None,
    notes: str | None = None,
    currency: str = "USD",
    price_at_add: float | None = None,
) -> dict[str, Any]:
    """Add a position by its dollar cost-basis (no quantity / no avg price).

    Used when the user knows only "I put $1,800 into NATO ETF" and not the
    exact share count or average buy price. The P&L is later approximated via
    price drift from `price_at_add` snapshot (compute_pnl handles the math).

    If `asset_class` is None, it's looked up via validate_asset_label.
    """
    if usd_invested < 0:
        raise ValueError("usd_invested must be >= 0")

    if asset_class is None:
        ok, cls_or_err = validate_asset_label(asset_label)
        if not ok:
            raise ValueError(cls_or_err)
        asset_class = cls_or_err

    now = datetime.now(timezone.utc).isoformat()

    async with get_db() as conn:
        # UPSERT semantics: if asset exists, replace its USD basis
        cur = await conn.execute(
            "SELECT id FROM portfolio_holdings WHERE asset_label = ?",
            (asset_label,),
        )
        row = await cur.fetchone()

        if row is None:
            await conn.execute(
                """
                INSERT INTO portfolio_holdings
                  (asset_label, asset_class, quantity, avg_buy_price,
                   currency, notes, added_at, updated_at,
                   usd_invested, isin, price_at_add)
                VALUES (?, ?, 0.0, 0.0, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_label,
                    asset_class,
                    currency,
                    notes,
                    now,
                    now,
                    usd_invested,
                    isin,
                    price_at_add,
                ),
            )
            log.info(
                "portfolio: seeded %s — $%.2f invested (ISIN %s)",
                asset_label, usd_invested, isin or "—",
            )
        else:
            await conn.execute(
                """
                UPDATE portfolio_holdings
                   SET asset_class = ?,
                       currency = ?,
                       notes = COALESCE(?, notes),
                       updated_at = ?,
                       usd_invested = ?,
                       isin = COALESCE(?, isin),
                       price_at_add = COALESCE(?, price_at_add)
                 WHERE asset_label = ?
                """,
                (
                    asset_class,
                    currency,
                    notes,
                    now,
                    usd_invested,
                    isin,
                    price_at_add,
                    asset_label,
                ),
            )
            log.info(
                "portfolio: updated USD basis of %s — $%.2f invested",
                asset_label, usd_invested,
            )
        await conn.commit()

    return await get_holding(asset_label) or {}


async def truncate_holdings() -> int:
    """Delete ALL portfolio holdings. Returns count of deleted rows. Used by seed scripts."""
    async with get_db() as conn:
        cur = await conn.execute("DELETE FROM portfolio_holdings")
        await conn.commit()
        return cur.rowcount or 0


async def adjust_usd_invested(asset_label: str, delta_usd: float) -> dict[str, Any] | None:
    """Add (positive delta) or subtract (negative delta) $ from a holding's
    usd_invested basis. Used by /portfolio inline buttons:
      💰 Add money    → adjust_usd_invested(label, +X)
      💸 Withdraw     → adjust_usd_invested(label, -X)

    Returns updated holding dict or None if asset doesn't exist or final
    usd_invested would be < 0 (in which case caller should use remove_holding).
    """
    existing = await get_holding(asset_label)
    if not existing:
        return None
    current_usd = float(existing.get("usd_invested") or 0)
    current_qty = float(existing.get("quantity") or 0)

    # If position uses qty+avg (not usd_invested mode), convert: usd = qty*avg
    if current_usd == 0 and current_qty > 0:
        current_usd = current_qty * float(existing.get("avg_buy_price") or 0)

    new_usd = current_usd + float(delta_usd)
    if new_usd < 0:
        return None  # caller should remove instead

    now = datetime.now(timezone.utc).isoformat()
    async with get_db() as conn:
        await conn.execute(
            """UPDATE portfolio_holdings
                  SET usd_invested = ?,
                      quantity = 0,
                      avg_buy_price = 0,
                      updated_at = ?
                WHERE asset_label = ?""",
            (new_usd, now, asset_label),
        )
        await conn.commit()
    log.info(
        "portfolio: adjusted %s by %+.2f USD → %.2f total",
        asset_label, delta_usd, new_usd,
    )
    return await get_holding(asset_label)


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


# Native listing currency for EU UCITS tickers we expose via market.EU_UCITS_SYMBOLS.
# This is what the LSE/Xetra/SIX returns — we convert to USD before P&L math.
#   USD: most Ireland-domiciled UCITS share classes (CSPX, NATO, NUCG, IB01) +
#        Swiss-listed 21Shares crypto ETPs (AETH)
#   EUR: anything traded on Xetra (.DE) or Milan (.MI)
EU_UCITS_CURRENCY: dict[str, str] = {
    "CSPX":     "USD",   # iShares CSPX.L — USD share class
    "SMH":      "EUR",   # VanEck SMH.MI on Borsa Italiana
    "NATO":     "USD",   # HANetf NATO.L — USD share class
    "NUCL":     "USD",   # VanEck NUCG.L — USD
    "EXH1":     "EUR",   # iShares EXH1.DE on Xetra
    "IB1T":     "EUR",   # iShares IB1T.DE — EUR-listed but underlying BTC USD
    "ETH-CORE": "USD",   # 21Shares AETH.SW — USD share class
    "IB01":     "USD",   # iShares $ Treasury — USD-denominated bond ETF
}


def _to_usd(price: float, currency: str, market_data: dict) -> float:
    """Convert a listed-currency price to USD using live forex rates.

    Falls back to a conservative approximate rate if the forex pair is missing
    from market_data (e.g. yfinance hiccup). The error band is acceptable for
    direction-of-movement reasoning in the morning brief.
    """
    if not currency or currency == "USD":
        return price
    forex = market_data.get("forex") or {}
    if currency == "EUR":
        rate = (forex.get("EURUSD") or {}).get("price") or 1.08
        return float(price) * float(rate)
    if currency == "GBP":
        rate = (forex.get("GBPUSD") or {}).get("price") or 1.27
        return float(price) * float(rate)
    if currency == "GBp":   # London quotes in pence — divide by 100, then *GBP/USD
        rate = (forex.get("GBPUSD") or {}).get("price") or 1.27
        return float(price) * float(rate) / 100.0
    if currency == "CHF":
        rate = (forex.get("CHFUSD") or {}).get("price") or 1.13
        return float(price) * float(rate)
    return float(price)


def _lookup_current_price(asset_label: str, market_data: dict) -> float | None:
    """Find the live USD price for asset_label across all market_data buckets.

    For EU-listed UCITS tickers (CSPX, SMH, NATO, NUCL, EXH1, IB1T, ETH-CORE,
    IB01) we apply EU_UCITS_CURRENCY → USD conversion via live forex rates
    so all downstream P&L math stays in USD. Other buckets are assumed USD-
    native (yfinance default for US-listed tickers).
    """
    for bucket_key in (
        "equities",
        "stocks",
        "nuclear",
        "drones_defense",
        "trump_political",
        "commodities",
        "forex",
        "indices",
        "eu_ucits",
        "crypto",
    ):
        bucket = market_data.get(bucket_key) or {}
        if asset_label in bucket:
            try:
                raw_price = float(bucket[asset_label].get("price", 0)) or None
            except (TypeError, ValueError):
                return None
            if raw_price is None:
                return None
            # Only EU-UCITS bucket needs currency conversion
            if bucket_key == "eu_ucits":
                currency = EU_UCITS_CURRENCY.get(asset_label, "USD")
                return _to_usd(raw_price, currency, market_data)
            return raw_price
    return None


def _lookup_market_metrics(asset_label: str, market_data: dict) -> dict[str, float | None]:
    """Return {'change_24h': pct, 'change_7d': pct} for asset_label, or None per field.

    These are LIVE movement numbers (not P&L drift from entry) — Maksim's
    morning advisor needs them to write actionable HOLD/TRIM/ADD advice
    based on what happened TODAY, not just months ago at entry.
    """
    for bucket_key in (
        "equities", "stocks", "nuclear", "drones_defense",
        "trump_political", "commodities", "forex", "indices",
        "eu_ucits", "crypto",
    ):
        bucket = market_data.get(bucket_key) or {}
        if asset_label in bucket:
            d = bucket[asset_label]
            return {
                "change_24h": _safe_float(d.get("change_24h")),
                "change_7d": _safe_float(d.get("change_7d")),
            }
    return {"change_24h": None, "change_7d": None}


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compute_pnl(
    holding: dict[str, Any],
    market_data: dict,
) -> dict[str, Any]:
    """Enrich a holding row with live market_value, unrealized_pnl, pnl_pct.

    Returns a NEW dict (does not mutate the input).

    Three modes:
      - Cash position (asset_class='cash'): price=1.0, P&L always 0
      - Quantity + avg_buy_price tracked: classic shares × price math
      - USD-only (usd_invested set, quantity=0): P&L approximated via current
        price vs price_at_add snapshot. If snapshot missing, P&L shown as 0
        and price_status='usd_basis' so renderers can label it appropriately.
    """
    enriched = dict(holding)
    qty = float(holding.get("quantity", 0) or 0)
    avg = float(holding.get("avg_buy_price", 0) or 0)
    asset_class = holding.get("asset_class", "")
    label = holding.get("asset_label", "")
    usd_invested = holding.get("usd_invested")
    price_at_add = holding.get("price_at_add")

    # Live market-movement metrics — populated for ALL non-cash assets that
    # have a live price. The morning advisor reads these to write advice
    # based on TODAY's movement, not just entry-point drift.
    metrics = _lookup_market_metrics(label, market_data)
    enriched["change_24h_pct"] = metrics["change_24h"]
    enriched["change_7d_pct"] = metrics["change_7d"]

    if asset_class == "cash":
        cash_amount = float(usd_invested) if usd_invested else qty
        enriched["current_price"] = 1.0
        enriched["market_value_usd"] = cash_amount
        enriched["cost_basis_usd"] = cash_amount
        enriched["unrealized_pnl_usd"] = 0.0
        enriched["pnl_pct"] = 0.0
        enriched["price_status"] = "cash"
        return enriched

    current = _lookup_current_price(label, market_data)

    # Mode B: USD-only position (qty=0, usd_invested set) — drift from snapshot
    if (qty <= 0) and usd_invested:
        cost_basis = float(usd_invested)
        if current is not None and price_at_add and price_at_add > 0:
            # Approximate: scale invested $ by relative price change since add
            drift_pct = (current / float(price_at_add) - 1.0) * 100.0
            market_value = cost_basis * (current / float(price_at_add))
            enriched["current_price"] = current
            enriched["market_value_usd"] = market_value
            enriched["cost_basis_usd"] = cost_basis
            enriched["unrealized_pnl_usd"] = market_value - cost_basis
            enriched["pnl_pct"] = drift_pct
            enriched["price_status"] = "usd_drift"
            return enriched
        # No snapshot or no live price — show invested $ as-is, P&L = 0
        enriched["current_price"] = current
        enriched["market_value_usd"] = cost_basis
        enriched["cost_basis_usd"] = cost_basis
        enriched["unrealized_pnl_usd"] = 0.0
        enriched["pnl_pct"] = 0.0
        enriched["price_status"] = "usd_basis"
        return enriched

    # Mode A: classic quantity × price
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

        usd_inv = h.get("usd_invested")
        isin = h.get("isin")
        isin_tag = f" ISIN:{isin}" if isin else ""
        c24 = h.get("change_24h_pct")
        c7d = h.get("change_7d_pct")

        def _fmt_pct(v: float | None) -> str:
            if v is None:
                return "n/a"
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.1f}%"

        live_tag = f" · 24h={_fmt_pct(c24)} · 7d={_fmt_pct(c7d)}" if (c24 is not None or c7d is not None) else ""

        if status == "live" and pnl_pct is not None:
            sign = "+" if pnl_pct >= 0 else ""
            lines.append(
                f"  {label} [{cls}]{isin_tag}: {qty:g} @ ${avg:,.2f} avg → "
                f"now ${cur:,.2f} · MV ${mv:,.0f} · {sign}{pnl_pct:.1f}% from entry{live_tag}"
                + (f" ({notes})" if notes else "")
            )
        elif status == "cash":
            cash_amt = float(usd_inv) if usd_inv else qty
            lines.append(
                f"  {label} [cash]: ${cash_amt:,.2f}" + (f" ({notes})" if notes else "")
            )
        elif status == "usd_drift":
            sign = "+" if (pnl_pct or 0) >= 0 else ""
            lines.append(
                f"  {label} [{cls}]{isin_tag}: ${float(usd_inv):,.0f} invested → "
                f"now ${mv:,.0f} ({sign}{pnl_pct:.1f}% from entry @ ${float(h.get('price_at_add') or 0):.2f}){live_tag}"
                + (f" ({notes})" if notes else "")
            )
        elif status == "usd_basis":
            lines.append(
                f"  {label} [{cls}]{isin_tag}: ${float(usd_inv or 0):,.0f} invested "
                f"(no entry-price snapshot — P&L unknown){live_tag}"
                + (f" ({notes})" if notes else "")
            )
        else:
            lines.append(
                f"  {label} [{cls}]{isin_tag}: {qty:g} @ ${avg:,.2f} avg "
                f"(no live price this run){live_tag}"
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
