"""Market data collector — Step 4.

Three sources, all running in parallel:

  - yfinance       (free, no API key)  — equities, ETFs, commodities, forex, indices
  - CoinGecko free (free, no API key)  — crypto prices + 24h change
  - FRED           (free, requires FRED_API_KEY) — macro series (Fed funds, CPI, etc.)

Output is a structured dict in state["market_data"]:

    {
      "equities":    {"SPY": {price, change_24h, change_7d}, ...},
      "stocks":      {"NVDA": {...}, ...},
      "commodities": {"GOLD": {...}, "OIL_WTI": {...}, ...},
      "forex":       {"EURUSD": {...}, "DXY": {...}, ...},
      "indices":     {"VIX": {...}, "TNX_10Y": {...}},
      "crypto":      {"BTC": {price, change_24h, last_updated}, ...},
      "macro":       {"FED_FUNDS_RATE": {value, as_of}, ...},
      "collected_at": "2026-04-07T..."
    }

Failure semantics: each source has its own try/except. One failed source
appends an error to state["errors"] (which uses operator.add to accumulate
across parallel collectors) but does not block the others. Per the user
spec: "return_exceptions=True in asyncio.gather".

Note: market data is intentionally NOT written to the signals table.
Signals are textual evidence (articles, posts); market data is structured
numerical context. Step 13 (investment_analyzer) reads it directly from
state["market_data"].
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
import yfinance as yf

from ..config import get_settings
from ..state import OracleState

log = logging.getLogger(__name__)


# ============================================================================
# Symbol catalogs
# ============================================================================

# yfinance — equities & sector ETFs
EQUITY_SYMBOLS: dict[str, str] = {
    "SPY": "SPY",   # S&P 500
    "QQQ": "QQQ",   # NASDAQ-100
    "XLK": "XLK",   # Tech sector
    "XLE": "XLE",   # Energy sector
    "XLF": "XLF",   # Financials sector
    "VNQ": "VNQ",   # REITs
}

# yfinance — individual stocks (Maksim's interest: AI / mega-cap tech / EV / chips)
STOCK_SYMBOLS: dict[str, str] = {
    # Mega-cap tech + AI infra
    "NVDA":  "NVDA",   # Nvidia — AI chips
    "MSFT":  "MSFT",   # Microsoft — cloud + OpenAI
    "GOOGL": "GOOGL",  # Alphabet — search + Gemini
    "AAPL":  "AAPL",   # Apple — devices
    "TSLA":  "TSLA",   # Tesla — EV + robotics
    "AMD":   "AMD",    # AMD — chips alt to Nvidia
    "META":  "META",   # Meta — ads + Llama
    "AMZN":  "AMZN",   # Amazon — AWS + retail
    "TSM":   "TSM",    # TSMC — chip foundry
    "AVGO":  "AVGO",   # Broadcom — networking + AI infra
    "NFLX":  "NFLX",   # Netflix — streaming benchmark
    "PLTR":  "PLTR",   # Palantir — defense AI
    # AI-specific picks (chips, data-center, voice AI)
    "SMCI":  "SMCI",   # Super Micro Computer — AI server OEM
    "ARM":   "ARM",    # ARM Holdings — chip IP
    "ASML":  "ASML",   # ASML — EUV chip lithography monopoly
    "MU":    "MU",     # Micron — HBM memory for AI
    "VRT":   "VRT",    # Vertiv — data center cooling (AI power pick-and-shovel)
    "SOUN":  "SOUN",   # SoundHound AI — voice AI
    "AI":    "AI",     # C3.ai — enterprise AI software
}

# yfinance — nuclear energy + nuclear-for-AI (data center power deals)
# Thesis: AI training clusters need gigawatts; nuclear is having a renaissance.
# Small modular reactors (SMRs), uranium supply, and hyperscaler PPA deals
# (Microsoft-Constellation, Amazon-Talen, Google-Kairos) are the catalysts.
NUCLEAR_SYMBOLS: dict[str, str] = {
    "SMR":  "SMR",    # NuScale Power — SMR developer
    "OKLO": "OKLO",   # Oklo Inc — advanced fission (Sam Altman backed)
    "NNE":  "NNE",    # Nano Nuclear Energy — micro-reactors
    "LEU":  "LEU",    # Centrus Energy — uranium enrichment (US-onshore)
    "CCJ":  "CCJ",    # Cameco — uranium miner (largest Western)
    "BWXT": "BWXT",   # BWX Technologies — naval + commercial nuclear reactors
    "UEC":  "UEC",    # Uranium Energy Corp — uranium miner
    "URA":  "URA",    # Global X Uranium ETF — diversified uranium basket
    "VST":  "VST",    # Vistra — nuclear fleet + AI data-center power PPAs
    "CEG":  "CEG",    # Constellation Energy — largest US nuclear fleet, MSFT deal
}

# yfinance — drones / unmanned systems / defense AI
# Thesis: drone warfare is reshaping defense (Ukraine, Red Sea, Taiwan).
# US primes are slow; upstarts with AI autonomy are the growth story.
DRONE_DEFENSE_SYMBOLS: dict[str, str] = {
    "AVAV": "AVAV",   # AeroVironment — Switchblade loitering munitions
    "KTOS": "KTOS",   # Kratos Defense — unmanned combat aircraft (Valkyrie)
    "RCAT": "RCAT",   # Red Cat Holdings — military drones (Teal Drones)
    "ONDS": "ONDS",   # Ondas Holdings — drone networks + autonomous systems
    "RKLB": "RKLB",   # Rocket Lab — space launch + defense payloads
    "EH":   "EH",     # EHang — Chinese eVTOL / passenger drones
    "UMAC": "UMAC",   # Unusual Machines — drone components (Don Jr. advisor)
}

# yfinance — Trump / family-connected politically-driven stocks
# Thesis: these tickers move on Trump news cycles, policy announcements, and
# family-member board activity. High volatility, high meme-narrative exposure.
# NOT a recommendation — tracked so the investment_analyzer can explain WHY
# they're moving when headlines hit.
TRUMP_POLITICAL_SYMBOLS: dict[str, str] = {
    "DJT":  "DJT",    # Trump Media & Technology Group (Truth Social parent)
    "RUM":  "RUM",    # Rumble — video platform, Trump family ties
    "PSQH": "PSQH",   # PublicSquare — "anti-woke" marketplace, Don Jr. investor
    "PHUN": "PHUN",   # Phunware — Trump 2020 campaign app vendor
}

# yfinance — commodities (futures continuous contracts)
COMMODITY_SYMBOLS: dict[str, str] = {
    "GOLD":      "GC=F",  # Gold front-month
    "SILVER":    "SI=F",  # Silver front-month
    "OIL_WTI":   "CL=F",  # WTI crude
    "OIL_BRENT": "BZ=F",  # Brent crude
    "NATGAS":    "NG=F",  # Henry Hub natural gas
    "COPPER":    "HG=F",  # Copper (industrial bellwether)
    "WHEAT":     "ZW=F",  # Wheat (food + Ukraine war proxy)
}

# yfinance — forex (Maksim is in Warsaw, family in Belarus → PLN/BYN matter)
FOREX_SYMBOLS: dict[str, str] = {
    "EURUSD": "EURUSD=X",  # Euro / US dollar
    "GBPUSD": "GBPUSD=X",  # GBP / USD — needed for UCITS-on-LSE listings in GBp
    "CHFUSD": "CHFUSD=X",  # CHF / USD — needed for SIX Swiss listings (21Shares ETPs)
    "USDPLN": "USDPLN=X",  # US dollar / Polish zloty
    "EURPLN": "EURPLN=X",  # Euro / Polish zloty
    "USDBYN": "USDBYN=X",  # US dollar / Belarusian ruble (may be flaky on yfinance)
    "EURBYN": "EURBYN=X",  # Euro / Belarusian ruble
    "USDRUB": "USDRUB=X",  # US dollar / Russian ruble (regional context)
    "DXY":    "DX-Y.NYB",  # US Dollar Index (broad USD strength)
}

# yfinance — indices
INDEX_SYMBOLS: dict[str, str] = {
    "VIX":     "^VIX",   # Volatility index
    "TNX_10Y": "^TNX",   # 10-year Treasury yield
}

# CoinGecko — crypto
CRYPTO_COINGECKO: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
}

# yfinance — European UCITS ETFs & crypto-ETPs that Maksim actually holds.
# These need exchange suffixes: `.L` London, `.DE` Xetra, `.AS` Amsterdam,
# `.SW` SIX Swiss, `.MI` Milan. The label is what Maksim sees and what's
# stored in portfolio_holdings.asset_label — the value is the yfinance ticker.
#
# IMPORTANT: yfinance returns prices in the LISTING CURRENCY (often EUR or
# GBp pence), not always USD. For now we treat them as USD-equivalent for
# P&L display — close enough for "is my position up or down" purposes.
# A proper FX conversion belongs to Step 21.
EU_UCITS_SYMBOLS: dict[str, str] = {
    "CSPX":     "CSPX.L",    # iShares Core S&P 500 UCITS (ISIN IE00B5BMR087)
    "SMH":      "SMH.MI",    # VanEck Semiconductor UCITS (ISIN IE00BMC38736)
    "NATO":     "NATO.L",    # HANetf Future of Defence UCITS (ISIN IE000OJ5TQP4)
    "NUCL":     "NUCG.L",    # VanEck Uranium & Nuclear Technologies (IE000M7V94E1)
    "EXH1":     "EXH1.DE",   # iShares STOXX Europe 600 Oil & Gas (DE000A0H08M3)
    "IB1T":     "IB1T.DE",   # iShares Bitcoin ETP (XS2940466316)
    "ETH-CORE": "AETH.SW",   # 21Shares Ethereum Core (SIX listing)
    "IB01":     "IB01.L",    # iShares $ Treasury 0-1 yr UCITS (IE00B3VWN518)
}

# FRED — macro economic series
FRED_SERIES: dict[str, str] = {
    "FED_FUNDS_RATE": "DFF",      # Effective Federal Funds Rate (daily)
    "CPI":            "CPIAUCSL", # Consumer Price Index for All Urban Consumers
    "UNEMPLOYMENT":   "UNRATE",   # Unemployment Rate
    "REAL_10Y":       "DFII10",   # 10-Year TIPS yield (real interest rate)
}

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
FRED_BASE = "https://api.stlouisfed.org/fred"

HTTP_TIMEOUT = 20.0


# ============================================================================
# yfinance — sync calls offloaded to a thread
# ============================================================================


def _fetch_yfinance_batch_sync(symbols: dict[str, str]) -> tuple[dict, list[str]]:
    """Synchronous yfinance fetcher — runs in a worker thread.

    Pulls 7 days of daily history per symbol so we can compute both 24h and
    7d changes. Each symbol failure is per-symbol; one bad ticker doesn't
    kill the whole batch.
    """
    results: dict[str, dict] = {}
    errors: list[str] = []
    for label, symbol in symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="7d", interval="1d", auto_adjust=True)
            if hist.empty:
                errors.append(f"yfinance:{symbol}: empty history")
                continue
            closes = hist["Close"]
            latest = float(closes.iloc[-1])
            prev_24h = float(closes.iloc[-2]) if len(closes) >= 2 else latest
            week_ago = float(closes.iloc[0])
            results[label] = {
                "symbol": symbol,
                "price": latest,
                "change_24h": ((latest / prev_24h - 1) * 100) if prev_24h else 0.0,
                "change_7d": ((latest / week_ago - 1) * 100) if week_ago else 0.0,
            }
        except Exception as e:  # noqa: BLE001 — yfinance raises many things
            errors.append(f"yfinance:{symbol}: {type(e).__name__}: {e}")
    return results, errors


async def fetch_yfinance(category: str, symbols: dict[str, str]) -> tuple[dict, list[str]]:
    """Async wrapper around the sync yfinance fetcher."""
    log.debug("yfinance: fetching %s (%d symbols)", category, len(symbols))
    return await asyncio.to_thread(_fetch_yfinance_batch_sync, symbols)


# ============================================================================
# CoinGecko — single batched HTTP request
# ============================================================================


async def fetch_coingecko(coins: dict[str, str]) -> tuple[dict, list[str]]:
    """Fetch latest crypto prices from CoinGecko free tier in one request."""
    errors: list[str] = []
    ids = ",".join(coins.values())
    params = {
        "ids": ids,
        "vs_currencies": "usd",
        "include_24hr_change": "true",
        "include_last_updated_at": "true",
    }
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.get(f"{COINGECKO_BASE}/simple/price", params=params)
            r.raise_for_status()
            payload = r.json()
    except Exception as e:  # noqa: BLE001
        errors.append(f"coingecko: {type(e).__name__}: {e}")
        return {}, errors

    results: dict[str, dict] = {}
    for label, cg_id in coins.items():
        item = payload.get(cg_id)
        if not item:
            errors.append(f"coingecko: missing id {cg_id}")
            continue
        results[label] = {
            "coin_id": cg_id,
            "price": float(item.get("usd", 0)),
            "change_24h": float(item.get("usd_24h_change", 0) or 0),
            "last_updated": int(item.get("last_updated_at", 0) or 0),
        }
    return results, errors


# ============================================================================
# FRED — sequential HTTP per series (only 4 series, no parallelism needed)
# ============================================================================


async def fetch_fred(series: dict[str, str], api_key: str) -> tuple[dict, list[str]]:
    """Fetch latest observation per FRED series.

    Gracefully no-ops if FRED_API_KEY is not set — the rest of the
    market collector still runs.
    """
    errors: list[str] = []
    if not api_key:
        log.info("fred: no FRED_API_KEY in env — macro skipped (set in .env to enable)")
        return {}, []

    results: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        for label, series_id in series.items():
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            }
            try:
                r = await client.get(f"{FRED_BASE}/series/observations", params=params)
                r.raise_for_status()
                obs = r.json().get("observations", [])
                if not obs:
                    errors.append(f"fred:{series_id}: no observations")
                    continue
                latest = obs[0]
                value = latest.get("value")
                results[label] = {
                    "series_id": series_id,
                    "value": float(value) if value not in (None, ".") else None,
                    "as_of": latest.get("date", ""),
                }
            except Exception as e:  # noqa: BLE001
                errors.append(f"fred:{series_id}: {type(e).__name__}: {e}")
    return results, errors


# ============================================================================
# Orchestration
# ============================================================================


async def collect_market_data() -> tuple[dict[str, Any], list[str]]:
    """Run all sources in parallel; return (market_data, errors)."""
    settings = get_settings()

    tasks = {
        "equities":         fetch_yfinance("equities",         EQUITY_SYMBOLS),
        "stocks":           fetch_yfinance("stocks",           STOCK_SYMBOLS),
        "nuclear":          fetch_yfinance("nuclear",          NUCLEAR_SYMBOLS),
        "drones_defense":   fetch_yfinance("drones_defense",   DRONE_DEFENSE_SYMBOLS),
        "trump_political":  fetch_yfinance("trump_political",  TRUMP_POLITICAL_SYMBOLS),
        "commodities":      fetch_yfinance("commodities",      COMMODITY_SYMBOLS),
        "forex":            fetch_yfinance("forex",            FOREX_SYMBOLS),
        "indices":          fetch_yfinance("indices",          INDEX_SYMBOLS),
        "eu_ucits":         fetch_yfinance("eu_ucits",         EU_UCITS_SYMBOLS),
        "crypto":           fetch_coingecko(CRYPTO_COINGECKO),
        "macro":            fetch_fred(FRED_SERIES, settings.fred_api_key),
    }

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    market_data: dict[str, Any] = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    all_errors: list[str] = []

    for category, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            all_errors.append(f"{category}: unhandled {type(result).__name__}: {result}")
            market_data[category] = {}
            continue
        data, errors = result
        market_data[category] = data
        all_errors.extend(errors)

    return market_data, all_errors


# ============================================================================
# LangGraph node — Step 4 replaces the Step 1 placeholder
# ============================================================================


async def market_node(state: OracleState) -> dict:
    """LangGraph node — collects all market data and returns a state patch.

    Returns:
        {"market_data": {...}, "errors": [...]}
    """
    log.info("market: fetching from yfinance + CoinGecko + FRED")
    market_data, errors = await collect_market_data()

    # Compact summary log — pulls a few headline numbers if present
    summary: list[str] = []
    if (btc := market_data.get("crypto", {}).get("BTC")):
        summary.append(f"BTC ${btc['price']:,.0f} ({btc['change_24h']:+.1f}%)")
    if (spy := market_data.get("equities", {}).get("SPY")):
        summary.append(f"SPY ${spy['price']:.2f} ({spy['change_24h']:+.1f}%)")
    if (gold := market_data.get("commodities", {}).get("GOLD")):
        summary.append(f"Gold ${gold['price']:.0f} ({gold['change_24h']:+.1f}%)")
    if (vix := market_data.get("indices", {}).get("VIX")):
        summary.append(f"VIX {vix['price']:.1f}")
    if summary:
        log.info("market: %s", " · ".join(summary))

    if errors:
        log.warning("market: %d source error(s); first 3: %s", len(errors), errors[:3])

    return {
        "market_data": market_data,
        "errors": errors,
    }


# ============================================================================
# Standalone CLI: `uv run python -m oracle.agents.market`
# ============================================================================


if __name__ == "__main__":
    import json
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
        data, errors = await collect_market_data()
        print(json.dumps({"market_data": data, "errors": errors}, indent=2, default=str))

    asyncio.run(_main())
