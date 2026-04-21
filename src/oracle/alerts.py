"""Real-time alerts — Step 16.

Lightweight polling loop that runs inside the bot process alongside the
scheduled morning/evening jobs (Step 15). Fires every 15 minutes by
default, fetches a fresh market snapshot, and pages the user via Telegram
if any threshold rule is breached.

Threshold rules (hardcoded for Step 16, move to Settings later if needed):

  vix_spike        — VIX > 30 (absolute) — risk-off signal for equities
  btc_delta_poll   — BTC moved ±5% since last poll (~15 min)
  gold_delta_24h   — Gold moved ±3% in 24h
  spy_delta_24h    — SPY moved ±2% in 24h
  oil_delta_24h    — WTI oil moved ±5% in 24h

Design:
  - Poll job fetches `collect_market_data()` — same function used by the
    market collector and the scheduled morning brief. No new API calls;
    just a fresh snapshot.
  - The previous snapshot is stored in `learning_weights` under key
    `alerts:last_snapshot` as a JSON blob of compact fields (price only).
    Comparison against the prior snapshot gives us the ~15-min delta.
  - Per-rule dedup: each rule has a `last_fired_at` key so we don't
    re-alert on the same condition repeatedly. Default dedup window = 6h.
  - Pause awareness: if `is_paused()` returns True, the poll job skips
    entirely — user asked for quiet.
  - Error isolation: if one rule check raises, others continue. If market
    fetch fails, poll job logs and exits until the next tick.

Cost:
  - yfinance + CoinGecko fetches: free, no API key needed
  - LLM calls: none (pure threshold checks)
  - Telegram messages: free
  - Azure compute: ~$3/mo for the always-on lightweight container

Per the user spec: "polls prices every 15 min · alerts if: VIX > 30,
BTC ±10%/hour, Gold ±3%/day". We approximate BTC ±10%/hour with
±5%/poll since the poll interval is 15 min.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, TYPE_CHECKING

from .config import get_settings
from .db import get_db
from .learning import _upsert_weight

if TYPE_CHECKING:
    from telegram.ext import Application

log = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

POLL_INTERVAL_MINUTES = 15
DEDUP_WINDOW_HOURS = 6   # don't re-fire same rule within this window


# ============================================================================
# Alert rule definitions
# ============================================================================


@dataclass
class AlertHit:
    """One triggered alert — ready to render and deliver."""
    rule_id: str
    priority: str               # HIGH | MEDIUM | LOW
    title: str                  # short headline
    details: str                # multi-line body
    asset: str                  # which asset triggered (for rendering)
    current_value: float


def _get(market: dict, category: str, key: str, field: str = "price") -> float | None:
    d = (market.get(category) or {}).get(key)
    if not d:
        return None
    val = d.get(field)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _check_vix_spike(market: dict, last: dict) -> AlertHit | None:
    vix = _get(market, "indices", "VIX")
    if vix is None or vix <= 30:
        return None
    return AlertHit(
        rule_id="vix_spike",
        priority="HIGH",
        asset="VIX",
        current_value=vix,
        title=f"VIX spike — {vix:.1f}",
        details=(
            f"VIX is at <b>{vix:.1f}</b> (threshold: &gt;30).\n\n"
            f"Risk-off signal. Equities and risk assets often lead a drawdown "
            f"when VIX crosses 30 and stays there. Watch SPY, QQQ, BTC for "
            f"follow-through."
        ),
    )


def _check_btc_delta_poll(market: dict, last: dict) -> AlertHit | None:
    """BTC moved ±5% since last poll (~15 min)."""
    btc_now = _get(market, "crypto", "BTC")
    btc_last = (last or {}).get("BTC")
    if btc_now is None or not btc_last:
        return None
    try:
        delta_pct = (btc_now / float(btc_last) - 1) * 100
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if abs(delta_pct) < 5.0:
        return None
    direction = "📈" if delta_pct > 0 else "📉"
    return AlertHit(
        rule_id="btc_delta_poll",
        priority="HIGH",
        asset="BTC",
        current_value=btc_now,
        title=f"BTC moved {delta_pct:+.1f}% in ~15 min {direction}",
        details=(
            f"BTC: <b>${btc_now:,.0f}</b> "
            f"(previous poll: ${float(btc_last):,.0f}, Δ {delta_pct:+.1f}%)\n\n"
            f"Short-interval move of this size usually means a liquidation "
            f"cascade or a major news headline just dropped. Check /digest "
            f"for cross-signal context."
        ),
    )


def _check_24h_delta(
    market: dict,
    category: str,
    key: str,
    threshold_pct: float,
    display_name: str,
    price_fmt: str = "${price:,.2f}",
) -> AlertHit | None:
    d = (market.get(category) or {}).get(key)
    if not d:
        return None
    price = d.get("price")
    change = d.get("change_24h")
    if price is None or change is None:
        return None
    try:
        change = float(change)
        price = float(price)
    except (TypeError, ValueError):
        return None
    if abs(change) < threshold_pct:
        return None
    direction = "📈" if change > 0 else "📉"
    return AlertHit(
        rule_id=f"{key.lower()}_delta_24h",
        priority="MEDIUM",
        asset=display_name,
        current_value=price,
        title=f"{display_name} {change:+.1f}% in 24h {direction}",
        details=(
            f"{display_name}: <b>{price_fmt.format(price=price)}</b> "
            f"(24h change: <b>{change:+.1f}%</b>, threshold ±{threshold_pct}%)\n\n"
            f"Notable 24h move. Check the evening /digest for cluster "
            f"context and any relevant cross-signals."
        ),
    )


def _check_gold_delta_24h(market: dict, last: dict) -> AlertHit | None:
    return _check_24h_delta(market, "commodities", "GOLD", 3.0, "Gold", "${price:,.2f}")


def _check_spy_delta_24h(market: dict, last: dict) -> AlertHit | None:
    return _check_24h_delta(market, "equities", "SPY", 2.0, "SPY (S&amp;P 500)", "${price:.2f}")


def _check_oil_delta_24h(market: dict, last: dict) -> AlertHit | None:
    return _check_24h_delta(market, "commodities", "OIL_WTI", 5.0, "Oil (WTI)", "${price:.2f}")


# Rule registry — add new rules here
ALERT_RULES: dict[str, Callable[[dict, dict], AlertHit | None]] = {
    "vix_spike":       _check_vix_spike,
    "btc_delta_poll":  _check_btc_delta_poll,
    "gold_delta_24h":  _check_gold_delta_24h,
    "spy_delta_24h":   _check_spy_delta_24h,
    "oil_delta_24h":   _check_oil_delta_24h,
}


# ============================================================================
# Snapshot / dedup state (stored in learning_weights K/V)
# ============================================================================


async def get_last_snapshot() -> dict:
    """Read the last compact market snapshot (price-only) from learning_weights."""
    async with get_db() as conn:
        async with conn.execute(
            "SELECT text_value FROM learning_weights WHERE key = 'alerts:last_snapshot'"
        ) as cur:
            row = await cur.fetchone()
    if not row or not row[0]:
        return {}
    try:
        return json.loads(row[0])
    except Exception:  # noqa: BLE001
        return {}


async def save_snapshot(market: dict) -> None:
    """Save a compact price-only snapshot for the next poll's delta check."""
    snap: dict[str, float] = {}
    for category in ("crypto", "equities", "stocks", "commodities", "forex", "indices"):
        bucket = market.get(category) or {}
        for key, d in bucket.items():
            price = d.get("price")
            if price is not None:
                try:
                    snap[key] = float(price)
                except (TypeError, ValueError):
                    pass
    snap["_taken_at"] = datetime.now(timezone.utc).isoformat()
    await _upsert_weight(
        "alerts:last_snapshot", None, json.dumps(snap),
        "market snapshot for next alert delta check",
    )


async def is_rule_in_dedup_window(rule_id: str) -> bool:
    """True if the rule fired within the dedup window — skip re-firing."""
    async with get_db() as conn:
        async with conn.execute(
            "SELECT text_value FROM learning_weights WHERE key = ?",
            (f"alerts:last_fired:{rule_id}",),
        ) as cur:
            row = await cur.fetchone()
    if not row or not row[0]:
        return False
    try:
        last_fired = datetime.fromisoformat(row[0])
    except ValueError:
        return False
    return datetime.now(timezone.utc) - last_fired < timedelta(hours=DEDUP_WINDOW_HOURS)


async def mark_rule_fired(rule_id: str) -> None:
    """Record that we fired this rule now (for dedup)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    await _upsert_weight(
        f"alerts:last_fired:{rule_id}", None, now_iso,
        f"alert fired at {now_iso}",
    )


# ============================================================================
# Check + render + deliver
# ============================================================================


async def check_all_rules(market: dict) -> list[AlertHit]:
    """Run every alert rule against the current market snapshot.

    Returns hits that are NOT in the dedup window. One failing rule check
    never blocks the others (isolated try/except per rule).
    """
    last = await get_last_snapshot()
    hits: list[AlertHit] = []
    for rule_id, checker in ALERT_RULES.items():
        try:
            hit = checker(market, last)
        except Exception as e:  # noqa: BLE001
            log.error("alerts: rule %s check failed: %s", rule_id, e)
            continue
        if not hit:
            continue
        if await is_rule_in_dedup_window(rule_id):
            log.debug("alerts: rule %s in dedup window, skipping", rule_id)
            continue
        hits.append(hit)
    return hits


PRIORITY_EMOJI = {
    "HIGH":   "🚨",
    "MEDIUM": "⚠️",
    "LOW":    "ℹ️",
}


def render_alert(hit: AlertHit) -> str:
    """Render an AlertHit as an HTML Telegram message."""
    emoji = PRIORITY_EMOJI.get(hit.priority, "🔔")
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"{emoji} <b>ORACLE ALERT · {hit.priority}</b>\n"
        f"<code>{now_utc}</code>\n\n"
        f"<b>{hit.title}</b>\n\n"
        f"{hit.details}\n\n"
        f"<i>Alerts fire every {POLL_INTERVAL_MINUTES} min. "
        f"Use /pause to silence. Dedup window: {DEDUP_WINDOW_HOURS}h per rule.</i>"
    )


async def deliver_alerts(application: "Application", hits: list[AlertHit]) -> None:
    """Send one Telegram message per hit + record dedup timestamps."""
    if not hits:
        return
    settings = get_settings()
    if not settings.telegram_chat_id:
        log.warning("alerts: TELEGRAM_CHAT_ID not set, cannot deliver alerts")
        return

    for hit in hits:
        try:
            await application.bot.send_message(
                chat_id=settings.telegram_chat_id,
                text=render_alert(hit),
                parse_mode="HTML",
            )
            await mark_rule_fired(hit.rule_id)
            log.info(
                "alerts: delivered [%s] %s (%s @ %.2f)",
                hit.priority, hit.rule_id, hit.asset, hit.current_value,
            )
        except Exception as e:  # noqa: BLE001
            log.error("alerts: delivery failed for %s: %s", hit.rule_id, e)


# ============================================================================
# Poll job — called by scheduler every N minutes
# ============================================================================


async def alerts_poll_job(application: "Application") -> None:
    """Fetch market, check rules, deliver hits. Called by APScheduler.

    Respects the /pause state — if user paused delivery, poll still runs
    to keep snapshot fresh but doesn't send alerts (silent observation).
    Actually no — if paused, we skip the whole thing including snapshot
    update, so when user resumes the first poll gives them a fresh baseline.
    """
    from .scheduler import is_paused  # lazy import to avoid circular
    from .agents.market import collect_market_data  # lazy import

    if await is_paused():
        log.debug("alerts: poll skipped (paused)")
        return

    try:
        market_data, errors = await collect_market_data()
    except Exception as e:  # noqa: BLE001
        log.error("alerts: market fetch failed: %s", e)
        return

    if errors:
        log.debug("alerts: market fetch had %d errors (non-blocking)", len(errors))

    if not market_data:
        log.warning("alerts: market_data empty, skipping rule checks")
        return

    # Check rules BEFORE saving new snapshot so btc_delta_poll compares
    # against the PREVIOUS snapshot
    hits = await check_all_rules(market_data)

    if hits:
        log.info("alerts: %d rule(s) triggered: %s",
                 len(hits), ", ".join(h.rule_id for h in hits))
        await deliver_alerts(application, hits)
    else:
        log.debug("alerts: no rules triggered this poll")

    # Save fresh snapshot AFTER checks — the NEXT poll compares against this
    await save_snapshot(market_data)


# ============================================================================
# Standalone CLI: `uv run python -m oracle.alerts` (dry run)
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

        print()
        print("=" * 70)
        print("ORACLE alerts dry-run")
        print("=" * 70)

        market_data, errors = await collect_market_data()
        print(f"Market fetch: {len(market_data)} categories, {len(errors)} errors")

        last = await get_last_snapshot()
        taken_at = last.get("_taken_at", "(never)")
        print(f"Previous snapshot: {len([k for k in last if not k.startswith('_')])} "
              f"assets, taken at {taken_at}")

        print()
        print("--- rule checks ---")
        for rule_id, checker in ALERT_RULES.items():
            try:
                hit = checker(market_data, last)
            except Exception as e:  # noqa: BLE001
                print(f"  {rule_id}: ERROR {type(e).__name__}: {e}")
                continue
            dedup = await is_rule_in_dedup_window(rule_id)
            if hit:
                status = " (DEDUP'ed)" if dedup else " → WOULD FIRE"
                print(f"  {rule_id}: {hit.title}{status}")
            else:
                print(f"  {rule_id}: no trigger")

        # Save snapshot for next run
        await save_snapshot(market_data)
        print()
        print("Snapshot saved. Next run will compute delta against this.")

        if "--render" in sys.argv:
            # Render a sample alert so we can see the HTML
            print()
            print("=" * 70)
            print("Sample alert render:")
            print("=" * 70)
            sample = AlertHit(
                rule_id="vix_spike",
                priority="HIGH",
                asset="VIX",
                current_value=32.5,
                title="VIX spike — 32.5",
                details=(
                    "VIX is at <b>32.5</b> (threshold: &gt;30).\n\n"
                    "Risk-off signal. Equities and risk assets often lead a "
                    "drawdown when VIX crosses 30 and stays there."
                ),
            )
            print(render_alert(sample))

    asyncio.run(_main())
