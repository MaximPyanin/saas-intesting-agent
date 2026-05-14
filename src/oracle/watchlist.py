"""Investment watchlist — assets Maksim tapped "📌 Watch this" on.

Pure CRUD + the price-drift check used by alerts.py to fire custom
"watched asset moved ±5%" notifications. Coexists with `feedback` table
(which has a `save` row when Watch is tapped) — feedback is for learning,
watchlist is for active-tracking.

Schema (oracle_data.db.investment_watchlist):
  asset_label, baseline_price, added_at, source_sig_id, user_note,
  last_alert_at, is_active

Dedup window for alerts: 6h (don't spam Maksim on the same threshold).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from .db import get_db

log = logging.getLogger(__name__)

# Price-move threshold from baseline that triggers a watchlist alert.
WATCHLIST_TRIGGER_PCT = 5.0

# Dedup window — don't re-fire the same watch within N hours of last alert.
WATCHLIST_DEDUP_HOURS = 6


async def add_to_watchlist(
    asset_label: str,
    baseline_price: float,
    *,
    source_sig_id: str | None = None,
    user_note: str | None = None,
) -> None:
    """UPSERT a watch on an asset. Re-tapping Watch resets the baseline."""
    now = datetime.now(timezone.utc).isoformat()
    async with get_db() as conn:
        await conn.execute(
            """INSERT INTO investment_watchlist
               (asset_label, baseline_price, added_at, source_sig_id,
                user_note, is_active)
               VALUES (?, ?, ?, ?, ?, 1)
               ON CONFLICT(asset_label) DO UPDATE SET
                   baseline_price = excluded.baseline_price,
                   added_at = excluded.added_at,
                   source_sig_id = excluded.source_sig_id,
                   user_note = COALESCE(excluded.user_note, user_note),
                   last_alert_at = NULL,
                   is_active = 1""",
            (asset_label, float(baseline_price), now, source_sig_id, user_note),
        )
        await conn.commit()
    log.info("watchlist: added/refreshed %s @ baseline $%.2f", asset_label, baseline_price)


async def remove_from_watchlist(asset_label: str) -> bool:
    """Soft-remove (is_active=0) so historical data survives."""
    async with get_db() as conn:
        cur = await conn.execute(
            "UPDATE investment_watchlist SET is_active = 0 WHERE asset_label = ?",
            (asset_label,),
        )
        await conn.commit()
        removed = cur.rowcount > 0
    if removed:
        log.info("watchlist: removed %s", asset_label)
    return removed


async def list_watchlist(active_only: bool = True) -> list[dict[str, Any]]:
    """Return all watchlist rows, dict-like."""
    query = "SELECT * FROM investment_watchlist"
    if active_only:
        query += " WHERE is_active = 1"
    query += " ORDER BY added_at DESC"
    async with get_db() as conn:
        async with conn.execute(query) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def check_watchlist_for_alerts(market_data: dict) -> list[dict[str, Any]]:
    """Compare each active watched asset's current price to its baseline.
    Returns list of triggering watches: {asset, baseline_price, current_price,
    drift_pct, user_note, source_sig_id}.

    Respects WATCHLIST_DEDUP_HOURS — won't include a watch that fired within
    the dedup window.
    """
    from .portfolio import _lookup_current_price  # noqa: PLC0415 — lazy

    watches = await list_watchlist(active_only=True)
    if not watches:
        return []

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=WATCHLIST_DEDUP_HOURS)
    hits: list[dict[str, Any]] = []

    for w in watches:
        asset = w["asset_label"]
        baseline = float(w.get("baseline_price") or 0)
        if baseline <= 0:
            continue
        current = _lookup_current_price(asset, market_data)
        if current is None or current <= 0:
            continue
        drift_pct = (current / baseline - 1.0) * 100.0
        if abs(drift_pct) < WATCHLIST_TRIGGER_PCT:
            continue
        # Dedup
        last = w.get("last_alert_at")
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
                if last_dt > cutoff:
                    continue
            except ValueError:
                pass
        hits.append({
            "asset": asset,
            "baseline_price": baseline,
            "current_price": float(current),
            "drift_pct": drift_pct,
            "user_note": w.get("user_note") or "",
            "source_sig_id": w.get("source_sig_id") or "",
        })
    return hits


async def mark_watch_fired(asset_label: str) -> None:
    """Record that we fired an alert for this watch now (dedup)."""
    now = datetime.now(timezone.utc).isoformat()
    async with get_db() as conn:
        await conn.execute(
            "UPDATE investment_watchlist SET last_alert_at = ? WHERE asset_label = ?",
            (now, asset_label),
        )
        await conn.commit()
