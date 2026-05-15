"""Scheduler — Step 15.

Runs inside the bot process (Option A from the design decision). Uses
APScheduler's AsyncIOScheduler integrated with python-telegram-bot's
asyncio event loop. Two cron jobs:

  07:00 Warsaw  →  morning_brief_job  (fast: market collector + top breaking
                                        signals from DB, ~5 sec, no LLM)
  19:00 Warsaw  →  evening_digest_job (full graph: collectors + LLM chain +
                                        validator, ~60-90 sec with OpenAI key)

Both jobs check the `pause_until` state before firing — the user can pause
delivery via /pause N days without stopping the bot or affecting recording.

Key management:
  - Scheduler instance lives in `application.bot_data["scheduler"]`
  - Pause state in `learning_weights` table (generic K/V store from Step 2 v2)
  - Digest items stored in `application.bot_data["idea_store"]` /
    `application.bot_data["signal_store"]` — same pattern as the /digest
    command from Step 3. Callbacks work within the bot session.

If the bot restarts, in-flight digest interactions lose their backing store
(users see "idea not found" on button click), but feedback already persisted
in the DB is preserved. Digest runs are triggered on schedule; missed runs
are NOT rerun automatically (APScheduler default: coalesce).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .config import get_settings
from .db import get_db
from .learning import _upsert_weight
from .models import BusinessIdea, InvestmentSignal

if TYPE_CHECKING:
    from telegram.ext import Application

log = logging.getLogger(__name__)


# ============================================================================
# Pause state helpers (reuses learning_weights K/V table)
# ============================================================================


async def get_pause_until_iso() -> str:
    """Read `pause_until` — ISO timestamp or empty string if not paused."""
    async with get_db() as conn:
        async with conn.execute(
            "SELECT text_value FROM learning_weights WHERE key = 'pause_until'"
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row and row[0] else ""


async def is_paused() -> bool:
    """True if pause_until is in the future."""
    iso = await get_pause_until_iso()
    if not iso:
        return False
    try:
        until = datetime.fromisoformat(iso)
    except ValueError:
        return False
    return datetime.now(timezone.utc) < until


async def set_pause_days(days: int) -> str:
    """Set pause to now + days. Returns the new pause_until ISO (empty if cleared)."""
    if days <= 0:
        await _upsert_weight("pause_until", None, "", "resumed manually")
        return ""
    until = datetime.now(timezone.utc) + timedelta(days=days)
    until_iso = until.isoformat()
    await _upsert_weight(
        "pause_until", None, until_iso,
        f"paused for {days} day(s) on {datetime.now(timezone.utc).isoformat()}",
    )
    return until_iso


# ============================================================================
# Morning brief job (07:00 Warsaw)
# ============================================================================


async def morning_brief_job(application: "Application") -> None:
    """Morning brief at 07:00 Warsaw — two sections:

      A. 🚨 СРОЧНО — active alerts right now + recent fires (overnight)
      B. 💼 Совет по портфелю — per-holding HOLD/TRIM/ADD/WATCH advice

    The legacy market snapshot (gold/BTC/SP500/oil/VIX) is REPLACED by these
    two more actionable sections per Maksim's request.

    Latency target: ~10-15s total (one cheap gpt-4o-mini call for the
    portfolio advice + market fetch). Cost: ~$0.02-0.05 per morning.
    """
    log.info("scheduler: morning_brief_job firing")

    if await is_paused():
        log.info("scheduler: morning brief skipped (paused)")
        return

    settings = get_settings()
    if not settings.telegram_chat_id:
        log.warning("scheduler: TELEGRAM_CHAT_ID not set, cannot deliver morning brief")
        return

    # Lazy imports to avoid circular issues and keep scheduler light at startup
    from .agents.market import collect_market_data
    from .agents.portfolio_advisor import generate_morning_portfolio_advice
    from .alerts import get_active_alerts_no_dedup, get_recently_fired_alerts
    from .bot.views import (
        render_portfolio_morning_advice,
        render_urgent_section,
    )
    from .portfolio import get_portfolio_with_pnl

    today_iso = datetime.now().strftime("%Y-%m-%d")

    # 1. Fast market snapshot (needed for both alerts AND portfolio P&L)
    try:
        market_data, market_errors = await collect_market_data()
    except Exception as e:  # noqa: BLE001
        log.error("scheduler: morning market fetch failed: %s", e)
        market_data, market_errors = {}, [str(e)]

    # 2. Build СРОЧНО section
    try:
        active_hits = await get_active_alerts_no_dedup(market_data)
    except Exception as e:  # noqa: BLE001
        log.error("scheduler: alerts (no-dedup) failed: %s", e)
        active_hits = []
    try:
        recent_fires = await get_recently_fired_alerts(hours=12)
    except Exception as e:  # noqa: BLE001
        log.error("scheduler: recent-fires fetch failed: %s", e)
        recent_fires = []

    # Top breaking news from last 12h (DB query — fast)
    breaking_news: list[dict] = []
    try:
        async with get_db() as conn:
            async with conn.execute(
                """SELECT title, source_id, url
                   FROM signals
                   WHERE is_breaking = 1
                     AND datetime(published_at) >= datetime('now', '-12 hours')
                   ORDER BY published_at DESC
                   LIMIT 4"""
            ) as cur:
                breaking_news = [dict(r) for r in await cur.fetchall()]
    except Exception as e:  # noqa: BLE001
        log.error("scheduler: breaking news fetch failed: %s", e)

    # Portfolio holdings with biggest 24h moves (|change| >= 2%)
    portfolio_for_urgent = await get_portfolio_with_pnl(market_data)
    portfolio_movers = sorted(
        [
            h for h in (portfolio_for_urgent.get("holdings") or [])
            if h.get("change_24h_pct") is not None
            and abs(h.get("change_24h_pct") or 0) >= 2.0
        ],
        key=lambda h: abs(h.get("change_24h_pct") or 0),
        reverse=True,
    )

    urgent_text = render_urgent_section(
        active_hits, recent_fires,
        date=today_iso,
        breaking_news=breaking_news,
        portfolio_movers=portfolio_movers,
    )

    try:
        await application.bot.send_message(
            chat_id=settings.telegram_chat_id,
            text=urgent_text,
            parse_mode="HTML",
        )
    except Exception as e:  # noqa: BLE001
        log.error("scheduler: morning urgent-section delivery failed: %s", e)

    # 3. Build portfolio advice section (reuse the portfolio we already fetched)
    portfolio = portfolio_for_urgent

    try:
        advice = await generate_morning_portfolio_advice(portfolio)
    except Exception as e:  # noqa: BLE001
        log.error("scheduler: portfolio_advisor failed: %s — using fallback", e)
        advice = []

    advice_text = render_portfolio_morning_advice(advice, portfolio, date=today_iso)

    try:
        await application.bot.send_message(
            chat_id=settings.telegram_chat_id,
            text=advice_text,
            parse_mode="HTML",
        )
    except Exception as e:  # noqa: BLE001
        log.error("scheduler: morning portfolio-advice delivery failed: %s", e)
        return

    # 4. Update last_morning_at
    await _upsert_weight(
        "last_morning_at", None,
        datetime.now(timezone.utc).isoformat(),
        "morning brief delivered (urgent + portfolio advice)",
    )
    log.info(
        "scheduler: morning brief delivered (%d active alerts · %d overnight fires · %d holdings)",
        len(active_hits), len(recent_fires), len(portfolio.get("holdings") or []),
    )


# ============================================================================
# Evening digest job (19:00 Warsaw) — runs the full graph
# ============================================================================


async def evening_digest_job(application: "Application") -> None:
    """Full ORACLE digest — runs the full graph then delivers cards.

    Uses the same in-memory bot_data store pattern as the /digest command
    from Step 3, so users can tap buttons and give feedback.
    """
    log.info("scheduler: evening_digest_job firing")

    if await is_paused():
        log.info("scheduler: evening digest skipped (paused)")
        return

    settings = get_settings()
    if not settings.telegram_chat_id:
        log.warning("scheduler: TELEGRAM_CHAT_ID not set, cannot deliver digest")
        return

    # Lazy imports
    from .bot.views import (
        idea_buttons,
        investment_buttons,
        render_idea_card,
        render_investment_card,
    )
    from .main import run_once

    # 1. Run the full graph
    try:
        result = await run_once()
    except Exception as e:  # noqa: BLE001
        log.error("scheduler: graph run failed: %s", e)
        try:
            await application.bot.send_message(
                chat_id=settings.telegram_chat_id,
                text=f"⚠️ ORACLE evening digest failed: <code>{type(e).__name__}</code>",
                parse_mode="HTML",
            )
        except Exception:  # noqa: BLE001
            pass
        return

    final_digest = result.get("final_digest", {}) or {}
    idea_dicts = final_digest.get("ideas", []) or []
    sig_dicts = final_digest.get("investments", []) or []

    # 2. Parse back into Pydantic for rendering
    ideas: list[BusinessIdea] = []
    for d in idea_dicts:
        try:
            ideas.append(BusinessIdea(**d))
        except Exception as e:  # noqa: BLE001
            log.warning("scheduler: skipping unparseable idea: %s", e)

    signals: list[InvestmentSignal] = []
    for d in sig_dicts:
        try:
            signals.append(InvestmentSignal(**d))
        except Exception as e:  # noqa: BLE001
            log.warning("scheduler: skipping unparseable signal: %s", e)

    digest_id = uuid.uuid4().hex[:8]

    # 3. Store in bot_data so idea/signal callbacks can look them up
    application.bot_data["current_digest_id"] = digest_id
    application.bot_data["idea_store"] = {}
    application.bot_data["signal_store"] = {}

    # 4. Deliver
    chat_id = settings.telegram_chat_id

    if not ideas and not signals:
        msg = (
            f"🦉 <b>ORACLE evening digest</b> · <code>#{digest_id}</code>\n\n"
            f"<i>Graph ran but produced no ideas or investment scenarios. "
            f"Possible reasons: OPENAI_API_KEY not set, no business_idea "
            f"clusters found in this run's signals, or all ideas got killed "
            f"by the critic.</i>"
        )
        try:
            await application.bot.send_message(chat_id=chat_id, text=msg, parse_mode="HTML")
        except Exception as e:  # noqa: BLE001
            log.error("scheduler: empty digest delivery failed: %s", e)
        return

    # Opening message
    try:
        await application.bot.send_message(
            chat_id=chat_id,
            text=(
                f"🦉 <b>Evening Digest</b> · <code>#{digest_id}</code>\n"
                f"{len(ideas)} business ideas · {len(signals)} investment signals\n"
                f"<i>Tap buttons to give feedback — learning calibration runs every 20 feedbacks.</i>"
            ),
            parse_mode="HTML",
        )
    except Exception as e:  # noqa: BLE001
        log.error("scheduler: opening message failed: %s", e)

    # Idea cards
    for i, idea in enumerate(ideas, start=1):
        idea_id = f"i{i}_{digest_id}"
        application.bot_data["idea_store"][idea_id] = idea
        try:
            await application.bot.send_message(
                chat_id=chat_id,
                text=render_idea_card(idea, i),
                reply_markup=idea_buttons(idea_id),
                parse_mode="HTML",
            )
        except Exception as e:  # noqa: BLE001
            log.error("scheduler: idea card %d delivery failed: %s", i, e)

    # Investment cards
    for i, sig in enumerate(signals, start=1):
        sig_id = f"s{i}_{digest_id}"
        application.bot_data["signal_store"][sig_id] = sig
        try:
            await application.bot.send_message(
                chat_id=chat_id,
                text=render_investment_card(sig, i),
                reply_markup=investment_buttons(sig_id),
                parse_mode="HTML",
            )
        except Exception as e:  # noqa: BLE001
            log.error("scheduler: signal card %d delivery failed: %s", i, e)

    # Update last_digest_at
    await _upsert_weight(
        "last_digest_at", None,
        datetime.now(timezone.utc).isoformat(),
        f"digest #{digest_id} delivered",
    )

    log.info(
        "scheduler: evening digest #%s delivered (%d ideas · %d signals)",
        digest_id, len(ideas), len(signals),
    )


# ============================================================================
# Scheduler setup — called from bot post_init
# ============================================================================


def setup_scheduler(application: "Application") -> AsyncIOScheduler:
    """Create the AsyncIOScheduler, add jobs, start it.

    Stores the scheduler instance on `application.bot_data["scheduler"]`
    so handlers can query job state (next-run time, etc.) and we can
    shut it down cleanly on bot stop.
    """
    settings = get_settings()
    tz_name = settings.timezone or "Europe/Warsaw"
    log.info("scheduler: initializing with timezone=%s", tz_name)

    scheduler = AsyncIOScheduler(timezone=tz_name)

    # Morning brief — 07:00 Warsaw local
    scheduler.add_job(
        morning_brief_job,
        # 09:30 Warsaw — после открытия EU бирж (London 09:00 BST,
        # Xetra/Milan/SIX 09:00 CET) yfinance уже имеет свежие котировки
        # для UCITS-тикеров (CSPX/SMH/NATO/NUCL/EXH1/IB1T/IB01). На 07:00
        # они возвращали NaN потому что биржи ещё закрыты.
        trigger=CronTrigger(hour=9, minute=30, timezone=tz_name),
        args=[application],
        id="morning_brief",
        name="ORACLE morning brief",
        replace_existing=True,
        max_instances=1,
        coalesce=True,  # if a fire is missed (downtime), don't replay
    )

    # Evening digest — 19:00 Warsaw local
    scheduler.add_job(
        evening_digest_job,
        trigger=CronTrigger(hour=19, minute=0, timezone=tz_name),
        args=[application],
        id="evening_digest",
        name="ORACLE evening digest",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    # Real-time alerts — every 15 min (Step 16)
    # Uses IntervalTrigger rather than CronTrigger for simplicity. First
    # fire is 15 min after bot startup; subsequent fires every 15 min.
    # The job itself respects /pause state and dedup windows.
    from .alerts import POLL_INTERVAL_MINUTES, alerts_poll_job
    scheduler.add_job(
        alerts_poll_job,
        trigger=IntervalTrigger(minutes=POLL_INTERVAL_MINUTES),
        args=[application],
        id="alerts_poll",
        name="ORACLE real-time alerts poll",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    application.bot_data["scheduler"] = scheduler

    # Log next-run times for visibility
    for job in scheduler.get_jobs():
        log.info("scheduler: job '%s' next run = %s", job.id, job.next_run_time)

    return scheduler


def shutdown_scheduler(application: "Application") -> None:
    """Stop the scheduler cleanly. Called from bot post_stop if needed."""
    scheduler: AsyncIOScheduler | None = application.bot_data.get("scheduler")
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)
        log.info("scheduler: shut down")
