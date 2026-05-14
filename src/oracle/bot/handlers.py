"""Command and callback handlers for the ORACLE Telegram bot.

Architecture:
  - One handler per slash command (cmd_*)
  - One callback dispatcher (callback_handler) routes by callback_data prefix
  - Helpers (_handle_*) do the actual work for each callback type
  - Per-digest item store lives in `application.bot_data` so callbacks
    can look up the original idea/signal by ID after the user clicks

Storage:
  - Items survive only until the bot process restarts (in-memory bot_data).
    Step 9+ will replace this with a digest_items table when the real
    pipeline produces persistent digests.
  - Feedback IS persisted to oracle_data.db.feedback (Step 2 schema).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes, ConversationHandler

from ..agents.custom import (
    VALID_CATEGORIES,
    VALID_SOURCE_TYPES,
    add_user_source,
    load_active_sources,
    remove_user_source,
)
from ..agents.market import collect_market_data
from ..db import get_db
from ..learning import (
    CALIBRATION_THRESHOLD,
    calibrate_from_feedback,
    feedbacks_at_last_calibration,
    get_weekly_summary,
    last_calibration_at,
    maybe_calibrate,
    total_feedback_count,
)
from ..models import BusinessIdea, InvestmentSignal
from ..observability import get_cost_summary
from ..portfolio import (
    add_or_update_holding,
    get_portfolio_with_pnl,
    list_holdings,
    remove_holding,
    validate_asset_label,
)
from ..scheduler import get_pause_until_iso, is_paused, set_pause_days
from .mock import (
    mock_business_ideas,
    mock_investment_signals,
    mock_morning_data,
)
from .views import (
    SOURCE_CATEGORY_LABELS,
    SOURCE_TYPE_LABELS,
    add_source_category_kb,
    add_source_confirm_kb,
    add_source_type_kb,
    dislike_reason_buttons,
    idea_buttons,
    investment_buttons,
    remove_source_list_kb,
    render_idea_card,
    render_investment_card,
    render_morning_brief,
    render_real_sources_card,
    render_source_stats_card,
    sources_buttons,
)

log = logging.getLogger(__name__)


# ============================================================================
# bot_data helpers — typed accessors for the per-application item store
# ============================================================================


def _idea_store(ctx: ContextTypes.DEFAULT_TYPE) -> dict[str, BusinessIdea]:
    return ctx.application.bot_data.setdefault("idea_store", {})


def _signal_store(ctx: ContextTypes.DEFAULT_TYPE) -> dict[str, InvestmentSignal]:
    return ctx.application.bot_data.setdefault("signal_store", {})


def _current_digest_id(ctx: ContextTypes.DEFAULT_TYPE) -> str:
    return ctx.application.bot_data.get("current_digest_id", "")


# ============================================================================
# /start
# ============================================================================


WELCOME_HTML = """\
👋 <b>Welcome to ORACLE</b>

I scan news, tech trends, Reddit, Telegram, YouTube, and websites to surface
SaaS / AI business ideas <b>before</b> they hype, plus secondary investment
signals.

<b>Commands:</b>
• /digest — full digest right now
• /morning — morning brief
• /sources — manage your sources
• /add_source — add a custom source
• /portfolio — your holdings + live P&amp;L
• /add_holding — add/update a position
• /remove_holding &lt;ASSET&gt; — drop a position
• /history &lt;topic&gt; — topic history
• /saved — your saved items
• /stats — feedback + learning calibration state
• /calibrate — force-run learning calibration now
• /cost — LLM cost breakdown (tokens, agents, days)
• /settings — focus weights
• /pause &lt;days&gt; — pause digests

<i>Tap 🔥 / ❌ / 📌 buttons on each idea — every 20 feedbacks I'll
auto-recalibrate to learn what you actually want.</i>
"""


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME_HTML, parse_mode="HTML")


# ============================================================================
# /digest — generate mock digest, send 3 idea + 3 investment cards
# ============================================================================


async def cmd_digest(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Run the REAL ORACLE pipeline and deliver live ideas + investment signals.

    Uses run_once() from oracle.main which executes the full graph:
    collectors -> synthesizer -> idea_generator<->critic (Reflexion) ->
    validator -> investment_analyzer -> formatter. The investment_analyzer
    reads the user's portfolio from oracle_data.db for personalized scenarios.
    """
    from ..main import run_once  # noqa: PLC0415 — local import avoids circular dep

    digest_id = uuid.uuid4().hex[:8]
    ctx.application.bot_data["current_digest_id"] = digest_id
    ctx.application.bot_data["idea_store"] = {}
    ctx.application.bot_data["signal_store"] = {}

    progress = await update.message.reply_text(
        f"📡 Running full ORACLE pipeline <code>#{digest_id}</code>...\n"
        f"<i>Collecting signals (market + scout + trend + custom) → "
        f"synthesizer → idea_generator ↔ critic (up to 3 rounds) → "
        f"validator → investment_analyzer.</i>\n"
        f"<i>This takes ~1-3 minutes. Your portfolio will be used for "
        f"personalized investment scenarios.</i>",
        parse_mode="HTML",
    )

    try:
        result = await run_once(thread_id=f"digest-{digest_id}")
    except Exception as e:  # noqa: BLE001
        log.exception("cmd_digest: pipeline failed")
        await progress.edit_text(
            f"⚠️ Pipeline failed: <code>{html_escape(str(e))}</code>\n\n"
            f"<i>Check bot logs for full traceback.</i>",
            parse_mode="HTML",
        )
        return

    final_digest = result.get("final_digest") or {}
    ideas_raw = final_digest.get("ideas") or []
    ideas_extra_raw = final_digest.get("ideas_extra") or []
    investments_raw = final_digest.get("investments") or []
    errors = result.get("errors") or []

    # Coerce dicts from graph state into Pydantic models for rendering
    ideas: list[BusinessIdea] = []
    for d in ideas_raw:
        try:
            ideas.append(BusinessIdea.model_validate(d))
        except Exception as e:  # noqa: BLE001
            log.warning("cmd_digest: failed to parse idea %r: %s", d.get("title"), e)

    # Stash extras for /more (raw dicts — parsed lazily there)
    ctx.application.bot_data["ideas_pool"] = ideas_extra_raw
    ctx.application.bot_data["ideas_picked_industries"] = list({
        (i.industry or "other").lower() for i in ideas
    })

    signals: list[InvestmentSignal] = []
    for d in investments_raw:
        try:
            signals.append(InvestmentSignal.model_validate(d))
        except Exception as e:  # noqa: BLE001
            log.warning("cmd_digest: failed to parse signal %r: %s", d.get("asset"), e)

    # Run-cost line (one-liner shown under the header)
    cost_line = ""
    try:
        from ..observability import get_last_run_cost  # noqa: PLC0415
        cost = await get_last_run_cost(f"digest-{digest_id}")
        if cost.get("total_calls"):
            cost_line = (
                f"\n💰 {cost['total_calls']} LLM calls · "
                f"${cost['total_cost_usd']:.4f} · "
                f"in={cost['total_input_tokens']} out={cost['total_output_tokens']} tokens"
            )
    except Exception as e:  # noqa: BLE001
        log.debug("cmd_digest: cost summary failed: %s", e)

    await progress.edit_text(
        f"✅ Pipeline done <code>#{digest_id}</code>\n"
        f"• {len(ideas)} business idea(s) survived\n"
        f"• {len(signals)} investment signal(s)\n"
        f"• {len(errors)} non-fatal source error(s)"
        f"{cost_line}",
        parse_mode="HTML",
    )

    # Ideas — one message per idea
    if ideas:
        for i, idea in enumerate(ideas, start=1):
            idea_id = f"i{i}_{digest_id}"
            _idea_store(ctx)[idea_id] = idea
            try:
                await update.message.reply_text(
                    render_idea_card(idea, i),
                    reply_markup=idea_buttons(idea_id),
                    parse_mode="HTML",
                )
            except Exception as e:  # noqa: BLE001
                log.warning("cmd_digest: render idea %d failed: %s", i, e)
    else:
        await update.message.reply_text(
            "📭 <i>No business ideas survived the critic this run. "
            "Try again later when signals refresh.</i>",
            parse_mode="HTML",
        )

    # Investment signals — one message per signal
    if signals:
        for i, sig in enumerate(signals, start=1):
            sig_id = f"s{i}_{digest_id}"
            _signal_store(ctx)[sig_id] = sig
            try:
                await update.message.reply_text(
                    render_investment_card(sig, i),
                    reply_markup=investment_buttons(sig_id),
                    parse_mode="HTML",
                )
            except Exception as e:  # noqa: BLE001
                log.warning("cmd_digest: render signal %d failed: %s", i, e)
    else:
        await update.message.reply_text(
            "📭 <i>No investment scenarios this run.</i>",
            parse_mode="HTML",
        )

    await update.message.reply_text(
        f"✅ Digest <code>#{digest_id}</code> delivered. "
        f"Tap 🔥 / ❌ / 📌 buttons to give feedback — every 20 feedbacks "
        f"triggers auto-calibration.",
        parse_mode="HTML",
    )


# ============================================================================
# /morning
# ============================================================================


async def cmd_clearfeedback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Wipe feedback + learning calibration to start with a clean slate.

    Usage:
      /clearfeedback           — show what would be deleted (dry run)
      /clearfeedback confirm   — actually delete

    Removes:
      - all rows from `feedback` table
      - cached `prompt_injection_ideas`, `weekly_summary`, `manual_preferences`
        from `learning_weights` (resets the auto-learned profile)
      - calibration counter

    Does NOT touch: portfolio_holdings, signals, user_sources, watchlist.
    """
    args = " ".join(ctx.args or []).strip().lower()
    confirm = args == "confirm"

    async with get_db() as conn:
        async with conn.execute("SELECT COUNT(*) FROM feedback") as cur:
            fb_count = (await cur.fetchone())[0]
        async with conn.execute(
            "SELECT COUNT(*) FROM learning_weights WHERE key IN "
            "('prompt_injection_ideas','weekly_summary','manual_preferences',"
            "'last_calibration_at','calibrations_run','feedbacks_at_last_calibration')"
        ) as cur:
            lw_count = (await cur.fetchone())[0]

    if not confirm:
        await update.message.reply_text(
            f"🗑️ <b>/clearfeedback</b> — dry run\n\n"
            f"Будет удалено:\n"
            f"• <b>{fb_count}</b> строк из <code>feedback</code>\n"
            f"• <b>{lw_count}</b> learning ключей (auto-prefs, manual prefs, weekly summary)\n\n"
            f"Не тронем: portfolio, signals, user_sources, watchlist.\n\n"
            f"Подтвердить: <code>/clearfeedback confirm</code>",
            parse_mode="HTML",
        )
        return

    async with get_db() as conn:
        await conn.execute("DELETE FROM feedback")
        await conn.execute(
            "DELETE FROM learning_weights WHERE key IN "
            "('prompt_injection_ideas','weekly_summary','manual_preferences',"
            "'last_calibration_at','calibrations_run','feedbacks_at_last_calibration')"
        )
        await conn.commit()
    log.info("clearfeedback: wiped %d feedback rows + %d learning keys", fb_count, lw_count)
    await update.message.reply_text(
        f"✅ Удалено: <b>{fb_count}</b> фидбэков + <b>{lw_count}</b> learning ключей.\n\n"
        f"<i>Бот начнёт собирать предпочтения с нуля.</i>",
        parse_mode="HTML",
    )


async def cmd_preferences(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Manual preference override for idea_generator.

    Usage:
      /preferences                  — show current manual prefs + auto-learned
      /preferences <text>           — set/replace manual preference text
      /preferences clear            — wipe manual preferences (back to auto only)

    The text is injected into idea_generator's system prompt with HIGHEST
    authority (above auto-learned preferences). Examples:
      /preferences хочу больше health и fitness, меньше b2b SaaS
      /preferences избегай ai_tools на ближайший месяц
      /preferences ищу только идеи где revenue $50-200/mo, MVP <= 4 weeks
    """
    from ..learning import (  # noqa: PLC0415
        get_manual_preferences,
        get_prompt_injection_ideas,
        set_manual_preferences,
    )

    args_text = " ".join(ctx.args or []).strip()

    if not args_text:
        # Show current state
        manual = await get_manual_preferences()
        learned = await get_prompt_injection_ideas()
        parts: list[str] = ["⚙️ <b>Preferences</b>\n"]
        parts.append("<b>Manual</b> (твои явные правила):")
        parts.append(f"<i>{html_escape(manual)}</i>" if manual else "<i>(не задано)</i>")
        parts.append("")
        parts.append("<b>Auto-learned</b> (из лайков/дизлайков, после ≥20 фидбэков):")
        parts.append(f"<i>{html_escape(learned)}</i>" if learned else "<i>(пока недостаточно фидбэков)</i>")
        parts.append("")
        parts.append("<b>Установить:</b> <code>/preferences хочу больше health, меньше b2b</code>")
        parts.append("<b>Очистить:</b> <code>/preferences clear</code>")
        await update.message.reply_text("\n".join(parts), parse_mode="HTML")
        return

    if args_text.lower() == "clear":
        await set_manual_preferences("")
        await update.message.reply_text(
            "🗑️ <i>Manual preferences cleared.</i> Используются только auto-learned.",
            parse_mode="HTML",
        )
        return

    if len(args_text) > 1000:
        await update.message.reply_text(
            "⚠️ Слишком длинное правило (>1000 символов). Сократи.",
            parse_mode="HTML",
        )
        return

    await set_manual_preferences(args_text)
    await update.message.reply_text(
        f"✅ <b>Manual preferences saved.</b>\n\n"
        f"<i>{html_escape(args_text)}</i>\n\n"
        f"Применится со следующего /digest. Очистить: <code>/preferences clear</code>",
        parse_mode="HTML",
    )


async def cmd_lastdigest(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Re-show ideas from the current bot session's last digest.

    Useful when Maksim clicked 'Like' on a card and wants to scroll back to it
    (the toast ack doesn't remove the card anymore, but if bot restarted or
    chat scrolled away, this re-renders them).

    Reads `idea_store` and `signal_store` from bot_data — survives until bot
    restart. No new LLM calls.
    """
    store = _idea_store(ctx)
    if not store:
        await update.message.reply_text(
            "📭 <i>Не нашёл идей в кэше — возможно, бот перезапускался "
            "или digest ещё не запускался. Запусти /digest.</i>",
            parse_mode="HTML",
        )
        return

    from .views import idea_buttons, render_idea_card  # noqa: PLC0415

    await update.message.reply_text(
        f"🔄 <b>Re-render: {len(store)} идей из последнего digest'а</b>",
        parse_mode="HTML",
    )
    for idea_id, idea in store.items():
        try:
            await update.message.reply_text(
                render_idea_card(idea, 0),
                reply_markup=idea_buttons(idea_id),
                parse_mode="HTML",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("cmd_lastdigest: render %s failed: %s", idea_id, e)


async def cmd_more(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Surface 3 MORE business ideas from this run's pool, prioritizing
    industries NOT yet shown. Reads `ideas_pool` + `ideas_picked_industries`
    populated by the previous /digest call.

    No new LLM calls — uses already-validated ideas that didn't make the
    top-3 cut due to diversity quota. Cheap (~0 cost).
    """
    pool_raw: list[dict] = ctx.application.bot_data.get("ideas_pool") or []
    seen_industries: set[str] = {
        i.lower() for i in ctx.application.bot_data.get("ideas_picked_industries") or []
    }
    if not pool_raw:
        await update.message.reply_text(
            "📭 <i>Нет дополнительных идей в пуле этого digest'а.\n"
            "Запусти новый /digest когда захочешь свежих.</i>",
            parse_mode="HTML",
        )
        return

    # Re-apply diversity selector against the pool, biased to new buckets.
    # Trick: feed the selector a copy of the pool but pre-seed seen_industries
    # by removing same-bucket candidates from contention.
    from ..agents.idea_generator import _normalize_industry, TECH_BUCKET  # noqa: PLC0415

    def _bucket(industry: str) -> str:
        return "TECH" if industry in TECH_BUCKET else industry

    seen_buckets = {_bucket(i) for i in seen_industries}

    # Normalize industry on every idea in pool
    for d in pool_raw:
        d["industry"] = _normalize_industry(d.get("industry"))

    # Sort by score same as diversity selector does
    def _score(d: dict) -> tuple[int, int, int, int]:
        verdict = (d.get("verdict") or "").upper()
        strong = 1 if verdict == "STRONG_PASS" else 0
        conf = int(d.get("confidence") or 0)
        rounds = int(d.get("reflexion_rounds_passed") or 0)
        tier = 1 if conf >= 60 else 0
        return (tier, strong, conf, rounds)

    pool_sorted = sorted(pool_raw, key=_score, reverse=True)

    picked_raw: list[dict] = []
    used_buckets: set[str] = set()
    # Pass 1: only NEW buckets (not in seen_buckets)
    for d in pool_sorted:
        b = _bucket(d.get("industry") or "other")
        if b in seen_buckets or b in used_buckets:
            continue
        picked_raw.append(d)
        used_buckets.add(b)
        if len(picked_raw) >= 3:
            break
    # Pass 2: if short, allow any bucket (still skipping previously-used in /digest)
    if len(picked_raw) < 3:
        picked_ids = {id(d) for d in picked_raw}
        for d in pool_sorted:
            if id(d) in picked_ids:
                continue
            b = _bucket(d.get("industry") or "other")
            if b in used_buckets:
                continue
            picked_raw.append(d)
            used_buckets.add(b)
            if len(picked_raw) >= 3:
                break

    if not picked_raw:
        await update.message.reply_text(
            "📭 <i>В пуле остались только идеи из тех же отраслей что уже показывал.\n"
            "Запусти свежий /digest для новых тем.</i>",
            parse_mode="HTML",
        )
        return

    # Render each via the standard idea card + register feedback buttons
    digest_id = ctx.application.bot_data.get("current_digest_id", "more")
    ideas: list[BusinessIdea] = []
    for d in picked_raw:
        try:
            ideas.append(BusinessIdea.model_validate(d))
        except Exception as e:  # noqa: BLE001
            log.warning("cmd_more: failed to parse idea: %s", e)

    if not ideas:
        await update.message.reply_text("⚠️ Не удалось распарсить идеи из пула.")
        return

    # Update tracker so a SECOND /more skips these buckets too
    new_industries = list({(i.industry or "other").lower() for i in ideas})
    ctx.application.bot_data["ideas_picked_industries"] = list(
        set(ctx.application.bot_data.get("ideas_picked_industries") or []) | set(new_industries)
    )
    # Remove picked from pool so subsequent /more doesn't repeat
    picked_titles = {i.title for i in ideas}
    ctx.application.bot_data["ideas_pool"] = [
        d for d in pool_raw if d.get("title") not in picked_titles
    ]

    from .views import idea_buttons, render_idea_card  # noqa: PLC0415

    await update.message.reply_text(
        f"🔄 <b>Ещё {len(ideas)} идей</b> из этого digest'а — другие отрасли.",
        parse_mode="HTML",
    )
    for i, idea in enumerate(ideas, start=1):
        idea_id = f"m{i}_{digest_id}"
        _idea_store(ctx)[idea_id] = idea
        try:
            await update.message.reply_text(
                render_idea_card(idea, i),
                reply_markup=idea_buttons(idea_id),
                parse_mode="HTML",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("cmd_more: render idea %d failed: %s", i, e)


async def cmd_morning(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually trigger the 07:00 Warsaw morning brief.

    Mirrors `scheduler.morning_brief_job` but writes to the user who typed
    /morning (so testing doesn't require the scheduled chat_id). Two
    messages: 🚨 СРОЧНО + 💼 совет по портфелю.
    """
    from ..agents.portfolio_advisor import generate_morning_portfolio_advice  # noqa: PLC0415
    from ..alerts import (  # noqa: PLC0415
        get_active_alerts_no_dedup,
        get_recently_fired_alerts,
    )
    from ..portfolio import get_portfolio_with_pnl  # noqa: PLC0415
    from .views import (  # noqa: PLC0415
        render_portfolio_morning_advice,
        render_urgent_section,
    )

    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        market_data, _ = await collect_market_data()
    except Exception as e:  # noqa: BLE001
        log.error("/morning: market fetch failed: %s", e)
        market_data = {}

    try:
        active = await get_active_alerts_no_dedup(market_data)
        recent = await get_recently_fired_alerts(hours=12)
    except Exception as e:  # noqa: BLE001
        log.error("/morning: alerts read failed: %s", e)
        active, recent = [], []

    # Top breaking news from last 12h
    breaking_news: list[dict] = []
    try:
        async with get_db() as conn:
            async with conn.execute(
                """SELECT title, source_id, url FROM signals
                   WHERE is_breaking = 1
                     AND datetime(published_at) >= datetime('now', '-12 hours')
                   ORDER BY published_at DESC LIMIT 4"""
            ) as cur:
                breaking_news = [dict(r) for r in await cur.fetchall()]
    except Exception as e:  # noqa: BLE001
        log.error("/morning: breaking news fetch failed: %s", e)

    try:
        portfolio = await get_portfolio_with_pnl(market_data)
    except Exception as e:  # noqa: BLE001
        log.error("/morning: portfolio fetch failed: %s", e)
        portfolio = {"holdings": [], "totals": {}}

    portfolio_movers = sorted(
        [
            h for h in (portfolio.get("holdings") or [])
            if h.get("change_24h_pct") is not None
            and abs(h.get("change_24h_pct") or 0) >= 2.0
        ],
        key=lambda h: abs(h.get("change_24h_pct") or 0),
        reverse=True,
    )

    await update.message.reply_text(
        render_urgent_section(
            active, recent,
            date=today_iso,
            breaking_news=breaking_news,
            portfolio_movers=portfolio_movers,
        ),
        parse_mode="HTML",
    )

    try:
        advice = await generate_morning_portfolio_advice(portfolio)
    except Exception as e:  # noqa: BLE001
        log.error("/morning: portfolio advice failed: %s", e)
        advice = []

    await update.message.reply_text(
        render_portfolio_morning_advice(advice, portfolio, date=today_iso),
        parse_mode="HTML",
    )


# ============================================================================
# /sources
# ============================================================================


async def cmd_sources(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show real user sources from oracle_data.db (Step 8)."""
    sources = await load_active_sources()
    await update.message.reply_text(
        render_real_sources_card(sources),
        reply_markup=sources_buttons(),
        parse_mode="HTML",
    )


# ============================================================================
# /add_source — interactive ConversationHandler (Step 8)
# ============================================================================
#
# State machine:
#
#   /add_source  →  ASK_TYPE
#   user picks type        →  ASK_CATEGORY
#   user picks category    →  ASK_URL  (text input)
#   user sends URL/handle  →  ASK_NAME (text input)
#   user sends name        →  ASK_CONFIRM (yes/cancel)
#   user confirms          →  END (insert into user_sources)
#   /cancel at any state   →  END
#
# Per-user draft lives in ctx.user_data["new_source"] — single user bot, but
# python-telegram-bot scopes user_data per Telegram user automatically.

ASK_TYPE, ASK_CATEGORY, ASK_URL, ASK_NAME, ASK_CONFIRM = range(5)

# Add-to-portfolio ConversationHandler states (Step 20.5 — Maksim's request)
# Entered via "➕ В мой портфель" inline button under each investment card.
#
#   click button         →  ASK_HOLDING_QTY  (text input, supports "usd 1500")
#   user sends quantity  →  ASK_HOLDING_PRICE (text input, or "-" for current)
#   user sends price     →  END (insert into portfolio_holdings)
#   /cancel              →  END (clears draft)
ASK_HOLDING_QTY, ASK_HOLDING_PRICE = range(100, 102)

# Portfolio adjust (+$/-$) ConversationHandler
#   Click 💰 +$ or 💸 -$ → ASK_ADJUST_AMOUNT (text input)
#   User sends number → adjust_usd_invested → END
ASK_ADJUST_AMOUNT = 200

# Portfolio "add new position from scratch" ConversationHandler
#   Click ➕ Добавить новую позицию → PICK_NEW_TICKER (inline keyboard with whitelist)
#   Pick ticker → ASK_NEW_AMOUNT (text input)
#   User sends $ amount → add_usd_holding → END
PICK_NEW_TICKER, ASK_NEW_AMOUNT = range(300, 302)


# Type-specific instructions for the URL input step
URL_INSTRUCTIONS = {
    "telegram_channel": (
        "💬 <b>Telegram channel</b>\n\n"
        "Send the channel handle (e.g. <code>@bensbites</code>) "
        "or a t.me link.\n\n"
        "<i>Note: requires one-time auth via "
        "</i><code>uv run python -m oracle.agents.custom auth-tg</code><i> "
        "from your terminal first.</i>"
    ),
    "youtube_channel": (
        "📺 <b>YouTube channel</b>\n\n"
        "Send the channel ID (e.g. <code>UCqhFigXTltY4tldNOyzj7sg</code>).\n"
        "Find it in the channel URL: <code>youtube.com/channel/UC...</code>\n\n"
        "<i>Requires </i><code>YOUTUBE_API_KEY</code><i> in .env.</i>"
    ),
    "website_blog": (
        "🌐 <b>Website / blog</b>\n\n"
        "Send the full site URL (e.g. <code>https://danluu.com</code>).\n\n"
        "<i>I'll auto-discover the RSS feed if there is one.</i>"
    ),
    "rss_custom": (
        "📰 <b>RSS feed</b>\n\n"
        "Send the full RSS/Atom URL "
        "(e.g. <code>https://example.com/feed.xml</code>)."
    ),
    "reddit_custom": (
        "🔥 <b>Subreddit</b>\n\n"
        "Send the subreddit name (e.g. <code>r/SaaS</code> or just <code>SaaS</code>)."
    ),
}


def _draft(ctx: ContextTypes.DEFAULT_TYPE) -> dict:
    return ctx.user_data.setdefault("new_source", {})


def _clear_draft(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    ctx.user_data.pop("new_source", None)


def _normalize_url_input(source_type: str, raw: str) -> str:
    """Normalize user input to the canonical form for each source type."""
    raw = raw.strip()
    if source_type == "telegram_channel":
        # @bensbites or t.me/bensbites or https://t.me/bensbites → @bensbites
        if "t.me/" in raw:
            raw = raw.split("t.me/", 1)[1].rstrip("/")
        if not raw.startswith("@"):
            raw = "@" + raw
        return raw
    if source_type == "youtube_channel":
        # https://youtube.com/channel/UC... → UC...
        if "/channel/" in raw:
            raw = raw.split("/channel/", 1)[1].split("/")[0]
        return raw
    if source_type == "reddit_custom":
        # /r/SaaS or r/SaaS or SaaS → r/SaaS
        cleaned = raw.lstrip("/").lstrip("r/").lstrip("/")
        return f"r/{cleaned}"
    if source_type in ("website_blog", "rss_custom"):
        if not raw.startswith(("http://", "https://")):
            raw = "https://" + raw
        return raw
    return raw


# ----- Entry points -----


async def cmd_add_source(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    """Entry point: /add_source slash command."""
    _clear_draft(ctx)
    await update.message.reply_text(
        "➕ <b>Add a new source</b>\n\n"
        "Where should I pull signals from?",
        reply_markup=add_source_type_kb(),
        parse_mode="HTML",
    )
    return ASK_TYPE


async def add_source_callback_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    """Entry point: ➕ Add source button on /sources card."""
    query = update.callback_query
    await query.answer()
    _clear_draft(ctx)
    await query.message.reply_text(
        "➕ <b>Add a new source</b>\n\n"
        "Where should I pull signals from?",
        reply_markup=add_source_type_kb(),
        parse_mode="HTML",
    )
    return ASK_TYPE


# ----- State handlers -----


async def add_source_type(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    """ASK_TYPE → ASK_CATEGORY: user picked source type."""
    query = update.callback_query
    await query.answer()
    source_type = query.data.replace("addtype_", "", 1)
    if source_type not in VALID_SOURCE_TYPES:
        await query.edit_message_text("⚠️ Invalid source type. Cancelled.")
        _clear_draft(ctx)
        return ConversationHandler.END

    _draft(ctx)["source_type"] = source_type
    type_label = SOURCE_TYPE_LABELS.get(source_type, source_type)
    await query.edit_message_text(
        f"➕ <b>Add source</b> · {type_label}\n\n"
        f"Now pick a <b>category</b>:\n"
        f"<i>(business ideas are ORACLE's primary focus)</i>",
        reply_markup=add_source_category_kb(),
        parse_mode="HTML",
    )
    return ASK_CATEGORY


async def add_source_category(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    """ASK_CATEGORY → ASK_URL: user picked category, ask for URL."""
    query = update.callback_query
    await query.answer()
    category = query.data.replace("addcat_", "", 1)
    if category not in VALID_CATEGORIES:
        await query.edit_message_text("⚠️ Invalid category. Cancelled.")
        _clear_draft(ctx)
        return ConversationHandler.END

    draft = _draft(ctx)
    draft["category"] = category

    source_type = draft.get("source_type", "")
    instructions = URL_INSTRUCTIONS.get(source_type, "Send the source URL or identifier.")

    await query.edit_message_text(
        f"{instructions}\n\n"
        f"<i>Send /cancel to abort.</i>",
        parse_mode="HTML",
    )
    return ASK_URL


async def add_source_url(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    """ASK_URL → ASK_NAME: user sent URL as text message."""
    raw_url = (update.message.text or "").strip()
    if not raw_url:
        await update.message.reply_text("⚠️ Empty URL. Send a valid URL or /cancel.")
        return ASK_URL
    if len(raw_url) > 500:
        await update.message.reply_text("⚠️ URL too long (>500 chars). Send a shorter one or /cancel.")
        return ASK_URL

    draft = _draft(ctx)
    source_type = draft.get("source_type", "")
    normalized = _normalize_url_input(source_type, raw_url)
    draft["source_url"] = normalized

    await update.message.reply_text(
        f"✓ URL: <code>{html_escape(normalized)}</code>\n\n"
        f"Now send a short <b>display name</b> for this source "
        f"(e.g. <i>Ben's Bites</i>, <i>YC News</i>, <i>Dan Luu blog</i>).",
        parse_mode="HTML",
    )
    return ASK_NAME


async def add_source_name(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    """ASK_NAME → ASK_CONFIRM: user sent name, show summary."""
    name = (update.message.text or "").strip()
    if not name:
        await update.message.reply_text("⚠️ Empty name. Send a name or /cancel.")
        return ASK_NAME
    if len(name) > 100:
        await update.message.reply_text("⚠️ Name too long (>100 chars). Send a shorter one or /cancel.")
        return ASK_NAME

    draft = _draft(ctx)
    draft["display_name"] = name

    type_label = SOURCE_TYPE_LABELS.get(draft.get("source_type", ""), draft.get("source_type", "?"))
    cat_label = SOURCE_CATEGORY_LABELS.get(draft.get("category", ""), draft.get("category", "?"))

    summary = (
        "<b>Add this source?</b>\n\n"
        f"Type:     {type_label}\n"
        f"Category: {cat_label}\n"
        f"URL:      <code>{html_escape(draft.get('source_url', ''))}</code>\n"
        f"Name:     <b>{html_escape(name)}</b>"
    )
    await update.message.reply_text(
        summary,
        reply_markup=add_source_confirm_kb(),
        parse_mode="HTML",
    )
    return ASK_CONFIRM


async def add_source_confirm(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    """ASK_CONFIRM → END: user confirmed, INSERT into user_sources."""
    query = update.callback_query
    await query.answer()

    if query.data != "addconf_yes":
        # treated as cancel — fall through to cancel path
        await query.edit_message_text("Cancelled. Source not added.")
        _clear_draft(ctx)
        return ConversationHandler.END

    draft = _draft(ctx)
    try:
        new_id = await add_user_source(
            source_type=draft["source_type"],
            source_url=draft["source_url"],
            display_name=draft["display_name"],
            category=draft["category"],
        )
    except Exception as e:  # noqa: BLE001
        log.exception("add_source insert failed")
        await query.edit_message_text(
            f"⚠️ Failed to add source: <code>{html_escape(str(e))}</code>",
            parse_mode="HTML",
        )
        _clear_draft(ctx)
        return ConversationHandler.END

    type_label = SOURCE_TYPE_LABELS.get(draft["source_type"], draft["source_type"])
    cat_label = SOURCE_CATEGORY_LABELS.get(draft["category"], draft["category"])
    await query.edit_message_text(
        f"✅ <b>Source #{new_id} added!</b>\n\n"
        f"{type_label} · {cat_label}\n"
        f"<b>{html_escape(draft['display_name'])}</b>\n"
        f"<code>{html_escape(draft['source_url'])}</code>\n\n"
        f"<i>It'll be included in the next digest collection.</i>",
        parse_mode="HTML",
    )
    _clear_draft(ctx)
    return ConversationHandler.END


async def add_source_cancel_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    """/cancel slash command — exits the conversation cleanly from any state."""
    _clear_draft(ctx)
    await update.message.reply_text("Cancelled. No source added.")
    return ConversationHandler.END


async def add_source_cancel_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel button — exits the conversation from any state."""
    query = update.callback_query
    await query.answer()
    _clear_draft(ctx)
    await query.edit_message_text("Cancelled. No source added.")
    return ConversationHandler.END


# Backwards compat helper used outside the conversation
def html_escape(s: str) -> str:
    import html
    return html.escape(s or "")


# ============================================================================
# Add-to-portfolio ConversationHandler (Step 20.5)
# ============================================================================
#
# Triggered by the "➕ В мой портфель" inline button under each investment
# card (callback_data="inv_add_<sig_id>"). Asks the user for quantity and
# average buy price, then INSERTs into portfolio_holdings.
#
# Supports two input formats for quantity:
#   - Plain number (e.g. "10" → 10 shares/units, normal mode)
#   - "usd <amount>" (e.g. "usd 1500" → $1500 invested, USD-only mode for ETFs
#     where the user knows only the $ amount, not share count). In this mode
#     the second question (price) is optional — entering "-" or "" means
#     "use current market price as price_at_add for future P&L drift".
#
# State storage: ctx.user_data["pending_holding"] = {asset_label, sig_id, ...}


def _holding_draft(ctx: ContextTypes.DEFAULT_TYPE) -> dict:
    return ctx.user_data.setdefault("pending_holding", {})


def _clear_holding_draft(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    ctx.user_data.pop("pending_holding", None)


async def add_holding_callback_entry(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE,
) -> int:
    """Entry point — callback ^inv_add_<sig_id>$ pressed by Maksim.

    Looks up the InvestmentSignal in bot_data to get the asset_label and the
    current market price (used as default if user skips the price prompt),
    then asks for quantity.
    """
    query = update.callback_query
    await query.answer()

    sig_id = (query.data or "")[len("inv_add_"):]
    store = _signal_store(ctx)
    sig: InvestmentSignal | None = store.get(sig_id)
    if sig is None:
        await query.message.reply_text(
            "⚠️ Сигнал не найден в текущем дайджесте (возможно, бот перезапускался). "
            "Добавь позицию руками: <code>/add_holding TICKER QTY PRICE</code>",
            parse_mode="HTML",
        )
        return ConversationHandler.END

    from ..portfolio import display_name, is_trackable_for_portfolio  # noqa: PLC0415

    # Whitelist check: only the 11 tickers we actually track are addable
    if not is_trackable_for_portfolio(sig.asset):
        await query.message.reply_text(
            f"⚠️ <b>{display_name(sig.asset)}</b> не в списке отслеживаемых для портфеля.\n\n"
            f"Бот трекает реальные позиции — добавлять можно только: "
            f"<code>CSPX, SMH, NATO, NUCL, EXH1, IB1T, ETH-CORE, IB01, CASH-USD, GOLD-PHYS, NVDA</code>.\n\n"
            f"Если хочешь экспозицию на {display_name(sig.asset)} — найди соответствующий "
            f"UCITS-ETF в твоём списке и добавь его.",
            parse_mode="HTML",
        )
        return ConversationHandler.END

    draft = _holding_draft(ctx)
    draft.clear()
    draft["sig_id"] = sig_id
    draft["asset_label"] = sig.asset
    draft["current_price"] = float(sig.price) if sig.price else 0.0
    draft["signal_type"] = sig.signal_type

    await query.message.reply_text(
        f"➕ <b>Добавляем {display_name(sig.asset)} в портфель</b>\n\n"
        f"Цена сейчас: <code>${sig.price:,.2f}</code>\n\n"
        f"Сколько штук купил?\n"
        f"• Число → <code>10</code> (10 штук)\n"
        f"• Или сумма в USD → <code>usd 1500</code> (для ETF без точного количества)\n\n"
        f"<i>Отмена: /cancel</i>",
        parse_mode="HTML",
    )
    return ASK_HOLDING_QTY


async def add_holding_qty(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    """ASK_HOLDING_QTY → ASK_HOLDING_PRICE — parse quantity or USD amount."""
    draft = _holding_draft(ctx)
    text = (update.message.text or "").strip().lower().replace(",", ".")

    if text.startswith("usd "):
        # USD-only mode
        try:
            usd_amount = float(text[4:].strip())
        except ValueError:
            await update.message.reply_text(
                "⚠️ Не могу прочитать сумму. Пример: <code>usd 1500</code>",
                parse_mode="HTML",
            )
            return ASK_HOLDING_QTY
        if usd_amount <= 0:
            await update.message.reply_text("⚠️ Сумма должна быть > 0.")
            return ASK_HOLDING_QTY
        draft["mode"] = "usd"
        draft["usd_invested"] = usd_amount
        current = draft.get("current_price") or 0.0
        await update.message.reply_text(
            f"💰 Сумма: <code>${usd_amount:,.2f}</code>\n\n"
            f"По какой средней цене входил?\n"
            f"• Число → <code>{current:.2f}</code> например\n"
            f"• Или <code>-</code> чтобы взять текущую (${current:,.2f}) — "
            f"P&L начнётся с 0%\n\n"
            f"<i>Отмена: /cancel</i>",
            parse_mode="HTML",
        )
        return ASK_HOLDING_PRICE

    # Plain quantity mode
    try:
        qty = float(text)
    except ValueError:
        await update.message.reply_text(
            "⚠️ Не могу прочитать число. Пример: <code>10</code> или <code>usd 1500</code>",
            parse_mode="HTML",
        )
        return ASK_HOLDING_QTY
    if qty <= 0:
        await update.message.reply_text("⚠️ Количество должно быть > 0.")
        return ASK_HOLDING_QTY
    draft["mode"] = "qty"
    draft["quantity"] = qty
    current = draft.get("current_price") or 0.0
    await update.message.reply_text(
        f"📦 Количество: <code>{qty:g}</code>\n\n"
        f"По какой средней цене входил? (USD)\n"
        f"• Число → <code>{current:.2f}</code>\n"
        f"• Или <code>-</code> чтобы взять текущую (${current:,.2f})\n\n"
        f"<i>Отмена: /cancel</i>",
        parse_mode="HTML",
    )
    return ASK_HOLDING_PRICE


async def add_holding_price(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE,
) -> int:
    """ASK_HOLDING_PRICE → END — parse price, insert into DB, confirm."""
    draft = _holding_draft(ctx)
    text = (update.message.text or "").strip().replace(",", ".")
    current_price = draft.get("current_price") or 0.0

    if text in ("-", ""):
        price = current_price
    else:
        try:
            price = float(text)
        except ValueError:
            await update.message.reply_text(
                "⚠️ Не могу прочитать цену. Пример: <code>185.50</code> или <code>-</code>",
                parse_mode="HTML",
            )
            return ASK_HOLDING_PRICE
        if price < 0:
            await update.message.reply_text("⚠️ Цена должна быть ≥ 0.")
            return ASK_HOLDING_PRICE

    asset_label = draft.get("asset_label", "?")
    mode = draft.get("mode", "qty")
    from ..portfolio import display_name  # noqa: PLC0415
    dname = display_name(asset_label)

    try:
        if mode == "usd":
            from ..portfolio import add_usd_holding  # noqa: PLC0415
            usd_inv = float(draft.get("usd_invested") or 0.0)
            await add_usd_holding(
                asset_label=asset_label,
                usd_invested=usd_inv,
                price_at_add=price if price > 0 else None,
                notes=f"added via inline button on signal {draft.get('sig_id', '?')}",
            )
            await update.message.reply_text(
                f"✅ Добавлено: <b>{dname}</b> — "
                f"${usd_inv:,.2f} @ ${price:,.2f}\n\n"
                f"<i>Посмотри полный портфель: /portfolio</i>",
                parse_mode="HTML",
            )
        else:
            qty = float(draft.get("quantity") or 0.0)
            await add_or_update_holding(
                asset_label=asset_label,
                quantity=qty,
                buy_price_usd=price,
                notes=f"added via inline button on signal {draft.get('sig_id', '?')}",
                price_at_add=price if price > 0 else None,
            )
            await update.message.reply_text(
                f"✅ Добавлено: <b>{dname}</b> — "
                f"{qty:g} шт @ ${price:,.2f}\n\n"
                f"<i>Посмотри полный портфель: /portfolio</i>",
                parse_mode="HTML",
            )
    except ValueError as e:
        await update.message.reply_text(
            f"⚠️ Не получилось: <code>{html_escape(str(e))}</code>",
            parse_mode="HTML",
        )
    except Exception as e:  # noqa: BLE001
        log.exception("add_holding via callback failed")
        await update.message.reply_text(
            f"⚠️ Ошибка: <code>{html_escape(str(e))}</code>",
            parse_mode="HTML",
        )

    _clear_holding_draft(ctx)
    return ConversationHandler.END


async def add_holding_cancel_cmd(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE,
) -> int:
    """/cancel inside the add-holding flow."""
    _clear_holding_draft(ctx)
    await update.message.reply_text(
        "❌ Отменено. Позиция не добавлена."
    )
    return ConversationHandler.END


# ============================================================================
# Portfolio adjust (+$/-$) ConversationHandler
# ============================================================================


async def pf_adjust_callback_entry(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE,
) -> int:
    """Entry: pf_addusd_<LABEL> or pf_subusd_<LABEL>."""
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if data.startswith("pf_addusd_"):
        mode = "add"
        label = data[len("pf_addusd_"):]
    elif data.startswith("pf_subusd_"):
        mode = "sub"
        label = data[len("pf_subusd_"):]
    else:
        return ConversationHandler.END

    ctx.user_data["pf_adjust"] = {"label": label, "mode": mode}
    from ..portfolio import display_name  # noqa: PLC0415

    verb = "Добавить" if mode == "add" else "Снять"
    sign = "+" if mode == "add" else "-"
    await query.message.reply_text(
        f"{sign}💰 <b>{verb} деньги: {display_name(label)}</b>\n\n"
        f"Сколько USD?\n"
        f"• Число → <code>500</code> ({verb.lower()} $500)\n\n"
        f"<i>Отмена: /cancel</i>",
        parse_mode="HTML",
    )
    return ASK_ADJUST_AMOUNT


async def pf_adjust_amount(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    """Parse amount, call adjust_usd_invested, confirm."""
    draft = ctx.user_data.get("pf_adjust") or {}
    label = draft.get("label", "")
    mode = draft.get("mode", "add")

    text = (update.message.text or "").strip().replace(",", ".").replace("$", "")
    try:
        amount = float(text)
    except ValueError:
        await update.message.reply_text(
            "⚠️ Не могу прочитать число. Пример: <code>500</code>",
            parse_mode="HTML",
        )
        return ASK_ADJUST_AMOUNT
    if amount <= 0:
        await update.message.reply_text("⚠️ Сумма должна быть > 0.")
        return ASK_ADJUST_AMOUNT

    delta = amount if mode == "add" else -amount
    from ..portfolio import (  # noqa: PLC0415
        adjust_usd_invested,
        display_name,
        remove_holding,
    )

    try:
        updated = await adjust_usd_invested(label, delta)
    except Exception as e:  # noqa: BLE001
        log.exception("pf_adjust failed")
        await update.message.reply_text(
            f"⚠️ Ошибка: <code>{html_escape(str(e))}</code>",
            parse_mode="HTML",
        )
        ctx.user_data.pop("pf_adjust", None)
        return ConversationHandler.END

    if updated is None and mode == "sub":
        # Final would be negative → offer to remove
        await update.message.reply_text(
            f"⚠️ Снятие ${amount:,.0f} опустит позицию ниже нуля.\n"
            f"Нажми 🗑️ <b>Удалить</b> в /portfolio чтобы убрать её полностью, "
            f"или попробуй снять меньшую сумму.",
            parse_mode="HTML",
        )
        ctx.user_data.pop("pf_adjust", None)
        return ConversationHandler.END

    if updated is None:
        await update.message.reply_text("⚠️ Позиция не найдена.")
        ctx.user_data.pop("pf_adjust", None)
        return ConversationHandler.END

    new_usd = float(updated.get("usd_invested") or 0)
    verb = "добавлено" if mode == "add" else "снято"
    await update.message.reply_text(
        f"✅ <b>{display_name(label)}</b>: {verb} ${amount:,.0f}\n"
        f"💰 Текущая позиция: ${new_usd:,.0f}\n\n"
        f"<i>Применится в следующем /morning и /digest.</i>",
        parse_mode="HTML",
    )
    ctx.user_data.pop("pf_adjust", None)
    return ConversationHandler.END


async def pf_adjust_cancel_cmd(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE,
) -> int:
    """/cancel inside the +/- adjust flow."""
    ctx.user_data.pop("pf_adjust", None)
    await update.message.reply_text("❌ Отменено.")
    return ConversationHandler.END


async def pf_addnew_entry(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE,
) -> int:
    """Entry: '➕ Добавить новую позицию' inline button.

    Shows a keyboard with all whitelist tickers NOT yet in portfolio.
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup  # noqa: PLC0415

    from ..portfolio import (  # noqa: PLC0415
        TRACKABLE_PORTFOLIO_TICKERS,
        display_name,
        list_holdings,
    )

    query = update.callback_query
    await query.answer()

    existing = {h["asset_label"].upper() for h in await list_holdings()}
    addable = [t for t in TRACKABLE_PORTFOLIO_TICKERS if t.upper() not in existing]

    if not addable:
        await query.message.reply_text(
            "📭 <i>Все 11 трекаемых тикеров уже в портфеле. "
            "Удалить ненужный через 🗑️ или используй 💸 -$ чтобы обнулить.</i>",
            parse_mode="HTML",
        )
        return ConversationHandler.END

    # Build keyboard 2 per row for readability
    buttons = [
        InlineKeyboardButton(display_name(t), callback_data=f"pf_pick_{t}")
        for t in addable
    ]
    rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    rows.append([InlineKeyboardButton("❌ Отмена", callback_data="pf_pick_cancel")])

    await query.message.reply_text(
        "➕ <b>Выбери тикер для новой позиции:</b>\n\n"
        "<i>Только из whitelist — это активы, которые мы реально трекаем "
        "(цены, новости, P&amp;L).</i>",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(rows),
    )
    return PICK_NEW_TICKER


async def pf_addnew_pick(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE,
) -> int:
    """User picked a ticker → ask for $ amount."""
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if data == "pf_pick_cancel":
        ctx.user_data.pop("pf_new", None)
        await query.edit_message_text("❌ Отменено.")
        return ConversationHandler.END

    label = data[len("pf_pick_"):]
    from ..portfolio import display_name, is_trackable_for_portfolio  # noqa: PLC0415

    if not is_trackable_for_portfolio(label):
        await query.edit_message_text(
            f"⚠️ <code>{label}</code> не в whitelist'е. Отменено.",
            parse_mode="HTML",
        )
        return ConversationHandler.END

    ctx.user_data["pf_new"] = {"label": label}
    await query.message.reply_text(
        f"➕ <b>Новая позиция: {display_name(label)}</b>\n\n"
        f"Сколько USD ты вложил?\n"
        f"• Число → <code>500</code> ($500)\n"
        f"• Или <code>0</code> если хочешь только трекать без денег "
        f"(например физ. золото)\n\n"
        f"<i>Отмена: /cancel</i>",
        parse_mode="HTML",
    )
    return ASK_NEW_AMOUNT


async def pf_addnew_amount(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE,
) -> int:
    """Parse amount, INSERT, confirm."""
    draft = ctx.user_data.get("pf_new") or {}
    label = draft.get("label", "")
    if not label:
        await update.message.reply_text("⚠️ Сессия потеряна. Запусти /portfolio заново.")
        return ConversationHandler.END

    text = (update.message.text or "").strip().replace(",", ".").replace("$", "")
    try:
        usd = float(text)
    except ValueError:
        await update.message.reply_text(
            "⚠️ Не могу прочитать число. Пример: <code>500</code>",
            parse_mode="HTML",
        )
        return ASK_NEW_AMOUNT
    if usd < 0:
        await update.message.reply_text("⚠️ Сумма должна быть ≥ 0.")
        return ASK_NEW_AMOUNT

    from ..agents.market import collect_market_data  # noqa: PLC0415
    from ..portfolio import (  # noqa: PLC0415
        _lookup_current_price,
        add_usd_holding,
        display_name,
    )

    # Capture entry-price snapshot so P&L drift works
    price_at_add = None
    try:
        market_data, _ = await collect_market_data()
        price_at_add = _lookup_current_price(label, market_data)
    except Exception as e:  # noqa: BLE001
        log.debug("pf_addnew: market fetch failed: %s", e)

    try:
        await add_usd_holding(
            asset_label=label,
            usd_invested=usd,
            price_at_add=price_at_add,
            notes=f"added via /portfolio ➕ button",
        )
    except Exception as e:  # noqa: BLE001
        log.exception("pf_addnew: insert failed")
        await update.message.reply_text(
            f"⚠️ Ошибка: <code>{html_escape(str(e))}</code>",
            parse_mode="HTML",
        )
        ctx.user_data.pop("pf_new", None)
        return ConversationHandler.END

    ctx.user_data.pop("pf_new", None)
    price_str = f" @ ${price_at_add:,.2f}" if price_at_add else ""
    await update.message.reply_text(
        f"✅ <b>{display_name(label)}</b> добавлена в портфель: "
        f"${usd:,.0f}{price_str}\n\n"
        f"<i>Появится в следующем /morning и /digest. "
        f"Посмотри: /portfolio</i>",
        parse_mode="HTML",
    )
    return ConversationHandler.END


async def pf_addnew_cancel_cmd(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE,
) -> int:
    ctx.user_data.pop("pf_new", None)
    await update.message.reply_text("❌ Отменено.")
    return ConversationHandler.END


async def _handle_pf_remove(
    ctx: ContextTypes.DEFAULT_TYPE,
    query: Any,
    label: str,
) -> None:
    """🗑️ Удалить — drop position from portfolio_holdings."""
    from ..portfolio import display_name, remove_holding  # noqa: PLC0415

    removed = await remove_holding(label)
    if removed:
        await query.edit_message_text(
            f"🗑️ <b>{display_name(label)}</b> удалена из портфеля.\n"
            f"<i>В следующем /morning и /digest её больше не будет.</i>",
            parse_mode="HTML",
        )
    else:
        await query.edit_message_text(
            f"⚠️ Позиция <b>{label}</b> не найдена (возможно, уже удалена).",
            parse_mode="HTML",
        )


# ============================================================================
# /sources card button handlers (Step 8)
# ============================================================================


async def _handle_source_stats(query: Any) -> None:
    """📊 Source stats button — shows aggregated counts."""
    stats = await _gather_source_stats()
    await query.message.reply_text(
        render_source_stats_card(stats),
        parse_mode="HTML",
    )


async def _handle_remove_source_list(query: Any) -> None:
    """🗑️ Remove source button — show list of sources, one button per source."""
    sources = await load_active_sources()
    if not sources:
        await query.message.reply_text(
            "📭 No sources to remove. Tap ➕ Add source to add one first."
        )
        return
    await query.message.reply_text(
        f"🗑 <b>Pick a source to remove</b> ({len(sources)} active):",
        reply_markup=remove_source_list_kb(sources),
        parse_mode="HTML",
    )


async def _handle_remove_source_action(query: Any, payload: str) -> None:
    """rmsrc_<id> or rmsrc_cancel — execute the removal."""
    if payload == "cancel":
        await query.edit_message_text("Cancelled. No source removed.")
        return
    try:
        source_pk = int(payload)
    except ValueError:
        await query.edit_message_text(f"⚠️ Invalid source ID: <code>{payload}</code>", parse_mode="HTML")
        return
    n = await remove_user_source(source_pk)
    if n:
        await query.edit_message_text(
            f"✅ Removed source #{source_pk}.\n\n"
            f"<i>It won't appear in future digests.</i>",
            parse_mode="HTML",
        )
    else:
        await query.edit_message_text(f"⚠️ No source with id={source_pk}.", parse_mode="HTML")


async def cmd_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    topic = " ".join(ctx.args) if ctx.args else None
    if not topic:
        await update.message.reply_text("Usage: <code>/history &lt;topic&gt;</code>", parse_mode="HTML")
        return
    await update.message.reply_text(
        f"🚧 <i>/history for </i><b>{topic}</b><i> ships in Step 9 with topic_timeline tracking.</i>",
        parse_mode="HTML",
    )


async def cmd_saved(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show saved items pulled from the feedback table (works in Step 3)."""
    async with get_db() as conn:
        async with conn.execute(
            "SELECT item_type, item_snapshot, created_at FROM feedback "
            "WHERE feedback_type = 'save' ORDER BY created_at DESC LIMIT 10"
        ) as cur:
            rows = await cur.fetchall()
    if not rows:
        await update.message.reply_text(
            "📭 No saved items yet. Tap 📌 <b>Save</b> on any idea or signal.",
            parse_mode="HTML",
        )
        return
    lines = ["📌 <b>Your saved items:</b>\n"]
    for row in rows:
        snapshot = json.loads(row["item_snapshot"])
        title = snapshot.get("title") or snapshot.get("asset") or "(unknown)"
        lines.append(f"• [{row['item_type']}] {title}")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Feedback stats + learning calibration state (Step 14)."""
    async with get_db() as conn:
        async with conn.execute(
            "SELECT feedback_type, COUNT(*) FROM feedback GROUP BY feedback_type"
        ) as cur:
            rows = await cur.fetchall()

    parts: list[str] = ["📊 <b>Feedback stats</b>\n"]

    if rows:
        for row in rows:
            parts.append(f"• {row[0]}: {row[1]}")
    else:
        parts.append("<i>No feedback yet.</i>")

    # Learning calibration state
    total = await total_feedback_count()
    last = await feedbacks_at_last_calibration()
    last_at = await last_calibration_at()
    pending = max(0, total - last)
    parts.append("")
    parts.append("🧠 <b>Learning calibration</b>")
    parts.append(f"• Total feedbacks: {total}")
    parts.append(f"• Since last calibration: {pending}")
    parts.append(f"• Threshold: every {CALIBRATION_THRESHOLD}")
    if last_at:
        parts.append(f"• Last calibrated: <code>{last_at[:19].replace('T', ' ')} UTC</code>")
    else:
        parts.append("• Last calibrated: <i>never</i>")

    # Weekly summary from latest calibration (if any)
    summary = await get_weekly_summary()
    if summary:
        parts.append("")
        parts.append("📈 <b>Latest weekly summary</b>")
        parts.append(html_escape(summary))
    else:
        parts.append("")
        parts.append("<i>Tap </i>/calibrate<i> to run learning calibration manually "
                     "(needs at least a few feedbacks + OPENAI_API_KEY).</i>")

    # Step 17: one-line cost summary (details via /cost)
    try:
        cost = await get_cost_summary(days=30)
        if cost["total_calls"]:
            parts.append("")
            parts.append(
                f"💰 <b>LLM cost (30d):</b> ${cost['total_cost_usd']:.4f} "
                f"· {cost['total_calls']} calls · /cost for details"
            )
    except Exception as e:  # noqa: BLE001
        log.debug("cmd_stats: cost summary failed: %s", e)

    await update.message.reply_text("\n".join(parts), parse_mode="HTML")


async def cmd_cost(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Detailed LLM cost breakdown (Step 17)."""
    args = ctx.args
    try:
        days = int(args[0]) if args else 30
    except ValueError:
        days = 30
    days = max(1, min(days, 365))

    cost = await get_cost_summary(days=days)

    if not cost["total_calls"]:
        await update.message.reply_text(
            f"💰 <b>LLM cost — last {days} days</b>\n\n"
            f"<i>No LLM calls logged yet. Set </i><code>OPENAI_API_KEY</code>"
            f"<i> in .env and run </i>/digest<i>.</i>",
            parse_mode="HTML",
        )
        return

    parts: list[str] = [
        f"💰 <b>LLM cost — last {days} days</b>\n",
        f"• Total calls: <b>{cost['total_calls']}</b>",
        f"• Total cost: <b>${cost['total_cost_usd']:.4f}</b>",
        f"• Input tokens: {cost['total_input_tokens']:,}",
        f"• Output tokens: {cost['total_output_tokens']:,}",
    ]

    # Monthly projection (naive extrapolation if days < 30)
    if days < 30 and cost["total_cost_usd"] > 0:
        projected = cost["total_cost_usd"] * (30 / days)
        parts.append(f"• Projected 30d: <b>${projected:.4f}</b>")

    if cost["by_agent"]:
        parts.append("")
        parts.append("<b>By agent:</b>")
        for agent, calls, agent_cost in cost["by_agent"]:
            pct = (agent_cost / cost["total_cost_usd"] * 100) if cost["total_cost_usd"] > 0 else 0
            parts.append(
                f"• <code>{agent}</code>: {calls} calls · "
                f"${agent_cost:.4f} ({pct:.0f}%)"
            )

    if cost["by_day"]:
        parts.append("")
        parts.append("<b>By day (recent):</b>")
        for day, day_cost in cost["by_day"][:7]:
            parts.append(f"• <code>{day}</code>: ${day_cost:.4f}")

    await update.message.reply_text("\n".join(parts), parse_mode="HTML")


async def cmd_calibrate(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Force-run a learning calibration NOW (Step 14)."""
    await update.message.reply_text(
        "🧠 Running calibration on your recent feedback...",
        parse_mode="HTML",
    )

    total = await total_feedback_count()
    if total == 0:
        await update.message.reply_text(
            "📭 No feedback yet. Tap 🔥/❌/📌 buttons on a /digest first.",
        )
        return

    result = await calibrate_from_feedback()
    if not result:
        await update.message.reply_text(
            "⚠️ Calibration did not run.\n\n"
            "Possible reasons:\n"
            "• <code>OPENAI_API_KEY</code> not set in .env\n"
            "• OpenAI API call failed (check bot logs)\n"
            "• <code>openai</code> package not installed",
            parse_mode="HTML",
        )
        return

    msg = (
        f"✅ <b>Calibration complete</b>\n\n"
        f"Analyzed: <b>{result.total_analyzed}</b> feedbacks "
        f"({result.likes} 🔥 / {result.dislikes} ❌ / {result.saves} 📌)\n\n"
        f"<b>Updated preferences (will steer next /digest):</b>\n"
        f"<i>{html_escape(result.prompt_injection_ideas)}</i>\n\n"
        f"<b>Weekly summary:</b>\n"
        f"{html_escape(result.weekly_summary)}"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def _gather_source_stats() -> dict:
    """Aggregate stats for the /source_stats card."""
    async with get_db() as conn:
        async with conn.execute(
            "SELECT COUNT(*) FROM user_sources WHERE is_active = 1"
        ) as cur:
            active = (await cur.fetchone())[0]
        async with conn.execute(
            "SELECT COUNT(*) FROM user_sources WHERE is_active = 0"
        ) as cur:
            inactive = (await cur.fetchone())[0]
        async with conn.execute("SELECT COUNT(*) FROM signals") as cur:
            total = (await cur.fetchone())[0]
        async with conn.execute(
            "SELECT COUNT(*) FROM signals WHERE is_breaking = 1"
        ) as cur:
            breaking = (await cur.fetchone())[0]
        async with conn.execute(
            "SELECT source_type, COUNT(*) AS n FROM signals "
            "GROUP BY source_type ORDER BY n DESC LIMIT 10"
        ) as cur:
            by_type = [(r["source_type"], r["n"]) for r in await cur.fetchall()]
        async with conn.execute(
            "SELECT us.display_name, COUNT(s.id) AS n "
            "FROM user_sources us LEFT JOIN signals s "
            "ON s.source_id LIKE '%/' || us.display_name "
            "  OR s.source_id LIKE '%' || us.display_name "
            "WHERE us.is_active = 1 "
            "GROUP BY us.id ORDER BY n DESC LIMIT 5"
        ) as cur:
            top_user = [(r["display_name"], r["n"]) for r in await cur.fetchall()]

    return {
        "user_sources_active": active,
        "user_sources_inactive": inactive,
        "signals_total": total,
        "signals_breaking": breaking,
        "by_source_type": by_type,
        "top_user_sources": top_user,
    }


async def cmd_settings(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🚧 <i>/settings UI ships in Step 14.</i>\n"
        "For now, edit <code>IDEA_FOCUS_WEIGHT</code> in <code>.env</code> "
        "(default 0.70 = 70% ideas / 30% investments).",
        parse_mode="HTML",
    )


async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Pause or resume the morning/evening scheduled digests (Step 15).

    Usage:
      /pause          → show current state
      /pause 3        → pause for 3 days
      /pause 0        → resume immediately
    """
    args = ctx.args

    # No args → show current state
    if not args:
        pause_iso = await get_pause_until_iso()
        if not pause_iso:
            await update.message.reply_text(
                "▶️ <b>ORACLE is running.</b>\n\n"
                "Morning brief @ 07:00 · Evening digest @ 19:00 (Warsaw).\n\n"
                "Use <code>/pause 3</code> to pause for 3 days, "
                "<code>/pause 0</code> to resume.",
                parse_mode="HTML",
            )
            return
        try:
            until = datetime.fromisoformat(pause_iso)
            now = datetime.now(timezone.utc)
            if until > now:
                remaining_h = int((until - now).total_seconds() / 3600)
                remaining_d = remaining_h // 24
                await update.message.reply_text(
                    f"⏸ <b>ORACLE is paused</b>\n\n"
                    f"Resumes: <code>{until.strftime('%Y-%m-%d %H:%M UTC')}</code>\n"
                    f"Remaining: ~{remaining_d}d {remaining_h % 24}h\n\n"
                    f"Use <code>/pause 0</code> to resume now.",
                    parse_mode="HTML",
                )
            else:
                await update.message.reply_text(
                    "▶️ ORACLE is running (paused timestamp expired).",
                )
        except ValueError:
            await update.message.reply_text(
                "⚠️ Pause state corrupt. Use <code>/pause 0</code> to reset.",
                parse_mode="HTML",
            )
        return

    # /pause N
    try:
        days = int(args[0])
    except ValueError:
        await update.message.reply_text(
            "Usage: <code>/pause &lt;days&gt;</code>\n\n"
            "Examples:\n"
            "• <code>/pause 3</code> — pause for 3 days\n"
            "• <code>/pause 0</code> — resume now\n"
            "• <code>/pause</code> — show current state",
            parse_mode="HTML",
        )
        return

    if days < 0:
        await update.message.reply_text("⚠️ Days must be ≥ 0")
        return

    if days == 0:
        await set_pause_days(0)
        await update.message.reply_text(
            "▶️ <b>ORACLE resumed.</b>\n"
            "Morning brief + evening digest back on schedule.",
            parse_mode="HTML",
        )
        return

    until_iso = await set_pause_days(days)
    try:
        until = datetime.fromisoformat(until_iso)
        until_human = until.strftime('%Y-%m-%d %H:%M UTC')
    except ValueError:
        until_human = until_iso

    await update.message.reply_text(
        f"⏸ <b>ORACLE paused for {days} day(s).</b>\n\n"
        f"Will resume at: <code>{until_human}</code>\n\n"
        f"Manual commands (/digest, /morning) still work. "
        f"Only the scheduled cron jobs are paused.",
        parse_mode="HTML",
    )


# ============================================================================
# /portfolio /add_holding /remove_holding — Step 20
# ============================================================================
#
# Portfolio commands let Maksim track his actual positions so investment_analyzer
# can give PERSONALIZED scenarios based on what he holds + live P&L.
#
# /portfolio                                       — show current holdings + P&L
# /add_holding <ASSET> <QTY> <BUY_PRICE> [notes]  — add or weighted-merge a position
# /remove_holding <ASSET>                          — drop a position
#
# Asset labels must match market.py catalogs: BTC, ETH, SOL, NVDA, AAPL, GOLD,
# OIL_WTI, NATGAS, EURPLN, USDBYN, etc. Plus CASH_USD/EUR/PLN/BYN for cash.


def _fmt_money(v: float | None, *, decimals: int = 2) -> str:
    if v is None:
        return "—"
    return f"${v:,.{decimals}f}"


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"


def _render_portfolio_html(portfolio: dict) -> str:
    """Render the /portfolio Telegram card."""
    holdings = portfolio.get("holdings") or []
    totals = portfolio.get("totals") or {}
    by_class = portfolio.get("by_class") or {}

    if not holdings:
        return (
            "📊 <b>Your portfolio is empty</b>\n\n"
            "Add a position with:\n"
            "<code>/add_holding BTC 0.15 52000 long-term</code>\n"
            "<code>/add_holding NVDA 12 480</code>\n"
            "<code>/add_holding GOLD 5 2100 hedge</code>\n"
            "<code>/add_holding CASH_USD 5000 1</code>\n\n"
            "<i>Asset labels must match what ORACLE tracks. "
            "Use /add_holding without args to see the full list.</i>"
        )

    tot_mv = totals.get("market_value_usd", 0.0)
    tot_cost = totals.get("cost_basis_usd", 0.0)
    tot_pnl = totals.get("unrealized_pnl_usd", 0.0)
    tot_pct = totals.get("pnl_pct", 0.0)
    pnl_emoji = "📈" if tot_pnl >= 0 else "📉"

    parts: list[str] = [
        "📊 <b>Your portfolio</b>\n",
        f"<b>Total value:</b> {_fmt_money(tot_mv, decimals=0)}",
        f"<b>Cost basis:</b> {_fmt_money(tot_cost, decimals=0)}",
        f"<b>Unrealized P&amp;L:</b> {pnl_emoji} {_fmt_money(tot_pnl, decimals=0)} ({_fmt_pct(tot_pct)})",
        "",
    ]

    if by_class:
        parts.append("<b>By asset class:</b>")
        for cls, b in sorted(
            by_class.items(),
            key=lambda kv: kv[1].get("market_value_usd", 0),
            reverse=True,
        ):
            mv = b.get("market_value_usd", 0)
            cnt = int(b.get("count", 0))
            pct = (mv / tot_mv * 100) if tot_mv else 0
            parts.append(f"• {cls}: {_fmt_money(mv, decimals=0)} ({pct:.0f}%) · {cnt} pos")
        parts.append("")

    from ..portfolio import display_name  # noqa: PLC0415

    parts.append("<b>Positions:</b>")
    for h in holdings:
        label = h["asset_label"]
        dname = display_name(label)
        qty = h["quantity"] or 0
        avg = h["avg_buy_price"] or 0
        cur = h.get("current_price")
        mv = h.get("market_value_usd")
        pnl_pct = h.get("pnl_pct")
        status = h.get("price_status", "?")
        usd_inv = h.get("usd_invested") or 0
        isin = h.get("isin")
        notes = h.get("notes") or ""

        if status == "cash":
            cash_amt = float(usd_inv) if usd_inv else float(qty)
            line = f"• <b>{dname}</b>: {_fmt_money(cash_amt, decimals=2)}"
        elif status == "live":
            line = (
                f"• <b>{dname}</b>: {qty:g} @ {_fmt_money(avg)} avg → "
                f"now {_fmt_money(cur)} · MV {_fmt_money(mv, decimals=0)} · {_fmt_pct(pnl_pct)}"
            )
        elif status == "usd_drift":
            entry = h.get("price_at_add") or 0
            line = (
                f"• <b>{dname}</b>: {_fmt_money(usd_inv, decimals=0)} invested → "
                f"now {_fmt_money(mv, decimals=0)} ({_fmt_pct(pnl_pct)} vs entry @ {_fmt_money(entry)})"
            )
        elif status == "usd_basis":
            line = (
                f"• <b>{dname}</b>: {_fmt_money(usd_inv, decimals=0)} invested "
                f"<i>(no entry-price snapshot — P&L unknown)</i>"
            )
        else:
            # Fallback for stale qty-mode rows (no live price)
            if usd_inv and not qty:
                line = (
                    f"• <b>{dname}</b>: {_fmt_money(usd_inv, decimals=0)} invested "
                    f"<i>(no live price)</i>"
                )
            else:
                line = (
                    f"• <b>{dname}</b>: {qty:g} @ {_fmt_money(avg)} avg "
                    f"<i>(no live price)</i>"
                )
        if isin:
            line += f" <code>{html_escape(isin)}</code>"
        if notes:
            line += f" <i>({html_escape(notes)})</i>"
        parts.append(line)

    parts.append("")
    parts.append(
        "<i>Investment analyzer will use these positions to personalize signals "
        "in your next /digest.</i>"
    )
    return "\n".join(parts)


async def cmd_portfolio(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current holdings with live P&L + per-holding management buttons.

    Always reads FRESH data from the DB. Morning brief and digest also
    re-read from DB on every run, so manual edits via these buttons are
    picked up immediately by the next /morning or /digest.
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup  # noqa: PLC0415

    from ..portfolio import display_name  # noqa: PLC0415

    await update.message.reply_text(
        "📡 Подгружаю свежие цены...",
        parse_mode="HTML",
    )
    try:
        market_data, errors = await collect_market_data()
        if errors:
            log.warning("portfolio: market collection had %d errors", len(errors))
        portfolio = await get_portfolio_with_pnl(market_data)
    except Exception as e:  # noqa: BLE001
        log.exception("cmd_portfolio failed")
        await update.message.reply_text(
            f"⚠️ Could not load portfolio: <code>{html_escape(str(e))}</code>",
            parse_mode="HTML",
        )
        return

    # 1. Aggregate overview
    await update.message.reply_text(
        _render_portfolio_html(portfolio),
        parse_mode="HTML",
        disable_web_page_preview=True,
    )

    # 2. Per-holding management cards (compact)
    holdings = portfolio.get("holdings") or []
    if not holdings:
        return

    for h in holdings:
        label = h["asset_label"]
        dname = display_name(label)
        usd_inv = h.get("usd_invested") or 0
        mv = h.get("market_value_usd") or usd_inv
        pnl_pct = h.get("pnl_pct")
        c24 = h.get("change_24h_pct")

        sign24 = "+" if (c24 or 0) >= 0 else ""
        live_24h = f"{sign24}{c24:.1f}%" if c24 is not None else "—"
        sign_pnl = "+" if (pnl_pct or 0) >= 0 else ""
        pnl_str = f"{sign_pnl}{pnl_pct:.1f}% от входа" if pnl_pct is not None else ""

        line = f"<b>{html_escape(dname)}</b>"
        if usd_inv:
            line += f"\n💰 ${usd_inv:,.0f} invested"
        if mv and mv != usd_inv:
            line += f" · сейчас ${mv:,.0f}"
        line += f"\n📊 24h: {live_24h}"
        if pnl_str:
            line += f" · {pnl_str}"

        # Inline buttons row
        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("💰 +$", callback_data=f"pf_addusd_{label}"),
                InlineKeyboardButton("💸 -$", callback_data=f"pf_subusd_{label}"),
            ],
            [
                InlineKeyboardButton("🗑️ Удалить", callback_data=f"pf_remove_{label}"),
            ],
        ])
        try:
            await update.message.reply_text(line, parse_mode="HTML", reply_markup=kb)
        except Exception as e:  # noqa: BLE001
            log.warning("cmd_portfolio: render %s failed: %s", label, e)

    # Footer + always-show "Add new position" button
    from ..portfolio import TRACKABLE_PORTFOLIO_TICKERS  # noqa: PLC0415
    existing = {(h.get("asset_label") or "").upper() for h in holdings}
    addable = [t for t in TRACKABLE_PORTFOLIO_TICKERS if t.upper() not in existing]

    legend_kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(
            f"➕ Добавить новую позицию ({len(addable)} доступно)",
            callback_data="pf_addnew",
        )],
    ])

    await update.message.reply_text(
        "<i>💰 +$ — добавить денег к позиции\n"
        "💸 -$ — снять деньги с позиции\n"
        "🗑️ — удалить позицию полностью\n"
        "➕ — добавить новый тикер из whitelist\n\n"
        "Изменения сразу попадают в /morning и /digest.</i>",
        parse_mode="HTML",
        reply_markup=legend_kb,
    )


ADD_HOLDING_USAGE = (
    "💼 <b>/add_holding</b> — add or update a position\n\n"
    "<b>Usage:</b>\n"
    "<code>/add_holding ASSET QUANTITY BUY_PRICE_USD [notes]</code>\n\n"
    "<b>Examples:</b>\n"
    "• <code>/add_holding BTC 0.15 52000 long-term hold</code>\n"
    "• <code>/add_holding NVDA 12 480 AI exposure</code>\n"
    "• <code>/add_holding GOLD 5 2100 hedge</code>\n"
    "• <code>/add_holding OIL_WTI 10 75 swing trade</code>\n"
    "• <code>/add_holding EURPLN 1000 4.30 vacation fund</code>\n"
    "• <code>/add_holding CASH_USD 5000 1</code>\n\n"
    "<b>If you re-add an asset, the avg buy price is weighted-averaged.</b>\n\n"
    "<b>Tracked assets:</b>\n"
    "<i>Crypto:</i> BTC, ETH, SOL, XRP\n"
    "<i>Mega-cap + AI stocks:</i> NVDA, MSFT, GOOGL, AAPL, TSLA, AMD, META, AMZN, "
    "TSM, AVGO, NFLX, PLTR, SMCI, ARM, ASML, MU, VRT, SOUN, AI\n"
    "<i>Nuclear:</i> SMR, OKLO, NNE, LEU, CCJ, BWXT, UEC, URA, VST, CEG\n"
    "<i>Drones &amp; defense:</i> AVAV, KTOS, RCAT, ONDS, RKLB, EH, UMAC\n"
    "<i>Trump/political:</i> DJT, RUM, PSQH, PHUN\n"
    "<i>ETFs:</i> SPY, QQQ, XLK, XLE, XLF, VNQ\n"
    "<i>Commodities:</i> GOLD, SILVER, OIL_WTI, OIL_BRENT, NATGAS, COPPER, WHEAT\n"
    "<i>Forex:</i> EURUSD, USDPLN, EURPLN, USDBYN, EURBYN, USDRUB, DXY\n"
    "<i>Cash:</i> CASH_USD, CASH_EUR, CASH_PLN, CASH_BYN"
)


async def cmd_add_holding(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Add or update a portfolio position."""
    args = ctx.args or []
    if len(args) < 3:
        await update.message.reply_text(ADD_HOLDING_USAGE, parse_mode="HTML")
        return

    asset_label = args[0].upper().strip()
    try:
        quantity = float(args[1].replace(",", "."))
        buy_price = float(args[2].replace(",", "."))
    except ValueError:
        await update.message.reply_text(
            "⚠️ QUANTITY and BUY_PRICE must be numbers.\n\n" + ADD_HOLDING_USAGE,
            parse_mode="HTML",
        )
        return

    notes = " ".join(args[3:]) if len(args) > 3 else None

    from ..portfolio import is_trackable_for_portfolio  # noqa: PLC0415

    ok, asset_class_or_err = validate_asset_label(asset_label)
    if not ok:
        await update.message.reply_text(
            f"⚠️ Unknown asset <code>{html_escape(asset_label)}</code>.\n\n"
            f"{ADD_HOLDING_USAGE}",
            parse_mode="HTML",
        )
        return

    if not is_trackable_for_portfolio(asset_label):
        await update.message.reply_text(
            f"⚠️ <code>{asset_label}</code> не в whitelist'е портфеля.\n\n"
            f"Бот трекает реальные позиции — добавлять можно только:\n"
            f"<code>CSPX, SMH, NATO, NUCL, EXH1, IB1T, ETH-CORE, IB01, "
            f"CASH-USD, GOLD-PHYS, NVDA</code>",
            parse_mode="HTML",
        )
        return

    if quantity <= 0:
        await update.message.reply_text("⚠️ Quantity must be > 0.")
        return
    if buy_price < 0:
        await update.message.reply_text("⚠️ Buy price must be ≥ 0.")
        return

    try:
        row = await add_or_update_holding(
            asset_label=asset_label,
            quantity=quantity,
            buy_price_usd=buy_price,
            notes=notes,
        )
    except Exception as e:  # noqa: BLE001
        log.exception("add_holding failed")
        await update.message.reply_text(
            f"⚠️ Could not add holding: <code>{html_escape(str(e))}</code>",
            parse_mode="HTML",
        )
        return

    final_qty = float(row.get("quantity", quantity))
    final_avg = float(row.get("avg_buy_price", buy_price))
    cls = row.get("asset_class", asset_class_or_err)
    notes_str = row.get("notes") or ""

    msg = (
        f"✅ <b>Position saved</b>\n\n"
        f"<b>{asset_label}</b> [{cls}]\n"
        f"Quantity: <b>{final_qty:g}</b>\n"
        f"Avg buy: <b>{_fmt_money(final_avg)}</b>\n"
        f"Cost basis: <b>{_fmt_money(final_qty * final_avg, decimals=0)}</b>\n"
    )
    if notes_str:
        msg += f"Notes: <i>{html_escape(notes_str)}</i>\n"
    msg += "\n<i>Use /portfolio to see live P&amp;L.</i>"

    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_remove_holding(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove a portfolio position by asset label."""
    args = ctx.args or []
    if len(args) != 1:
        # Show list of current holdings if no arg
        rows = await list_holdings()
        if not rows:
            await update.message.reply_text(
                "📭 No holdings to remove. Add one with /add_holding first.",
                parse_mode="HTML",
            )
            return
        labels = ", ".join(f"<code>{r['asset_label']}</code>" for r in rows)
        await update.message.reply_text(
            "💼 <b>/remove_holding</b> — drop a position\n\n"
            "<b>Usage:</b> <code>/remove_holding ASSET</code>\n\n"
            f"<b>Your positions:</b> {labels}",
            parse_mode="HTML",
        )
        return

    asset_label = args[0].upper().strip()
    removed = await remove_holding(asset_label)
    if removed:
        await update.message.reply_text(
            f"✅ Removed <b>{asset_label}</b> from your portfolio.",
            parse_mode="HTML",
        )
    else:
        await update.message.reply_text(
            f"⚠️ No position with label <code>{html_escape(asset_label)}</code>. "
            f"Use /portfolio to see your current positions.",
            parse_mode="HTML",
        )


# ============================================================================
# Callback dispatcher — single entry point for ALL inline button presses
# ============================================================================


async def callback_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()  # acknowledge — removes the spinner
    data = query.data or ""
    log.info("callback: %s", data)

    # ----- Idea card buttons -----
    if data.startswith("like_"):
        await _handle_idea_feedback(ctx, query, data[len("like_"):], "like", reason=None)
    elif data.startswith("dislike_"):
        idea_id = data[len("dislike_"):]
        # Edit message to show 6 reason buttons — that's how the system learns
        await query.edit_message_reply_markup(reply_markup=dislike_reason_buttons(idea_id))
    elif data.startswith("save_"):
        await _handle_idea_feedback(ctx, query, data[len("save_"):], "save", reason=None)
    elif data.startswith("build_"):
        # High-intent signal — heaviest weight in learning loop
        await _handle_idea_feedback(ctx, query, data[len("build_"):], "build", reason=None)
    elif data.startswith("deep_"):
        await _handle_idea_deep_dive(ctx, query, data[len("deep_"):])
    elif data.startswith("reason_"):
        # reason_<reason>_<idea_id>  →  3-part split
        parts = data.split("_", 2)
        if len(parts) != 3:
            log.warning("malformed reason callback: %s", data)
            return
        _, reason, idea_id = parts
        await _handle_idea_feedback(ctx, query, idea_id, "dislike", reason=reason)

    # ----- Investment card buttons -----
    elif data.startswith("inv_like_"):
        await _handle_signal_feedback(ctx, query, data[len("inv_like_"):], "like")
    elif data.startswith("inv_skip_"):
        await _handle_signal_feedback(ctx, query, data[len("inv_skip_"):], "dislike")
    elif data.startswith("inv_watch_"):
        sig_id = data[len("inv_watch_"):]
        # Watch = feedback save + add to active watchlist (alerts fire on ±5%)
        await _handle_signal_feedback(ctx, query, sig_id, "save")
        try:
            sig = _signal_store(ctx).get(sig_id)
            if sig and sig.price and sig.price > 0:
                from ..watchlist import add_to_watchlist  # noqa: PLC0415
                await add_to_watchlist(
                    asset_label=sig.asset,
                    baseline_price=float(sig.price),
                    source_sig_id=sig_id,
                )
        except Exception as e:  # noqa: BLE001
            log.warning("inv_watch_: failed to add to watchlist: %s", e)
    elif data.startswith("inv_deep_"):
        await _handle_signal_deep_dive(ctx, query, data[len("inv_deep_"):])

    # ----- Portfolio management buttons (per-holding) -----
    elif data.startswith("pf_remove_"):
        await _handle_pf_remove(ctx, query, data[len("pf_remove_"):])
    # pf_addusd_ / pf_subusd_ are caught by their ConversationHandler entry

    # ----- Sources card buttons (Step 8 — real implementations) -----
    # Note: "add_source" is captured by ConversationHandler entry_point, NOT here
    elif data == "source_stats":
        await _handle_source_stats(query)
    elif data == "remove_source":
        await _handle_remove_source_list(query)
    elif data.startswith("rmsrc_"):
        await _handle_remove_source_action(query, data[len("rmsrc_"):])

    else:
        log.warning("unknown callback: %s", data)
        await query.edit_message_text(f"⚠️ Unknown callback: <code>{data}</code>", parse_mode="HTML")


# ============================================================================
# Callback helpers
# ============================================================================


async def _handle_idea_feedback(
    ctx: ContextTypes.DEFAULT_TYPE,
    query: Any,
    idea_id: str,
    feedback_type: str,
    reason: str | None,
) -> None:
    idea = _idea_store(ctx).get(idea_id)
    if not idea:
        await query.edit_message_text(
            "⚠️ Idea not found (bot may have restarted since the digest was sent).",
        )
        return
    await _save_feedback(
        item_type="idea",
        item_id=idea_id,
        digest_id=_current_digest_id(ctx),
        feedback_type=feedback_type,
        reason=reason,
        item_snapshot=idea.model_dump(),
    )
    label = {
        "like":    "🔥 Loved",
        "save":    "📌 Saved",
        "dislike": "❌ Rejected",
        "build":   "🚀 Going to build",
    }.get(feedback_type, feedback_type)
    suffix = f" ({reason})" if reason else ""

    # For LIKE / SAVE / BUILD — only toast, do NOT replace card text.
    # User wants to keep the card visible to scroll back / press Deep dive.
    # For DISLIKE with reason — replace, since the reason-keyboard already
    # replaced the original markup and user finished the flow.
    if feedback_type in ("like", "save", "build"):
        try:
            await query.answer(
                text=f"{label}: записано",
                show_alert=False,
            )
        except Exception:  # noqa: BLE001
            pass
        return

    # Dislike / dislike-with-reason → replace card to confirm completion
    await query.edit_message_text(
        f"{label}{suffix}: <b>{idea.title}</b>\n\n"
        f"<i>Feedback recorded — ORACLE learns from this.</i>",
        parse_mode="HTML",
    )


async def _handle_signal_feedback(
    ctx: ContextTypes.DEFAULT_TYPE,
    query: Any,
    sig_id: str,
    feedback_type: str,
) -> None:
    sig = _signal_store(ctx).get(sig_id)
    if not sig:
        await query.edit_message_text(
            "⚠️ Signal not found (bot may have restarted since the digest was sent).",
        )
        return
    await _save_feedback(
        item_type="investment_signal",
        item_id=sig_id,
        digest_id=_current_digest_id(ctx),
        feedback_type=feedback_type,
        reason=None,
        item_snapshot=sig.model_dump(),
    )
    label = {
        "like": "✅ Marked useful",
        "save": "📌 Watching",
        "dislike": "❌ Skipped",
    }[feedback_type]
    from ..portfolio import display_name  # noqa: PLC0415

    # Like / Save / dislike on investment signals — toast only, keep card visible.
    # User wants to read the card again, click Deep dive, Add to portfolio etc.
    try:
        await query.answer(
            text=f"{label}: {display_name(sig.asset)}",
            show_alert=False,
        )
    except Exception:  # noqa: BLE001
        pass


async def _handle_idea_deep_dive(
    ctx: ContextTypes.DEFAULT_TYPE,
    query: Any,
    idea_id: str,
) -> None:
    """Real deep-dive on an idea — runs a focused LLM call (gpt-5.4-mini)
    that produces pricing analysis, CAC estimate, top 3 actionable next
    steps, and risk-callouts. Cheap (~$0.01) but high-value.

    Fallback (no LLM creds) — static placeholder.
    """
    idea = _idea_store(ctx).get(idea_id)
    if not idea:
        await query.edit_message_text("⚠️ Idea not found (bot may have restarted).")
        return

    # Show working state — deep-dive takes 5-15 sec
    try:
        progress = await query.message.reply_text(
            f"🔍 <i>Готовлю deep-dive по «{html_escape(idea.title)}»...\n"
            f"Анализ цен / CAC / next steps. ~10 сек.</i>",
            parse_mode="HTML",
        )
    except Exception:  # noqa: BLE001
        progress = None

    try:
        from ..agents.deep_dive import run_idea_deep_dive  # noqa: PLC0415
        dive = await run_idea_deep_dive(idea.model_dump())
    except Exception as e:  # noqa: BLE001
        log.warning("deep_dive: failed — falling back to static: %s", e)
        dive = None

    if dive:
        text = (
            f"🔍 <b>Deep dive: {html_escape(idea.title)}</b>\n\n"
            f"💵 <b>Pricing анализ:</b>\n{html_escape(dive.get('pricing_analysis', '—'))}\n\n"
            f"🎯 <b>CAC оценка:</b>\n{html_escape(dive.get('cac_estimate', '—'))}\n\n"
            f"📊 <b>Конкуренты (углублённо):</b>\n"
            f"{html_escape(dive.get('competitor_landscape', '—'))}\n\n"
            f"🚀 <b>Top 3 next steps:</b>\n"
        )
        next_steps = dive.get("next_steps") or []
        for i, step in enumerate(next_steps[:3], start=1):
            text += f"{i}. {html_escape(step)}\n"
        risks = dive.get("risk_callouts") or []
        if risks:
            text += f"\n⚠️ <b>Риски/блокеры:</b>\n"
            for r in risks[:3]:
                text += f"• {html_escape(r)}\n"
        if progress:
            try:
                await progress.edit_text(text, parse_mode="HTML")
                return
            except Exception:
                pass
        await query.message.reply_text(text, parse_mode="HTML")
        return

    # Fallback static
    competitors = "\n".join(f"• {c}" for c in idea.competitors) if idea.competitors else "<i>(none found)</i>"
    text = (
        f"🔍 <b>Deep dive: {idea.title}</b>\n\n"
        f"<b>Competitors:</b>\n{competitors}\n\n"
        f"<b>Customer acquisition path (placeholder):</b>\n"
        f"1. Post on relevant subreddits with case study\n"
        f"2. Launch on ProductHunt with demo video\n"
        f"3. Outreach to first 20 leads on LinkedIn\n\n"
        f"<b>Architecture sketch:</b>\n<code>{' → '.join(idea.mvp_stack)}</code>\n\n"
        f"🚧 <i>Full deep-dive (web search + market sizing) ships in Step 12 (validator).</i>"
    )
    await query.message.reply_text(text, parse_mode="HTML")


async def _handle_signal_deep_dive(
    ctx: ContextTypes.DEFAULT_TYPE,
    query: Any,
    sig_id: str,
) -> None:
    """Real investment deep-dive — LLM call producing technical levels +
    upcoming catalysts + historical analogues + sizing + downside risk.

    Anchored to Maksim's actual portfolio allocation (computed fresh
    from DB). Cost ~$0.01 per click.
    """
    sig = _signal_store(ctx).get(sig_id)
    if not sig:
        await query.edit_message_text("⚠️ Signal not found (bot may have restarted).")
        return

    from ..portfolio import display_name  # noqa: PLC0415

    progress = None
    try:
        progress = await query.message.reply_text(
            f"📊 <i>Готовлю deep-dive по {display_name(sig.asset)}...\n"
            f"Technical levels + catalysts + sizing для твоего портфеля. ~10 сек.</i>",
            parse_mode="HTML",
        )
    except Exception:  # noqa: BLE001
        pass

    # Fetch fresh portfolio snapshot (always current — reads DB)
    portfolio_summary = ""
    try:
        from ..agents.market import collect_market_data  # noqa: PLC0415
        from ..portfolio import (  # noqa: PLC0415
            format_portfolio_for_llm,
            get_portfolio_with_pnl,
        )
        market_data, _ = await collect_market_data()
        portfolio = await get_portfolio_with_pnl(market_data)
        portfolio_summary = format_portfolio_for_llm(portfolio)
    except Exception as e:  # noqa: BLE001
        log.warning("invest_deep_dive: portfolio fetch failed: %s", e)
        portfolio_summary = "(portfolio context unavailable)"

    # Call the deep-dive agent
    dive = None
    try:
        from ..agents.invest_deep_dive import run_invest_deep_dive  # noqa: PLC0415
        dive = await run_invest_deep_dive(sig.model_dump(), portfolio_summary)
    except Exception as e:  # noqa: BLE001
        log.warning("invest_deep_dive: failed: %s", e)

    if dive:
        catalysts = "\n".join(f"• {c}" for c in (dive.get("upcoming_catalysts") or []))
        text = (
            f"📊 <b>Deep-dive · {display_name(sig.asset)}</b>\n"
            f"💵 ${sig.price:,.2f} ({sig.change_24h:+.1f}% 24h)\n\n"
            f"📐 <b>Технические уровни:</b>\n{html_escape(dive.get('technical_levels', '—'))}\n\n"
            f"📅 <b>Ближайшие катализаторы:</b>\n{html_escape(catalysts) if catalysts else '<i>—</i>'}\n\n"
            f"📚 <b>Историческая аналогия:</b>\n{html_escape(dive.get('historical_analogues', '—'))}\n\n"
            f"⚖️ <b>Сайзинг под твой портфель:</b>\n{html_escape(dive.get('sizing_recommendation', '—'))}\n\n"
            f"⚠️ <b>Downside-риск:</b> {html_escape(dive.get('downside_risk', '—'))}\n\n"
            f"<i>Educational only. NOT financial advice.</i>"
        )
        try:
            if progress:
                await progress.edit_text(text, parse_mode="HTML")
                return
        except Exception:
            pass
        await query.message.reply_text(text, parse_mode="HTML")
        return

    # Fallback: re-show signal card content if LLM failed
    news = "\n".join(f"• {n}" for n in (sig.news_highlights or [])[:5]) or "<i>(нет новостей)</i>"
    text = (
        f"📊 <b>Полный анализ: {display_name(sig.asset)}</b>\n\n"
        f"💵 <b>Цена:</b> ${sig.price:,.2f} ({sig.change_24h:+.1f}% 24h)\n\n"
        f"📰 <b>Новости:</b>\n{news}\n\n"
        f"📊 <b>Тренд:</b> {sig.trend or '—'}\n\n"
        f"🐂 <b>Бык:</b> {sig.critic_bull or '—'}\n\n"
        f"🐻 <b>Медведь:</b> {sig.critic_bear or '—'}\n\n"
        f"🔮 <b>Прогноз:</b> {sig.prediction or sig.future_outlook or '—'}\n\n"
        f"<i>(Deep-dive LLM упал — показываю основную карточку)</i>"
    )
    if progress:
        try:
            await progress.edit_text(text, parse_mode="HTML")
            return
        except Exception:
            pass
    await query.message.reply_text(text, parse_mode="HTML")


# ============================================================================
# Feedback persistence — writes to oracle_data.db.feedback (Step 2 schema)
# ============================================================================


async def _save_feedback(
    *,
    item_type: str,
    item_id: str,
    digest_id: str,
    feedback_type: str,
    reason: str | None,
    item_snapshot: dict,
) -> None:
    """Persist a feedback row + maybe trigger learning calibration (Step 14)."""
    async with get_db() as conn:
        await conn.execute(
            """INSERT INTO feedback (item_type, item_id, digest_id, feedback_type,
                                       reason, item_snapshot, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                item_type,
                item_id,
                digest_id,
                feedback_type,
                reason,
                json.dumps(item_snapshot, default=str),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await conn.commit()
    log.info(
        "feedback saved: type=%s id=%s feedback=%s reason=%s",
        item_type, item_id, feedback_type, reason,
    )

    # Step 14: fire-and-forget calibration trigger. Doesn't block the user's
    # button-click response. The bot keeps responding instantly; calibration
    # runs in the background. maybe_calibrate() is a no-op if threshold not
    # reached, and never raises (catches all errors internally).
    asyncio.create_task(_background_calibration())


async def _background_calibration() -> None:
    """Wraps maybe_calibrate so the create_task fire-and-forget is clean."""
    try:
        result = await maybe_calibrate()
        if result:
            log.info("learning: background calibration completed (%d analyzed)",
                     result.total_analyzed)
    except Exception as e:  # noqa: BLE001 — never crash the bot
        log.error("learning: background calibration error: %s", e)
