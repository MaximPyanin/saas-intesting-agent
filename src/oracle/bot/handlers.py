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

<i>Tap 🔥 / ❌ / 📌 buttons on each idea — every 30 feedbacks I'll
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
    investments_raw = final_digest.get("investments") or []
    errors = result.get("errors") or []

    # Coerce dicts from graph state into Pydantic models for rendering
    ideas: list[BusinessIdea] = []
    for d in ideas_raw:
        try:
            ideas.append(BusinessIdea.model_validate(d))
        except Exception as e:  # noqa: BLE001
            log.warning("cmd_digest: failed to parse idea %r: %s", d.get("title"), e)

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


async def cmd_morning(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        render_morning_brief(mock_morning_data()),
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

    parts.append("<b>Positions:</b>")
    for h in holdings:
        label = h["asset_label"]
        qty = h["quantity"]
        avg = h["avg_buy_price"]
        cur = h.get("current_price")
        mv = h.get("market_value_usd")
        pnl_pct = h.get("pnl_pct")
        status = h.get("price_status", "?")
        notes = h.get("notes") or ""

        if status == "cash":
            line = f"• <b>{label}</b>: {_fmt_money(qty, decimals=2)}"
        elif status == "live":
            line = (
                f"• <b>{label}</b>: {qty:g} @ {_fmt_money(avg)} avg → "
                f"now {_fmt_money(cur)} · MV {_fmt_money(mv, decimals=0)} · {_fmt_pct(pnl_pct)}"
            )
        else:
            line = (
                f"• <b>{label}</b>: {qty:g} @ {_fmt_money(avg)} avg "
                f"<i>(no live price)</i>"
            )
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
    """Show current holdings with live P&L."""
    await update.message.reply_text(
        "📡 Fetching live prices for your portfolio...",
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

    await update.message.reply_text(
        _render_portfolio_html(portfolio),
        parse_mode="HTML",
        disable_web_page_preview=True,
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

    ok, asset_class_or_err = validate_asset_label(asset_label)
    if not ok:
        await update.message.reply_text(
            f"⚠️ Unknown asset <code>{html_escape(asset_label)}</code>.\n\n"
            f"{ADD_HOLDING_USAGE}",
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
        await _handle_signal_feedback(ctx, query, data[len("inv_watch_"):], "save")
    elif data.startswith("inv_deep_"):
        await _handle_signal_deep_dive(ctx, query, data[len("inv_deep_"):])

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
        "like": "🔥 Loved",
        "save": "📌 Saved",
        "dislike": "❌ Rejected",
    }[feedback_type]
    suffix = f" <i>({reason})</i>" if reason else ""
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
    await query.edit_message_text(
        f"{label}: <b>{sig.asset}</b>\n\n<i>Feedback recorded.</i>",
        parse_mode="HTML",
    )


async def _handle_idea_deep_dive(
    ctx: ContextTypes.DEFAULT_TYPE,
    query: Any,
    idea_id: str,
) -> None:
    idea = _idea_store(ctx).get(idea_id)
    if not idea:
        await query.edit_message_text("⚠️ Idea not found (bot may have restarted).")
        return
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
    sig = _signal_store(ctx).get(sig_id)
    if not sig:
        await query.edit_message_text("⚠️ Signal not found (bot may have restarted).")
        return
    key_events = "\n".join(f"• {e}" for e in sig.key_events) if sig.key_events else "<i>(none)</i>"
    text = (
        f"📊 <b>Full analysis: {sig.asset}</b>\n\n"
        f"<b>Bull thesis ({sig.bull_prob}%):</b> {sig.bull_scenario}\n"
        f"<i>Trigger:</i> {sig.bull_trigger}\n\n"
        f"<b>Bear thesis ({sig.bear_prob}%):</b> {sig.bear_scenario}\n"
        f"<i>Trigger:</i> {sig.bear_trigger}\n\n"
        f"<b>Key dates:</b>\n{key_events}\n\n"
        f"<b>Geo factor:</b> {sig.geopolitical_note}\n\n"
        f"⚠️ <i>Educational analysis only. NOT financial advice.</i>\n"
        f"🚧 <i>Real-time price re-verification ships in Step 12.</i>"
    )
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
