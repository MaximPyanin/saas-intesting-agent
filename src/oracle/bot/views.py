"""ORACLE Telegram view layer — message templates and inline keyboards.

All templates use HTML parse_mode (cleaner than MarkdownV2 for structured
content). User-provided strings are HTML-escaped before formatting so LLM
or scraped content can never break the markup.
"""

from __future__ import annotations

import html
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from ..models import BusinessIdea, InvestmentSignal


# ============================================================================
# Lifecycle stage explanations (literal from user spec)
# ============================================================================

LIFECYCLE_EXPLANATIONS: dict[str, str] = {
    "EMERGING": "just starting, few know — risky but high upside",
    "GROWING": "gaining momentum — BEST TIME to build",
    "PEAK": "everyone knows — late but possible if niche",
    "DECLINING": "fading — avoid building here",
}


# ============================================================================
# Freshness indicators (from user spec)
# ============================================================================

def freshness_indicator(hours_ago: int) -> str:
    """🔴 BREAKING (<2h) | 🟡 TODAY (<24h) | 🟢 THIS WEEK (<7d) | ⚪ OLDER"""
    if hours_ago < 2:
        return "🔴 BREAKING (&lt;2h)"
    if hours_ago < 24:
        return "🟡 TODAY (&lt;24h)"
    if hours_ago < 168:
        return "🟢 THIS WEEK (&lt;7d)"
    return "⚪ OLDER"


# ============================================================================
# Templates (HTML parse_mode)
# ============================================================================

MORNING_TEMPLATE = """\
☀️ <b>ORACLE Morning Brief</b> · {date}
━━━━━━━━━━━━━━━━━━

{breaking_emoji} <b>OVERNIGHT:</b>
{top_3_events}

📊 <b>Markets:</b> Gold ${gold} {gold_pct} · BTC ${btc} {btc_pct}
S&amp;P {sp500} {sp500_pct} · Oil ${oil} {oil_pct} · VIX {vix}

💡 <b>Today's focus:</b> {key_signal_one_liner}

<i>Full digest at 19:00 Warsaw</i> ⏰
"""

IDEA_CARD_TEMPLATE = """\
💡 <b>IDEA #{n}</b> · {lifecycle_stage} · Passed {reflexion_rounds_passed}/3 critique rounds
━━━━━━━━━━━━━━━━━━━━━━

🏷️ <b>{title}</b>
{one_liner}

🎯 <b>Problem:</b> {problem}
⚡ <b>Solution:</b> {solution}
👤 <b>Customer:</b> {target_customer}
💰 <b>Revenue:</b> {revenue_model}

🕐 <b>Why NOW:</b> {why_now}
⏳ <b>Window:</b> {window_months} months before market saturates
📅 <b>Lifecycle:</b> {lifecycle_stage} — {lifecycle_explanation}

🛠️ <b>MVP:</b> {mvp_weeks} weeks · <b>Stack:</b> {mvp_stack_short}
💵 <b>Estimated cost:</b> {estimated_cost_usd}
🏆 <b>Advantage:</b> {unfair_advantage}

🔎 <b>Similar projects:</b>
{similar_projects_block}

🚀 <b>Launch steps:</b>
{launch_steps_block}

📣 <b>Where to advertise:</b>
{marketing_channels_block}

🔍 <b>Competitors:</b> {competitors_or_none}
📊 <b>Confidence:</b> {confidence}/100

📡 <b>Signals:</b> {signal_sources_short}
"""

INVESTMENT_CARD_TEMPLATE = """\
📈 <b>SIGNAL #{n}</b> · {asset} · {signal_type}
━━━━━━━━━━━━━━━━━━

🚨 <b>{do_now}</b>

💡 <b>Почему сейчас:</b> {why_now_short}

📌 <b>Как делать:</b> {how_to_execute}

{portfolio_line}📊 <b>Цена:</b> ${price} ({change_24h} сегодня, {change_7d} за неделю)
⚡ <b>Сила сигнала:</b> {strength}/10 · <b>Горизонт:</b> {timeframe}

🔁 <b>План после:</b> {post_action}

🛑 <b>Стоп-условие</b> (если пойдёт не туда): {bear_trigger}
📅 <b>Ключевые даты:</b> {key_events_str}
🌍 <b>Гео-фактор:</b> {geopolitical_note}

<i>Расклад Bull/Bear (краткая справка): 🟢 {bull_prob}% vs 🔴 {bear_prob}%</i>

⚠️ <i>Educational analysis only. NOT financial advice.</i>
"""

SOURCES_CARD_TEMPLATE = """\
📡 <b>Your Sources</b> ({total} active)

💡 <b>BUSINESS IDEAS</b> ({ideas_count}):
{ideas_sources_list}

📈 <b>INVESTMENTS</b> ({inv_count}):
{inv_sources_list}

🌍 <b>GEOPOLITICS</b> ({geo_count}):
{geo_sources_list}
"""


# ============================================================================
# Renderers
# ============================================================================


def _esc(value: Any) -> str:
    """HTML-escape any value, coercing to str first."""
    return html.escape(str(value))


def render_morning_brief(data: dict) -> str:
    return MORNING_TEMPLATE.format(
        date=_esc(data.get("date", "")),
        breaking_emoji=_esc(data.get("breaking_emoji", "")),
        top_3_events=_esc(data.get("top_3_events", "")),
        gold=_esc(data.get("gold", "")),
        gold_pct=_esc(data.get("gold_pct", "")),
        btc=_esc(data.get("btc", "")),
        btc_pct=_esc(data.get("btc_pct", "")),
        sp500=_esc(data.get("sp500", "")),
        sp500_pct=_esc(data.get("sp500_pct", "")),
        oil=_esc(data.get("oil", "")),
        oil_pct=_esc(data.get("oil_pct", "")),
        vix=_esc(data.get("vix", "")),
        key_signal_one_liner=_esc(data.get("key_signal_one_liner", "")),
    )


def _bullet_block(items: list[str], empty: str = "—") -> str:
    """Render a list as '• item' lines, HTML-escaped. Returns `empty` if no items."""
    if not items:
        return f"<i>{empty}</i>"
    return "\n".join(f"• {_esc(x)}" for x in items)


def _numbered_block(items: list[str], empty: str = "—") -> str:
    """Render a list as '1. item' lines, HTML-escaped."""
    if not items:
        return f"<i>{empty}</i>"
    return "\n".join(f"{i}. {_esc(x)}" for i, x in enumerate(items, start=1))


def render_idea_card(idea: BusinessIdea, n: int) -> str:
    return IDEA_CARD_TEMPLATE.format(
        n=n,
        title=_esc(idea.title),
        one_liner=_esc(idea.one_liner),
        problem=_esc(idea.problem),
        solution=_esc(idea.solution),
        target_customer=_esc(idea.target_customer),
        revenue_model=_esc(idea.revenue_model),
        why_now=_esc(idea.why_now),
        window_months=idea.window_months,
        lifecycle_stage=_esc(idea.lifecycle_stage),
        lifecycle_explanation=_esc(LIFECYCLE_EXPLANATIONS.get(idea.lifecycle_stage, "")),
        mvp_weeks=idea.mvp_weeks,
        mvp_stack_short=_esc(", ".join(idea.mvp_stack)),
        estimated_cost_usd=_esc(idea.estimated_cost_usd or "—"),
        unfair_advantage=_esc(idea.unfair_advantage or "—"),
        similar_projects_block=_bullet_block(idea.similar_projects, empty="(none listed)"),
        launch_steps_block=_numbered_block(idea.launch_steps, empty="(none listed)"),
        marketing_channels_block=_bullet_block(idea.marketing_channels, empty="(none listed)"),
        competitors_or_none=_esc(", ".join(idea.competitors) if idea.competitors else "(none found)"),
        confidence=idea.confidence,
        signal_sources_short=_esc(", ".join(idea.signal_sources) if idea.signal_sources else "—"),
        reflexion_rounds_passed=idea.reflexion_rounds_passed,
    )


ACTION_EMOJI: dict[str, str] = {
    "BUY":   "🟢",
    "ADD":   "🟢",
    "HOLD":  "🟡",
    "TRIM":  "🟠",
    "SELL":  "🔴",
    "WAIT":  "⚪",
    "AVOID": "⛔",
}


def render_investment_card(sig: InvestmentSignal, n: int) -> str:
    action = (sig.action or "WAIT").upper()

    # do_now: prefer the new field, fallback to building from action + reason
    do_now = sig.do_now.strip()
    if not do_now:
        emoji = ACTION_EMOJI.get(action, "⚪")
        do_now = f"{emoji} {action}"
        if sig.action_reason:
            do_now += f" — {sig.action_reason}"

    how_to = sig.how_to_execute.strip() or "—"
    why_now = sig.why_now_short.strip() or "—"
    post = sig.post_action.strip() or "—"

    # Portfolio line — ONLY shown if the asset is held.
    if sig.is_portfolio_holding:
        portfolio_line = (
            "💼 <b>Твоя позиция:</b> держишь этот актив — план выше "
            "учитывает твой P&amp;L\n"
        )
    else:
        portfolio_line = ""

    return INVESTMENT_CARD_TEMPLATE.format(
        n=n,
        asset=_esc(sig.asset),
        signal_type=_esc(sig.signal_type),
        do_now=_esc(do_now),
        why_now_short=_esc(why_now),
        how_to_execute=_esc(how_to),
        portfolio_line=portfolio_line,
        post_action=_esc(post),
        price=_esc(f"{sig.price:,.2f}"),
        change_24h=_esc(f"{sig.change_24h:+.1f}%"),
        change_7d=_esc(f"{sig.change_7d:+.1f}%"),
        strength=sig.strength,
        timeframe=_esc(sig.timeframe),
        freshness=freshness_indicator(sig.signal_age_hours),
        bull_prob=sig.bull_prob,
        bear_prob=sig.bear_prob,
        bear_trigger=_esc(sig.bear_trigger),
        key_events_str=_esc(", ".join(sig.key_events) if sig.key_events else "—"),
        geopolitical_note=_esc(sig.geopolitical_note or "—"),
    )


def render_sources_card(data: dict) -> str:
    return SOURCES_CARD_TEMPLATE.format(
        total=data.get("total", 0),
        ideas_count=data.get("ideas_count", 0),
        ideas_sources_list=_esc(data.get("ideas_sources_list", "")),
        inv_count=data.get("inv_count", 0),
        inv_sources_list=_esc(data.get("inv_sources_list", "")),
        geo_count=data.get("geo_count", 0),
        geo_sources_list=_esc(data.get("geo_sources_list", "")),
    )


# ============================================================================
# Inline keyboards (literal callback_data scheme from user spec)
# ============================================================================


def idea_buttons(idea_id: str) -> InlineKeyboardMarkup:
    """4 buttons per idea card: Love / Not for me / Save / Go deeper."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔥 Love it", callback_data=f"like_{idea_id}"),
            InlineKeyboardButton("❌ Not for me", callback_data=f"dislike_{idea_id}"),
        ],
        [
            InlineKeyboardButton("📌 Save for later", callback_data=f"save_{idea_id}"),
            InlineKeyboardButton("🔍 Go deeper", callback_data=f"deep_{idea_id}"),
        ],
    ])


def dislike_reason_buttons(idea_id: str) -> InlineKeyboardMarkup:
    """6 reason buttons shown after a dislike — that's how the system learns."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🏆 Has competitors",   callback_data=f"reason_competitors_{idea_id}")],
        [InlineKeyboardButton("👤 Wrong customer",    callback_data=f"reason_customer_{idea_id}")],
        [InlineKeyboardButton("💰 Bad revenue model", callback_data=f"reason_revenue_{idea_id}")],
        [InlineKeyboardButton("🛠️ Too hard to build", callback_data=f"reason_complex_{idea_id}")],
        [InlineKeyboardButton("📉 Market too small",  callback_data=f"reason_market_{idea_id}")],
        [InlineKeyboardButton("⏰ Bad timing",         callback_data=f"reason_timing_{idea_id}")],
    ])


def investment_buttons(sig_id: str) -> InlineKeyboardMarkup:
    """4 buttons per investment card: Useful / Skip / Watch / Full analysis."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Useful",        callback_data=f"inv_like_{sig_id}"),
            InlineKeyboardButton("❌ Not relevant",  callback_data=f"inv_skip_{sig_id}"),
        ],
        [
            InlineKeyboardButton("📌 Watch this",    callback_data=f"inv_watch_{sig_id}"),
            InlineKeyboardButton("📊 Full analysis", callback_data=f"inv_deep_{sig_id}"),
        ],
    ])


def sources_buttons() -> InlineKeyboardMarkup:
    """3 buttons on the sources management card."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ Add source",     callback_data="add_source")],
        [InlineKeyboardButton("📊 Source stats",   callback_data="source_stats")],
        [InlineKeyboardButton("🗑️ Remove source",  callback_data="remove_source")],
    ])


# ============================================================================
# /add_source conversation flow keyboards (Step 8)
# ============================================================================

# Type picker — what kind of source to add. Spec types from Step 7.
SOURCE_TYPE_LABELS: dict[str, str] = {
    "telegram_channel": "💬 Telegram channel",
    "youtube_channel":  "📺 YouTube channel",
    "website_blog":     "🌐 Website / Blog",
    "rss_custom":       "📰 RSS feed",
    "reddit_custom":    "🔥 Subreddit",
}

# Category picker — primary axis is business ideas vs investments per user spec.
SOURCE_CATEGORY_LABELS: dict[str, str] = {
    "business_ideas": "💡 Business ideas",
    "investments":    "📈 Investments",
    "ai_tech":        "🤖 AI / Tech",
    "geopolitics":    "🌍 Geopolitics",
    "other":          "📦 Other",
}


def add_source_type_kb() -> InlineKeyboardMarkup:
    """Step 1 of /add_source: pick the source type."""
    rows = [[InlineKeyboardButton(label, callback_data=f"addtype_{key}")]
            for key, label in SOURCE_TYPE_LABELS.items()]
    rows.append([InlineKeyboardButton("❌ Cancel", callback_data="addsrc_cancel")])
    return InlineKeyboardMarkup(rows)


def add_source_category_kb() -> InlineKeyboardMarkup:
    """Step 2 of /add_source: pick the category.

    Business ideas first per user priority (70% ideas / 30% investments).
    """
    rows = [[InlineKeyboardButton(label, callback_data=f"addcat_{key}")]
            for key, label in SOURCE_CATEGORY_LABELS.items()]
    rows.append([InlineKeyboardButton("❌ Cancel", callback_data="addsrc_cancel")])
    return InlineKeyboardMarkup(rows)


def add_source_confirm_kb() -> InlineKeyboardMarkup:
    """Final step: confirm or cancel before INSERT."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Add it",  callback_data="addconf_yes"),
            InlineKeyboardButton("❌ Cancel",  callback_data="addsrc_cancel"),
        ],
    ])


def remove_source_list_kb(sources: list[dict]) -> InlineKeyboardMarkup:
    """Build a button list for removing sources, one button per source.

    callback_data format: `rmsrc_<id>` (or `rmsrc_cancel`).
    """
    rows: list[list[InlineKeyboardButton]] = []
    for s in sources[:20]:  # cap at 20 to keep keyboard sane
        label = f"🗑 #{s['id']} [{s['source_type']}] {s['display_name'][:30]}"
        rows.append([InlineKeyboardButton(label, callback_data=f"rmsrc_{s['id']}")])
    rows.append([InlineKeyboardButton("❌ Cancel", callback_data="rmsrc_cancel")])
    return InlineKeyboardMarkup(rows)


# ============================================================================
# Real /sources card renderer (Step 8) — reads from user_sources table
# ============================================================================


def render_real_sources_card(sources: list[dict]) -> str:
    """Render a sources card from a list of user_sources rows.

    Groups by category. Empty categories are skipped.
    """
    if not sources:
        return (
            "📡 <b>Your Sources</b> (0 active)\n\n"
            "<i>No custom sources yet. Tap </i>➕ Add source<i> below to add your "
            "first one — Telegram channel, YouTube, website, RSS feed, or subreddit.</i>"
        )

    by_category: dict[str, list[dict]] = {}
    for s in sources:
        by_category.setdefault(s["category"], []).append(s)

    cat_headers = {
        "business_ideas": "💡 BUSINESS IDEAS",
        "investments":    "📈 INVESTMENTS",
        "ai_tech":        "🤖 AI / TECH",
        "geopolitics":    "🌍 GEOPOLITICS",
        "other":          "📦 OTHER",
    }

    parts: list[str] = [f"📡 <b>Your Sources</b> ({len(sources)} active)\n"]
    for cat in ("business_ideas", "investments", "ai_tech", "geopolitics", "other"):
        srcs = by_category.get(cat, [])
        if not srcs:
            continue
        parts.append(f"\n<b>{cat_headers[cat]}</b> ({len(srcs)}):")
        for s in srcs:
            t = s["source_type"]
            name = _esc(s["display_name"])
            parts.append(f"• [{t}] {name}")
    return "\n".join(parts)


def render_source_stats_card(stats: dict) -> str:
    """Render a source stats card from aggregated query results."""
    parts = ["📊 <b>Source Stats</b>\n"]

    parts.append(f"<b>Custom sources:</b> {stats['user_sources_active']} active "
                 f"({stats['user_sources_inactive']} inactive)")
    parts.append(f"<b>Total signals collected:</b> {stats['signals_total']}")
    parts.append(f"<b>Breaking signals (&lt;2h):</b> {stats['signals_breaking']}")
    parts.append("")

    if stats.get("by_source_type"):
        parts.append("<b>By source type:</b>")
        for st, n in stats["by_source_type"]:
            parts.append(f"• {st}: {n}")
        parts.append("")

    if stats.get("top_user_sources"):
        parts.append("<b>Your top custom sources (by signal count):</b>")
        for name, n in stats["top_user_sources"]:
            parts.append(f"• {_esc(name)}: {n}")

    return "\n".join(parts)
