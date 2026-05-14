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
🏷️ <b>{display_name}</b>
💵 ${price}  ·  📉 {change_24h} (24h)
{portfolio_line}━━━━━━━━━━━━━━━━━━

📰 <b>Новости:</b>
{news_block}

📊 <b>Тренд:</b> {trend}

🐂 <b>Бык:</b> {critic_bull}

🐻 <b>Медведь:</b> {critic_bear}

━━━━━━━━━━━━━━━━━━
🎯 <b>ВЕРДИКТ · {action}</b>
{do_now}
{verdict_exec_line}

🔮 <b>1-4 недели:</b> {prediction}

🗓️ <b>1-3 месяца:</b> {prediction_mid}

📅 <b>1-3 года:</b> {prediction_long}

⚠️ <i>Educational only. NOT financial advice.</i>
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
    """Legacy morning brief renderer — kept for mock/dry-run flows.

    The live 07:00 scheduled job uses render_urgent_section() and
    render_portfolio_morning_advice() instead (Maksim asked to replace
    the rates/market snapshot with СРОЧНО + per-holding advice).
    """
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


# ----------------------------------------------------------------------------
# New morning sections (Step 20.5 — replacing the legacy rates snapshot)
# ----------------------------------------------------------------------------


_ACTION_EMOJI: dict[str, str] = {
    "HOLD":      "🟢",
    "ADD":       "➕",
    "BUY":       "🟢",
    "TRIM":      "🟠",
    "SELL":      "🔴",
    "WATCH":     "🟡",
    "REBALANCE": "🔄",
    "AVOID":     "⛔",
    "WAIT":      "⏸",
}


def render_urgent_section(
    active_hits: list[Any],
    recent_fires: list[dict],
    *,
    date: str,
    breaking_news: list[dict] | None = None,
    portfolio_movers: list[dict] | None = None,
) -> str:
    """Render the '🚨 СРОЧНО' section of the morning brief.

    Args:
        active_hits: AlertHit objects currently triggered (HIGH/MEDIUM
            priority alerts active right now).
        recent_fires: {rule_id, fired_at} for alerts that fired in last 12h.
        date: ISO date for the header.
        breaking_news: top breaking signals from DB (is_breaking=1, last 12h).
            Each item: {title, source_id, url}.
        portfolio_movers: holdings with biggest 24h moves (|change_24h_pct| >= 2).
            Each item: {asset_label, change_24h_pct, price}.

    Returns HTML-formatted Telegram message.
    """
    lines: list[str] = [f"🚨 <b>СРОЧНО · {_esc(date)}</b>"]
    lines.append("─" * 25)

    has_content = bool(active_hits or recent_fires or breaking_news or portfolio_movers)
    if not has_content:
        lines.append("🟢 <i>Ничего срочного за ночь. Спи спокойно.</i>")
        return "\n".join(lines)

    # 1. Active alerts right now (highest urgency)
    if active_hits:
        lines.append("")
        lines.append("<b>⚡ Активно сейчас:</b>")
        for hit in active_hits[:5]:
            priority = getattr(hit, "priority", "MEDIUM")
            title = getattr(hit, "title", str(hit))
            asset = getattr(hit, "asset", "")
            details = getattr(hit, "details", "")
            emoji = "🚨" if priority == "HIGH" else "⚠️"
            asset_tag = f" <code>{_esc(asset)}</code>" if asset else ""
            lines.append(f"{emoji} <b>{_esc(title)}</b>{asset_tag}")
            first_line = details.split("\n", 1)[0] if details else ""
            if first_line:
                lines.append(f"   <i>{first_line}</i>")

    # 2. Biggest portfolio movers in last 24h (signal-rich for Maksim)
    if portfolio_movers:
        from ..portfolio import display_name  # noqa: PLC0415
        lines.append("")
        lines.append("<b>📊 Твои активы — самые большие движения за 24ч:</b>")
        for m in portfolio_movers[:5]:
            label = m.get("asset_label", "?")
            c24 = m.get("change_24h_pct", 0) or 0
            price = m.get("current_price")
            emoji = "🟢" if c24 >= 0 else "🔴"
            sign = "+" if c24 >= 0 else ""
            price_tag = f" @ ${price:,.2f}" if price else ""
            lines.append(
                f"{emoji} <b>{_esc(display_name(label))}</b>: "
                f"<b>{sign}{c24:.1f}%</b>{price_tag}"
            )

    # 3. Breaking news headlines from overnight
    if breaking_news:
        lines.append("")
        lines.append("<b>📰 Главное за ночь:</b>")
        for n in breaking_news[:4]:
            title = (n.get("title") or "(no title)")[:120]
            src = n.get("source_id") or ""
            src_tag = f" <i>({_esc(src)})</i>" if src else ""
            lines.append(f"• {_esc(title)}{src_tag}")

    # 4. Overnight alert fires (lowest priority — info only)
    if recent_fires:
        lines.append("")
        lines.append("<b>🔔 Сработало за ночь:</b>")
        for r in recent_fires[:5]:
            ts = r.get("fired_at")
            try:
                when = ts.strftime("%H:%M UTC") if ts else "?"
            except Exception:
                when = "?"
            lines.append(f"• <code>{when}</code> — {_esc(r.get('rule_id', '?'))}")

    return "\n".join(lines)


def render_portfolio_morning_advice(
    advice_list: list[Any],
    portfolio: dict,
    *,
    date: str,
) -> str:
    """Render the '💼 Совет по портфелю' section of the morning brief.

    Args:
        advice_list: list of PortfolioAdvice objects (asset/action/advice_short)
            from agents.portfolio_advisor.generate_morning_portfolio_advice().
        portfolio: dict from portfolio.get_portfolio_with_pnl() — used to attach
            P&L numbers to each line.
        date: ISO date for the header.

    Returns HTML-formatted Telegram message. One line per holding plus a
    totals footer.
    """
    from ..portfolio import display_name  # noqa: PLC0415

    holdings_by_label = {
        (h.get("asset_label") or "").upper(): h
        for h in (portfolio.get("holdings") or [])
    }

    lines: list[str] = [f"💼 <b>ПОРТФЕЛЬ — УТРО {_esc(date)}</b>"]
    lines.append("─" * 25)

    if not advice_list:
        lines.append("<i>(нет позиций в портфеле — добавь через ➕ кнопку или /add_holding)</i>")
        return "\n".join(lines)

    for adv in advice_list:
        asset = getattr(adv, "asset", "?")
        action = (getattr(adv, "action", "HOLD") or "HOLD").upper()
        text = getattr(adv, "advice_short", "")
        h = holdings_by_label.get(asset.upper(), {})

        # Prefer 24h live change over P&L drift — much more informative
        # for "what happened today" morning view.
        c24 = h.get("change_24h_pct")
        if c24 is not None:
            sign = "+" if c24 >= 0 else ""
            move_str = f"{sign}{c24:.1f}%"
        else:
            pnl_pct = h.get("pnl_pct")
            if pnl_pct is None:
                move_str = "—"
            else:
                sign = "+" if pnl_pct >= 0 else ""
                move_str = f"{sign}{pnl_pct:.1f}%"

        emoji = _ACTION_EMOJI.get(action, "•")
        # 2-line format keeps human-readable names readable without forcing
        # a fixed-width table. Action + 24h move + advice on the 2nd line.
        lines.append(
            f"{emoji} <b>{_esc(display_name(asset))}</b>\n"
            f"   <b>{_esc(action)}</b> · {_esc(move_str)} → {_esc(text)}"
        )

    totals = portfolio.get("totals") or {}
    mv = totals.get("market_value_usd") or 0.0
    pnl = totals.get("unrealized_pnl_usd") or 0.0
    pnl_pct = totals.get("pnl_pct") or 0.0
    sign = "+" if pnl >= 0 else ""
    lines.append("─" * 25)
    lines.append(
        f"<b>Общая стоимость:</b> ${mv:,.0f} "
        f"(<i>{sign}${pnl:,.0f} / {sign}{pnl_pct:.1f}%</i>)"
    )
    return "\n".join(lines)


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


def _truncate(text: str, limit: int) -> str:
    """Cap a string to `limit` chars, adding ellipsis if cut. Single-line."""
    if not text:
        return text
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


TELEGRAM_HARD_LIMIT = 3900  # Conservative cap; real limit is 4096


def _final_cap(text: str) -> str:
    """Final safety net — if the rendered card overflows Telegram's limit,
    chop the middle/tail and add a continuation note."""
    if len(text) <= TELEGRAM_HARD_LIMIT:
        return text
    head_keep = TELEGRAM_HARD_LIMIT - 100
    return text[:head_keep] + (
        "\n\n<i>… [card truncated to fit Telegram limit — full content "
        "stored in DB, ask /lastdigest to re-read]</i>"
    )


def render_idea_card(idea: BusinessIdea, n: int) -> str:
    """Render with per-section truncation to stay under Telegram's 4096
    character limit. Per-section caps tuned so a fully-populated card
    renders to ~3500 chars. Hard final cap via _final_cap as safety net."""
    result = IDEA_CARD_TEMPLATE.format(
        n=n,
        title=_esc(_truncate(idea.title, 100)),
        one_liner=_esc(_truncate(idea.one_liner, 250)),
        problem=_esc(_truncate(idea.problem, 350)),
        solution=_esc(_truncate(idea.solution, 350)),
        target_customer=_esc(_truncate(idea.target_customer, 280)),
        revenue_model=_esc(_truncate(idea.revenue_model, 220)),
        why_now=_esc(_truncate(idea.why_now, 280)),
        window_months=idea.window_months,
        lifecycle_stage=_esc(idea.lifecycle_stage),
        lifecycle_explanation=_esc(LIFECYCLE_EXPLANATIONS.get(idea.lifecycle_stage, "")),
        mvp_weeks=idea.mvp_weeks,
        mvp_stack_short=_esc(_truncate(", ".join(idea.mvp_stack), 150)),
        estimated_cost_usd=_esc(_truncate(idea.estimated_cost_usd or "—", 180)),
        unfair_advantage=_esc(_truncate(idea.unfair_advantage or "—", 300)),
        similar_projects_block=_bullet_block(
            [_truncate(s, 130) for s in (idea.similar_projects or [])[:3]],
            empty="(none listed)",
        ),
        launch_steps_block=_numbered_block(
            [_truncate(s, 160) for s in (idea.launch_steps or [])[:5]],
            empty="(none listed)",
        ),
        marketing_channels_block=_bullet_block(
            [_truncate(s, 100) for s in (idea.marketing_channels or [])[:4]],
            empty="(none listed)",
        ),
        competitors_or_none=_esc(_truncate(
            ", ".join(idea.competitors) if idea.competitors else "(none found)", 250)),
        confidence=idea.confidence,
        signal_sources_short=_esc(_truncate(
            ", ".join(idea.signal_sources) if idea.signal_sources else "—", 200)),
        reflexion_rounds_passed=idea.reflexion_rounds_passed,
    )
    return _final_cap(result)


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
    """Maksim-redesign card v2 (May 2026):
    display_name + price + 24h drawdown + portfolio line (if held)
    + news + trend + critic_bull + critic_bear + verdict + prediction.

    NEW vs previous: human-readable asset name, separate 24h drawdown,
    `trend` (current momentum) between news and critics, `prediction`
    (forecast) AFTER the verdict.
    """
    from ..portfolio import display_name  # noqa: PLC0415

    action = (sig.action or "WAIT").upper()

    do_now = (sig.do_now or "").strip()
    if not do_now:
        emoji = ACTION_EMOJI.get(action, "⚪")
        do_now = f"{emoji} {action}"

    # News block — up to 3 headlines as bullets
    news_items = [n for n in (sig.news_highlights or []) if n and n.strip()][:3]
    if news_items:
        news_block = "\n".join(f"• {_esc(item)}" for item in news_items)
    else:
        news_block = "<i>• нет свежих заголовков по сектору</i>"

    trend = (sig.trend or "").strip() or "<i>—</i>"
    bull = (sig.critic_bull or "").strip() or "<i>—</i>"
    bear = (sig.critic_bear or "").strip() or "<i>—</i>"
    # Backward compat: if `prediction` empty but legacy `future_outlook` is set, use it
    prediction = (sig.prediction or sig.future_outlook or "").strip() or "<i>—</i>"
    prediction_mid = (sig.prediction_mid or "").strip() or "<i>—</i>"
    prediction_long = (sig.prediction_long or "").strip() or "<i>—</i>"

    # Portfolio line — only when held
    if sig.is_portfolio_holding:
        portfolio_line = "\n💼 <i>В портфеле — расчёты учитывают твой P&amp;L</i>\n"
    else:
        portfolio_line = "\n"

    # Verdict execution line: prefer how_to_execute, fallback to why_now_short
    how_to = (sig.how_to_execute or "").strip()
    why = (sig.why_now_short or "").strip()
    if how_to and how_to != "—":
        verdict_exec_line = f"<i>{_esc(how_to)}</i>"
    elif why:
        verdict_exec_line = f"<i>{_esc(why)}</i>"
    else:
        verdict_exec_line = ""

    return INVESTMENT_CARD_TEMPLATE.format(
        display_name=_esc(display_name(sig.asset)),
        price=_esc(f"{sig.price:,.2f}"),
        change_24h=_esc(f"{sig.change_24h:+.1f}%"),
        portfolio_line=portfolio_line,
        news_block=news_block,
        trend=_esc(trend) if not trend.startswith("<i>") else trend,
        critic_bull=_esc(bull) if not bull.startswith("<i>") else bull,
        critic_bear=_esc(bear) if not bear.startswith("<i>") else bear,
        action=_esc(action),
        do_now=_esc(do_now),
        verdict_exec_line=verdict_exec_line,
        prediction=_esc(prediction) if not prediction.startswith("<i>") else prediction,
        prediction_mid=_esc(prediction_mid) if not prediction_mid.startswith("<i>") else prediction_mid,
        prediction_long=_esc(prediction_long) if not prediction_long.startswith("<i>") else prediction_long,
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
    """5 buttons per idea card.

    Row 1: 🔥 Love it / ❌ Not for me
    Row 2: 📌 Save for later / 🔍 Go deeper
    Row 3: 🚀 Going to build  ← high-intent signal for learning loop
    """
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔥 Love it", callback_data=f"like_{idea_id}"),
            InlineKeyboardButton("❌ Not for me", callback_data=f"dislike_{idea_id}"),
        ],
        [
            InlineKeyboardButton("📌 Save for later", callback_data=f"save_{idea_id}"),
            InlineKeyboardButton("🔍 Go deeper", callback_data=f"deep_{idea_id}"),
        ],
        [
            InlineKeyboardButton("🚀 Going to build", callback_data=f"build_{idea_id}"),
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
    """5 buttons per investment card: Useful / Skip / Watch / Full analysis /
    + В мой портфель (Add to portfolio — starts the ConversationHandler that
    asks the user for quantity + avg buy price, then inserts into the DB)."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Useful",        callback_data=f"inv_like_{sig_id}"),
            InlineKeyboardButton("❌ Not relevant",  callback_data=f"inv_skip_{sig_id}"),
        ],
        [
            InlineKeyboardButton("📌 Watch this",    callback_data=f"inv_watch_{sig_id}"),
            InlineKeyboardButton("📊 Full analysis", callback_data=f"inv_deep_{sig_id}"),
        ],
        [
            InlineKeyboardButton("➕ В мой портфель", callback_data=f"inv_add_{sig_id}"),
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
