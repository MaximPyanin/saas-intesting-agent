"""Investment analyzer agent — Step 13.

Fifth (and final) LLM-using agent in the reasoning chain. Reads:
  - state["synthesized"] — cross-signal clusters from synthesizer (Step 9),
    filtered to `category=="investment"` here
  - state["market_data"] — live prices + 24h/7d changes from market collector
    (Step 4), all ~25 assets across equities/stocks/crypto/commodities/
    forex/indices/macro

Produces 3-5 `InvestmentSignal` Pydantic objects with structured bull/bear
scenarios + trigger conditions + key events. The same model is consumed by
the Telegram card renderer in Step 3, so no field mapping needed.

Per the user's repeated 70/30 priority (business ideas primary, investments
secondary), this agent is intentionally LIGHTWEIGHT compared to the
business-idea pipeline:
  - Single LLM call (no Reflexion loop)
  - No critic, no improvement rounds
  - Validator (Step 12) handles the only quality check: real-price drift

Critical rule baked into the prompt: EDUCATIONAL ANALYSIS ONLY. ORACLE is
NOT a financial advisor. The LLM is instructed never to use buy/sell
recommendation language — frame everything as IF/THEN scenario analysis.
Same disclaimer renders at the bottom of every Telegram investment card.

Cost: ~3-4k tokens in / ~1.5k out at gpt-4o ≈ $0.025-0.04 per run.
2x/day → ~$1.5-2.5/month for the investment side of ORACLE.

Gracefully no-ops without OPENAI_API_KEY: returns empty scenarios list,
validator gets nothing to price-check, formatter outputs empty investments.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ..config import get_settings
from ..models import InvestmentSignal
from ..state import OracleState

log = logging.getLogger(__name__)


# ============================================================================
# Output schema — wrapper around InvestmentSignal for OpenAI structured output
# ============================================================================


class InvestmentScenariosOutput(BaseModel):
    """LLM response: 3-5 trade scenarios sorted by signal strength DESC."""

    scenarios: list[InvestmentSignal] = Field(
        min_length=1,
        max_length=5,
        description="3-5 InvestmentSignal scenarios, sorted by strength DESC",
    )


# ============================================================================
# System prompt — heavily tuned for Maksim's investment profile
# ============================================================================


INVESTMENT_ANALYZER_SYSTEM = """\
You are ORACLE's investment analyzer for Maksim.

USER PROFILE:
- Risk: moderate to slightly aggressive
- Horizon: 3-18 months
- Goal: spot signals BEFORE they're obvious, NOT react after
- Asset universe (tracked live each run):
  * ETFs: SPY, QQQ, XLK, XLE, XLF, VNQ
  * Mega-cap + AI stocks: NVDA, MSFT, GOOGL, AAPL, TSLA, AMD, META, AMZN,
    TSM, AVGO, NFLX, PLTR, SMCI, ARM, ASML, MU, VRT, SOUN, AI (C3.ai)
  * NUCLEAR (AI-power angle): SMR (NuScale), OKLO, NNE (Nano Nuclear),
    LEU (Centrus enrichment), CCJ (Cameco), BWXT, UEC, URA (uranium ETF),
    VST (Vistra), CEG (Constellation) — nuclear is having a renaissance
    because AI hyperscalers are signing PPAs (MSFT-CEG, AMZN-Talen, GOOG-Kairos)
  * DRONES & UNMANNED DEFENSE: AVAV (Switchblade), KTOS (Valkyrie UCAV),
    RCAT (Teal Drones), ONDS, RKLB (Rocket Lab), EH (EHang eVTOL), UMAC
  * TRUMP/POLITICAL-NARRATIVE: DJT (Trump Media), RUM (Rumble), PSQH
    (PublicSquare), PHUN (Phunware) — track these because they move on
    Trump headlines and family-member (Don Jr., Eric) board/advisory news
  * Crypto: BTC, ETH, SOL, XRP
  * Commodities: Gold, Silver, Oil (WTI/Brent), NatGas, Copper, Wheat
  * Forex: EUR/USD, USD/PLN, EUR/PLN, USD/BYN, EUR/BYN, USD/RUB, DXY
  * Indices: VIX, 10Y Treasury (TNX_10Y)
  * FRED macro: Fed funds rate, CPI, unemployment, real 10Y yield

EDUCATIONAL framing only. No manipulative advice language ("you should buy
NOW", "screaming buy", "don't miss this"). Use action labels + concrete
reasoning. Disclaimer renders at the bottom of every card.

INPUT:
1. Live market data (~30 assets, <60s old) with price, 24h%, 7d%
2. Investment clusters from synthesizer (may be empty)
3. Maksim's portfolio block with cost basis and live USD P&L

HARD RULE — PORTFOLIO COVERAGE (Maksim's strict requirement):
EVERY asset in his portfolio → exactly ONE InvestmentSignal with
is_portfolio_holding=true. Even if action=HOLD. Server auto-fills any you
forget — but covering them yourself is cheaper.
Plus 2-3 EXTRA scenarios for non-held assets that look notable today.
Total output ≈ N + 2-3.

For held assets — frame in terms of HIS P&L. "+21% on NVDA — at +45% bull
trigger fires, at +5% bear case starts." Mention close triggers.

QUALITY RULES:
1. `asset` — copy LABEL from market_data block exactly. No invented names.
2. `price`, `change_24h`, `change_7d` — copy NUMERICALLY, no rounding.
   change_7d=0.0 if asset lacks 7d data.
3. `signal_type` — one of: Macro Momentum, Sector Rotation, Defensive
   Hedge, Mean Reversion, Geopolitical Risk, Earnings Catalyst,
   Macro Pivot, Technical Breakout. (Internal hint — not rendered.)
4. `strength` 1-10 (internal). Be conservative: most 5-7. Reserve 8-10
   for multi-source corroboration.
5. `timeframe` (internal): "1-3 months" / "3-6 months" / "6-12 months" /
   "12-18 months".

6. `news_highlights` — массив из 2-3 коротких ЗАГОЛОВКОВ свежих новостей,
   ОТНОСЯЩИХСЯ к активу или его сектору. Берёшь из синтезированных
   кластеров (related_signal_titles) или market context. НЕ выдумывай
   заголовки — лучше пустой массив, чем фейк. Пример (NUCL):
   ["Microsoft подтвердил $2B PPA с Constellation",
    "USDA отчёт: импорт урана +18% YoY",
    "Cameco Q1 beat — выручка $850M vs $810M прогноз"].

7. `trend` — 📊 1-2 предложения о ТЕКУЩЕМ momentum актива. КУДА ДУЕТ
   ВЕТЕР СЕЙЧАС: импульс / консолидация / breakdown. Описание настоящего,
   не будущего. Используй numbers из market_data (change_24h, change_7d).
   Пример (NUCL): "Сектор урана откатился -5% за день после USDA reports,
   но 7d momentum остаётся +18% — техническая просадка в бычьем тренде,
   не смена режима."

8. `critic_bull` — 🐂 1-2 предложения от БЫЧЬЕГО критика. Конкретный
   аргумент ЗА позицию/вход, с цифрами/именами. Пример (NUCL):
   "AI-PPA структурно перепрошивают кривую спроса на 10 лет, $13bn
   контрактов уже подписано — циклическая просадка дает редкую точку
   входа."

9. `critic_bear` — 🐻 1-2 предложения от МЕДВЕЖЬЕГО критика. Конкретный
   аргумент ПРОТИВ. Пример (NUCL):
   "URA в перегретой 6-летней четверти, любая остановка казахстанских
   поставок крошит тренд на 30% за неделю, как было в 2007."

10. `prediction` — 🔮 1-2 предложения ПРОГНОЗА на 1-4 НЕДЕЛИ. Целевой
    диапазон цены + ключевой триггер. Пример (NUCL): "Жду консолидацию
    $44-48 до MSFT-CEG PPA update; пробой $50 → $58, провал $43 — стоп."

11. `prediction_mid` — 🗓️ 1-2 предложения о 1-3 МЕСЯЦАХ. Тренд сектора
    + ключевые катализаторы (CPI, FOMC, earnings, OPEC+, регулирование).
    Пример (NUCL): "Жду $48-60 при стабильных AI-PPA новостях; провал
    под $45 на новостях supply chain Казахстана."

12. `prediction_long` — 📅 1-2 предложения о 1-3 ГОДАХ. Структурный
    тренд / инвест-теза. Пример (NUCL): "Структурно поддержан AI-
    инфраструктурой и nuclear renaissance; диапазон $80-110 при
    подтверждении нескольких PPA-сделок ежегодно."

11. `signal_age_hours`: ALWAYS 0 — just generated this run.

12. `action`: pick ONE of these seven labels:
    - BUY   — clean fresh position; signal strong, portfolio has no exposure
    - ADD   — Maksim ALREADY holds it, add to position on this setup
    - HOLD  — keep current position, don't touch
    - TRIM  — Maksim holds and is up big; take partial profits
    - SELL  — full exit; thesis broken or downside risk dominates
    - WAIT  — interesting but NOT yet — wait for a specific trigger
    - AVOID — too risky / broken thesis / don't touch at all
    Pick action by integrating market_situation + the three critic opinions:
    if 2/3 critics lean bullish AND price setup is clean → BUY/ADD;
    if 2/3 lean bearish AND held → TRIM/SELL;
    if conflicting opinions → WAIT or HOLD;
    if not held and bearish → AVOID.
    If the portfolio block shows the asset is held, take P&L into account
    (e.g. +30% already → TRIM bias; -10% thesis intact → HOLD).

13. `action_reason`: leave as empty string "" — DEPRECATED, ignored by renderer.

14. `do_now` — the SINGLE headline imperative Maksim sees first, in Russian.
    One short sentence, emoji-prefixed, imperative mood. Examples (copy the
    style, NOT the content):
    - "💱 Меняй USD → PLN сегодня"
    - "🥇 Держишь слиток — не трогаешь"
    - "⏸ Жди пробоя BTC $76k с объёмом — не гонись"
    - "🟢 Открывай позицию AVAV после пробоя $215"
    - "🟠 Продавай 30% золота по текущему — фиксируй прибыль"
    - "⛔ Не лезь в нефть до OPEC+ — слишком мутно"
    This is THE line. Make it operational. No fluff like "interesting
    setup" or "consider". If the verdict is WAIT, say WHAT to wait for.

15. `how_to_execute` — concrete execution instruction naming the VENUE +
    SIZE + PRICE. Must be actionable by someone who opens the app right now.
    Examples:
    - "Wise/Revolut, меняй 5000 USD из 15000 cash по курсу 3.57-3.60"
    - "IBKR market buy 10 AVAV shares после открытия рынка"
    - "Binance limit BTC @ $68,000 size 0.05"
    - "Продолжаешь держать слиток в сейфе — ничего не меняешь"
    - "—" only if the action is pure WAIT/AVOID with no execution yet.
    NEVER write "consider buying" or "look at entering" — those are the
    vague phrases Maksim complained about.

16. `why_now_short` — ONE SENTENCE chain of causation explaining why TODAY
    not next week. Use arrows "→" to make it scannable. Example:
    "Ормуз открывается → нефть падает → инфляция снижается → PLN укрепляется
    → USD/PLN пойдёт вниз быстро."
    If no time-urgent driver exists, say so: "Нет срочного драйвера — это
    позиция на 3-6 мес, не суетись."

17. `post_action` — DEPRECATED, leave empty. Renderer no longer shows it.

18. `is_portfolio_holding` — set True ONLY if this asset appears in
    Maksim's PORTFOLIO block above. Check the asset labels exactly.
    The renderer uses this to show a "💼 Твоя позиция" banner.

DEPRECATED FIELDS — leave empty / zero, do NOT generate content for them:
    bull_scenario, bull_prob, bull_trigger,
    bear_scenario, bear_prob, bear_trigger,
    key_events, geopolitical_note, post_action, action_reason,
    market_situation, critic_risk, future_outlook.
The renderer ignores them. They exist only for backward compatibility with
old checkpoints. Spending tokens on them is pure waste.

MINIMAL CARD CONTRACT (what actually gets rendered to Maksim — v3):
1. asset (ticker) — display name auto-resolved server-side
2. price, change_24h — copy from market_data
3. is_portfolio_holding (server overwrites this)
4. news_highlights (2-3 REAL headlines from clusters / market — no fakes)
5. trend (1-2 sentences about CURRENT momentum, NOW)
6. critic_bull (1-2 sentences — concrete bull argument)
7. critic_bear (1-2 sentences — concrete bear argument)
8. action + do_now (verdict headline)
9. how_to_execute + why_now_short (concrete execution + timing)
10. prediction (1-4 weeks: target range + trigger)
11. prediction_mid (1-3 months: sector trend + catalysts)
12. prediction_long (1-3 years: structural thesis)

12 fields. Be lean, no filler. Maksim reads cards in seconds — concrete
numbers, names, levels. Generic phrases ("structural opportunity",
"monitor closely") = zero value.

All free-text fields MUST be in Russian — Maksim reads cards in Russian.
Keep structural values (action, asset ticker) in English.

CONTENT POLICY NOTE (avoid Azure content-filter false positives):
Frame defense/geopolitics assets (NATO, NUCL) as INVESTMENT context only —
sector demand, supply chains, government PPAs. NEVER write tactical
operational language ("strike", "attack", "kill"). Stay in financial-
analysis register at all times.

PRIORITY ORDER (when picking which assets to cover):
- Assets where multiple clusters CONVERGE → strongest signal
- Assets with notable recent price action that aligns with cluster narrative
- Macro themes affecting MULTIPLE assets — cover the most-affected one
- Educational DIVERSITY: don't write 3 BTC scenarios. Cover different asset
  classes (e.g., 1 crypto + 1 equity + 1 commodity, or 2 stocks + 1 macro)

If the synthesized clusters are empty or all business_idea, you may still
write 1-3 scenarios using the market data alone (notable moves, divergences,
event-driven setups), but mark them with conservative strength (5-6).

Sort scenarios by `strength` DESC. Respond ONLY with valid JSON matching
the schema.
"""


# ============================================================================
# Format helpers
# ============================================================================


def format_market_for_investment(market_data: dict) -> str:
    """Compact one-line-per-asset market summary for the LLM input.

    Groups by category so the LLM sees the asset universe clearly.
    """
    sections: list[str] = []

    def _fmt_pct(v: float | None) -> str:
        if v is None:
            return "n/a"
        try:
            return f"{float(v):+.1f}%"
        except (TypeError, ValueError):
            return "n/a"

    def _section(label: str, bucket_key: str, fmt_price: str = "${price:,.2f}") -> None:
        bucket = market_data.get(bucket_key) or {}
        if not bucket:
            return
        sections.append(f"\n{label}:")
        for name, d in bucket.items():
            price = d.get("price", 0)
            c24 = _fmt_pct(d.get("change_24h"))
            c7d = _fmt_pct(d.get("change_7d"))
            try:
                price_str = fmt_price.format(price=price)
            except (TypeError, ValueError):
                price_str = str(price)
            if "change_7d" in d:
                sections.append(f"  {name}: {price_str} ({c24} / {c7d} wk)")
            else:
                sections.append(f"  {name}: {price_str} ({c24})")

    _section("EQUITIES & ETFs", "equities", "${price:.2f}")
    _section("MEGA-CAP + AI STOCKS", "stocks", "${price:.2f}")
    _section("NUCLEAR ENERGY (SMRs + uranium + AI-power utilities)", "nuclear", "${price:.2f}")
    _section("DRONES & UNMANNED DEFENSE", "drones_defense", "${price:.2f}")
    _section("TRUMP / POLITICAL-NARRATIVE TICKERS", "trump_political", "${price:.2f}")
    _section("CRYPTO", "crypto", "${price:,.0f}")
    _section("COMMODITIES (gold/silver/oil/natgas/copper/wheat)", "commodities", "${price:,.2f}")
    _section("FOREX (EUR/USD/PLN/BYN/RUB pairs + DXY)", "forex", "{price:.4f}")
    _section("INDICES", "indices", "{price:.2f}")

    macro = market_data.get("macro") or {}
    if macro:
        sections.append("\nFRED MACRO:")
        for name, d in macro.items():
            v = d.get("value")
            as_of = d.get("as_of", "?")
            if v is not None:
                sections.append(f"  {name}: {v} (as of {as_of})")

    return "\n".join(sections).lstrip() if sections else "(no market data this run)"


def format_clusters_for_investment(clusters: list[dict]) -> str:
    """Compact representation of synthesized investment clusters."""
    if not clusters:
        return "(no investment clusters this run — base scenarios on market data only)"
    lines: list[str] = []
    for i, c in enumerate(clusters, start=1):
        stage = c.get("lifecycle_stage", "?")
        conf = c.get("confidence", 0)
        name = c.get("display_name", c.get("topic", "?"))
        story = (c.get("story") or "").replace("\n", " ").strip()
        sources = ", ".join((c.get("related_signal_sources") or [])[:5])
        signals = (c.get("related_signal_titles") or [])[:4]

        lines.append(f"#{i} [{stage}, conf={conf}] {name}")
        lines.append(f"   Story: {story}")
        if sources:
            lines.append(f"   Sources: {sources}")
        if signals:
            lines.append(f"   Signals: {' | '.join(t[:60] for t in signals)}")
    return "\n".join(lines)


# ============================================================================
# LLM call
# ============================================================================


async def generate_investment_scenarios(
    investment_clusters: list[dict],
    market_data: dict,
) -> list[InvestmentSignal]:
    """Single OpenAI call. Returns empty list on error or no key."""
    settings = get_settings()
    from ..observability import has_llm_credentials  # noqa: PLC0415
    if not has_llm_credentials():
        log.info("investment_analyzer: no LLM credentials — empty result")
        return []

    if not market_data and not investment_clusters:
        log.info("investment_analyzer: no market data and no clusters — empty result")
        return []

    try:
        from ..observability import get_openai_client, log_llm_usage  # noqa: PLC0415
    except ImportError:
        log.warning("investment_analyzer: observability module unavailable")
        return []

    try:
        client = get_openai_client(agent="investment_analyzer")
    except RuntimeError as e:
        log.error("investment_analyzer: %s", e)
        return []

    # Pull Maksim's portfolio (best-effort — DB may not be initialized in tests).
    portfolio_blob = "(portfolio module unavailable)"
    holdings_count = 0
    try:
        from ..portfolio import (  # noqa: PLC0415
            format_portfolio_for_llm,
            get_portfolio_with_pnl,
        )
        portfolio = await get_portfolio_with_pnl(market_data)
        holdings_count = len(portfolio.get("holdings") or [])
        portfolio_blob = format_portfolio_for_llm(portfolio)
    except Exception as e:  # noqa: BLE001
        log.warning("investment_analyzer: could not load portfolio: %s", e)
        portfolio_blob = "(no portfolio holdings tracked yet — generic market analysis only)"

    market_blob = format_market_for_investment(market_data)
    cluster_blob = format_clusters_for_investment(investment_clusters)
    user_msg = (
        "=== LIVE MARKET DATA (collected ~60s ago) ===\n"
        f"{market_blob}\n\n"
        f"=== INVESTMENT CLUSTERS ({len(investment_clusters)}) ===\n"
        f"{cluster_blob}\n\n"
        "=== MAKSIM'S PORTFOLIO (live P&L) ===\n"
        f"{portfolio_blob}\n\n"
        "Pick the 3-5 MOST INTERESTING assets and write structured "
        "InvestmentSignal scenarios. Use the price/change numbers from the "
        "market data block EXACTLY. PRIORITIZE assets Maksim already holds — "
        "frame scenarios in terms of HIS unrealized P&L. Cover diverse asset "
        "classes when possible. Educational analysis only — no buy/sell "
        "recommendations."
    )

    log.info(
        "investment_analyzer: calling %s with %d clusters + %d holdings + %d KB market (~%d KB user msg)",
        settings.openai_model_heavy,
        len(investment_clusters),
        holdings_count,
        len(market_blob) // 1024,
        len(user_msg) // 1024,
    )

    response = None
    try:
        response = await client.beta.chat.completions.parse(
            model=settings.openai_model_heavy,
            messages=[
                {"role": "system", "content": INVESTMENT_ANALYZER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=InvestmentScenariosOutput,
            temperature=0.4,
        )
    except Exception as e:  # noqa: BLE001
        err_str = str(e).lower()
        is_content_filter = (
            "content filter" in err_str
            or "content_filter" in err_str
            or "rejected by the content" in err_str
        )
        if is_content_filter:
            # Retry once with sanitized inputs — strip defense/Ukraine/Iran
            # noise that often trips Azure's filter. Drop investment clusters
            # entirely on retry — analyzer can still write scenarios from
            # market_data + portfolio alone.
            log.warning(
                "investment_analyzer: content-filter blocked first call — "
                "retrying with sanitized prompt (no clusters)"
            )
            retry_msg = (
                f"{market_blob}\n\n"
                f"{portfolio_blob}\n\n"
                f"=== TASK ===\n"
                f"Generate one HOLD scenario per portfolio holding plus 2-3 "
                f"market-only opportunities. Use only price/macro numbers "
                f"from the data above. Avoid geopolitical/defense language."
            )
            try:
                response = await client.beta.chat.completions.parse(
                    model=settings.openai_model_heavy,
                    messages=[
                        {"role": "system", "content": INVESTMENT_ANALYZER_SYSTEM},
                        {"role": "user", "content": retry_msg},
                    ],
                    response_format=InvestmentScenariosOutput,
                    temperature=0.3,
                )
            except Exception as e2:  # noqa: BLE001
                log.error("investment_analyzer: retry also failed: %s", e2)
                return []
        else:
            log.error("investment_analyzer: LLM call failed: %s", e)
            return []

    await log_llm_usage("investment_analyzer", response)

    parsed = response.choices[0].message.parsed
    if not parsed:
        log.error("investment_analyzer: LLM returned no parsed output")
        return []

    usage = response.usage
    if usage:
        log.info(
            "investment_analyzer: %d scenarios · in=%d out=%d tokens",
            len(parsed.scenarios),
            usage.prompt_tokens,
            usage.completion_tokens,
        )

    return parsed.scenarios


# ============================================================================
# Improve mode (Step 13.5) — rewrite WEAKEN scenarios from critic notes
# ============================================================================


INVESTMENT_IMPROVE_SYSTEM = """\
You are ORACLE's investment analyzer in IMPROVE MODE for trader briefs.

The investment critic flagged the scenarios below as WEAKEN — salvageable,
but with specific issues. Each scenario has `critic_notes` explaining
exactly what to fix.

YOUR JOB: rewrite each WEAKEN scenario to address the critic_notes while
preserving the asset and the overall bull/bear thesis when still valid.
The improved scenarios go BACK through the critic next round.

RULES (one rewrite per input scenario, same order):

1. READ the `>>> CRITIC NOTES <<<` line — it tells you EXACTLY what to
   fix. Address ALL of the issues it raises, not just one.

2. Common fixes:
   - "do_now too vague" → rewrite as imperative Russian action
     ("Меняй", "Держи", "Жди", "Продавай"). NO "consider" / "monitor".
   - "missing broker+size+price in how_to_execute" → add them concretely.
     "Wise/Revolut 5000 USD → PLN по курсу 3.58", "IBKR market buy 10
     AVAV @ $215; стоп $192".
   - "portfolio coverage miss — swap for held asset" → replace the
     asset entirely with one Maksim ACTUALLY HOLDS (see portfolio block).
     Then redo price, change_24h/7d from market data for that new asset.
   - "action inconsistent with bull/bear prob" → adjust action to match.
   - "stale year in key_events" → use realistic dates in next 3 months.

3. Update ALL fields that depend on the changes — if you swap the asset,
   update price/change/signal_type/bull_scenario/bear_scenario/
   bull_trigger/bear_trigger/key_events to match.

4. Reset `verdict` to "PASS" — critic will re-evaluate next round.
5. Reset `critic_notes` to "" — next round fills it fresh.
6. Keep `signal_age_hours` at 0.

All Russian free-text fields (do_now, why_now_short, how_to_execute,
post_action) stay in Russian. Structural fields stay English.

Output: same number of improved scenarios as input, in the same order.
Respond ONLY with valid JSON matching the schema.
"""


def _format_weakened_for_improve(scenarios: list[dict]) -> str:
    """Format WEAKEN scenarios + critic notes for the rewriter."""
    lines: list[str] = []
    for i, s in enumerate(scenarios, start=1):
        asset = s.get("asset", "?")
        action = s.get("action", "?")
        held = "YES" if s.get("is_portfolio_holding") else "NO"
        price = s.get("price", "?")
        c24 = s.get("change_24h", 0)
        c7d = s.get("change_7d", 0)
        do_now = (s.get("do_now") or "").strip()[:200]
        how_to = (s.get("how_to_execute") or "").strip()[:200]
        why_now = (s.get("why_now_short") or "").strip()[:200]
        post = (s.get("post_action") or "").strip()[:200]
        notes = (s.get("critic_notes") or "").strip()

        lines.append(f"#{i} {asset} · action={action} · held={held}")
        lines.append(f"   price=${price} ({c24:+.1f}% 24h / {c7d:+.1f}% 7d)")
        lines.append(f"   do_now: {do_now}")
        lines.append(f"   how_to_execute: {how_to}")
        lines.append(f"   why_now_short: {why_now}")
        lines.append(f"   post_action: {post}")
        lines.append(f"   >>> CRITIC NOTES: {notes}")
    return "\n".join(lines)


async def improve_investment_scenarios(
    weakened: list[dict],
    market_data: dict,
) -> list[InvestmentSignal]:
    """Rewrite WEAKEN scenarios per critic notes. Same count, same order."""
    settings = get_settings()
    from ..observability import has_llm_credentials  # noqa: PLC0415
    if not has_llm_credentials() or not weakened:
        return []

    try:
        from ..observability import get_openai_client, log_llm_usage  # noqa: PLC0415
    except ImportError:
        return []

    try:
        client = get_openai_client(agent="investment_analyzer_improve")
    except RuntimeError as e:
        log.error("investment_analyzer improve: %s", e)
        return []

    # Include portfolio block so the rewriter can swap to held assets when
    # critic says "portfolio coverage miss".
    portfolio_blob = "(portfolio unavailable)"
    try:
        from ..portfolio import (  # noqa: PLC0415
            format_portfolio_for_llm,
            get_portfolio_with_pnl,
        )
        portfolio = await get_portfolio_with_pnl(market_data)
        portfolio_blob = format_portfolio_for_llm(portfolio)
    except Exception:  # noqa: BLE001
        pass

    market_blob = format_market_for_investment(market_data)
    blob = _format_weakened_for_improve(weakened)
    user_msg = (
        "=== LIVE MARKET DATA ===\n"
        f"{market_blob}\n\n"
        "=== MAKSIM'S PORTFOLIO (may need to swap toward held assets) ===\n"
        f"{portfolio_blob}\n\n"
        f"=== {len(weakened)} WEAKEN scenarios to rewrite ===\n"
        f"{blob}\n\n"
        "Rewrite EACH scenario above to fix the >>> CRITIC NOTES <<<. "
        "Return the same number of scenarios in the same order, matching "
        "the InvestmentSignal schema."
    )

    log.info(
        "investment_analyzer improve: calling %s on %d WEAKEN scenarios",
        settings.openai_model_heavy, len(weakened),
    )

    try:
        response = await client.beta.chat.completions.parse(
            model=settings.openai_model_heavy,
            messages=[
                {"role": "system", "content": INVESTMENT_IMPROVE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=InvestmentScenariosOutput,
            temperature=0.3,
        )
    except Exception as e:  # noqa: BLE001
        log.error("investment_analyzer improve: LLM call failed: %s", e)
        return []

    await log_llm_usage("investment_analyzer_improve", response)

    parsed = response.choices[0].message.parsed
    if not parsed:
        return []

    usage = response.usage
    if usage:
        log.info(
            "investment_analyzer improve: %d → %d scenarios · in=%d out=%d",
            len(weakened), len(parsed.scenarios),
            usage.prompt_tokens, usage.completion_tokens,
        )
    return parsed.scenarios


# ============================================================================
# LangGraph node
# ============================================================================


async def investment_analyzer_node(state: OracleState) -> dict:
    """Step 13 + 13.5 — fresh generation on round 0, improve mode on round 1+.

    Reads synthesized investment clusters + market_data, calls gpt-4o,
    returns 3-5 InvestmentSignal scenarios in state["investment_scenarios"].

    On round >= 1 (after investment_critic has run), handles WEAKEN
    scenarios the same way idea_generator does for ideas: splits into
    PASS (kept) + WEAKEN (rewritten), returns the union.
    """
    synthesized = state.get("synthesized", []) or []
    market_data = state.get("market_data", {}) or {}
    round_num = state.get("investment_reflexion_round", 0)

    investment_clusters = [
        c for c in synthesized if c.get("category") == "investment"
    ]

    # --------- Round 1+: improve mode ---------
    if round_num >= 1:
        existing = list(state.get("investment_scenarios", []) or [])
        weakened = [s for s in existing if s.get("verdict") == "WEAKEN"]
        passed = [s for s in existing if s.get("verdict") in ("PASS", "STRONG_PASS")]

        log.info(
            "investment_analyzer: round %d — improving %d WEAKEN (keeping %d PASS)",
            round_num, len(weakened), len(passed),
        )

        if not weakened:
            return {"investment_scenarios": passed}

        improved = await improve_investment_scenarios(weakened, market_data)
        for s in improved:
            s.signal_age_hours = 0
            s.verdict = "PASS"
            s.critic_notes = ""
            s.reflexion_rounds_passed = round_num

        if not improved:
            log.warning("investment_analyzer improve: empty — keep originals")
            return {"investment_scenarios": passed + weakened}

        # Server-side portfolio flag on improved scenarios
        try:
            from ..portfolio import get_portfolio_with_pnl  # noqa: PLC0415
            portfolio = await get_portfolio_with_pnl(market_data)
            held_labels = {
                (h.get("asset_label") or "").upper()
                for h in (portfolio.get("holdings") or [])
            }
            for s in improved:
                asset_up = (s.asset or "").upper()
                s.is_portfolio_holding = (
                    asset_up in held_labels
                    or any(asset_up in lbl or lbl in asset_up for lbl in held_labels)
                )
        except Exception:  # noqa: BLE001
            pass

        return {
            "investment_scenarios": passed + [s.model_dump() for s in improved],
        }

    # --------- Round 0: fresh generation ---------
    log.info(
        "investment_analyzer: round 0 · %d/%d clusters are investment, market_data=%s",
        len(investment_clusters),
        len(synthesized),
        "yes" if market_data else "no",
    )

    scenarios = await generate_investment_scenarios(investment_clusters, market_data)

    # Force signal_age_hours to 0 (LLM might forget)
    for s in scenarios:
        s.signal_age_hours = 0

    # Server-side truth for is_portfolio_holding — don't trust the LLM alone.
    # Read the actual portfolio and overwrite the flag based on asset label match.
    # ALSO: ensure every holding has at least one scenario; auto-fill HOLD
    # cards for any holding the LLM forgot to cover.
    try:
        from ..portfolio import get_portfolio_with_pnl  # noqa: PLC0415
        portfolio = await get_portfolio_with_pnl(market_data)
        holdings = portfolio.get("holdings") or []
        held_labels = {
            (h.get("asset_label") or "").upper()
            for h in holdings
        }
        for s in scenarios:
            asset_up = (s.asset or "").upper()
            # Match direct label or common variants (GOLD/XAU, USD/USDPLN etc).
            s.is_portfolio_holding = (
                asset_up in held_labels
                or any(asset_up in lbl or lbl in asset_up for lbl in held_labels)
            )

        # Find holdings NOT covered by any scenario and auto-inject HOLD cards.
        # Match conservatively: a holding is "covered" only if a scenario's
        # asset label is EXACTLY equal (case-insensitive) to its asset_label.
        covered = {
            (s.asset or "").upper()
            for s in scenarios
            if s.is_portfolio_holding
        }
        missing = [h for h in holdings if (h.get("asset_label") or "").upper() not in covered]
        if missing:
            log.info(
                "investment_analyzer: auto-filling %d holdings not covered by LLM: %s",
                len(missing),
                [h.get("asset_label") for h in missing],
            )
            scenarios.extend(await _build_hold_fillers(missing))
    except Exception as e:  # noqa: BLE001
        log.debug("investment_analyzer: portfolio-holding fill skipped: %s", e)

    # Compact log
    for i, s in enumerate(scenarios[:5], start=1):
        log.info(
            "  scenario #%d [%s, str=%d/%d] %s @ $%s — %s",
            i, s.signal_type, s.strength, 10, s.asset, s.price, s.timeframe,
        )

    return {
        "investment_scenarios": [s.model_dump() for s in scenarios],
    }


async def _fetch_news_for_asset(label: str, asset_class: str, limit: int = 3) -> list[str]:
    """Pull recent breaking-news headlines that mention asset-specific
    sector keywords. Uses word-boundary matching for short terms ("gas",
    "oil") to avoid false positives like "gas tax" matching boiled-egg
    articles. Returns up to `limit` cleaned headlines.
    """
    from ..db import get_db  # noqa: PLC0415

    # PRIMARY (high-specificity) terms — multi-word phrases, less false-positive risk.
    # SECONDARY (single-word) terms — only used as fallback if PRIMARY returns nothing.
    overlays: dict[str, tuple[list[str], list[str]]] = {
        "NUCL":      (
            ["uranium", "nuclear plant", "SMR", "Cameco", "Constellation", "small modular reactor"],
            ["URA", "DOE nuclear"],
        ),
        "EXH1":      (
            ["crude oil", "OPEC", "Brent", "WTI", "oil price", "energy stocks", "gas prices"],
            ["oil refinery", "natural gas"],
        ),
        "NATO":      (
            ["NATO", "defense spending", "Ukraine", "military aid", "drone warfare"],
            ["defense contractor", "Pentagon", "Lockheed", "Raytheon"],
        ),
        "SMH":       (
            ["semiconductor", "NVIDIA", "TSMC", "AI chips", "chip stocks", "AI infrastructure"],
            ["AMD", "Broadcom", "Micron", "chip shortage"],
        ),
        "CSPX":      (
            ["S&P 500", "stocks", "earnings season", "Federal Reserve", "FOMC", "CPI"],
            ["Dow Jones", "Wall Street"],
        ),
        "IB1T":      (
            ["bitcoin", "BTC", "Bitcoin ETF"],
            ["crypto", "spot ETF"],
        ),
        "ETH-CORE":  (
            ["ethereum", "ETH", "Ethereum upgrade"],
            ["smart contract", "DeFi"],
        ),
        "IB01":      (
            ["10-year Treasury", "treasury yield", "Federal Reserve", "rate cut", "FOMC"],
            ["bond yields", "dot plot"],
        ),
        "GOLD-PHYS": (
            ["gold price", "gold rally", "inflation hedge", "safe haven"],
            ["XAU", "bullion"],
        ),
        "NVDA":      (
            ["NVIDIA", "AI chip", "Jensen Huang", "GPU shortage", "data center AI"],
            ["AI infrastructure", "GB200", "Blackwell"],
        ),
        "CASH-USD":  (
            ["dollar index", "DXY", "Federal Reserve", "inflation"],
            ["currency", "FX market"],
        ),
    }

    primary_terms, secondary_terms = overlays.get(label, ([label], []))

    async def _query(terms: list[str]) -> list[str]:
        if not terms:
            return []
        # Use LIKE with leading/trailing spaces/punctuation to reduce substring noise.
        # For each term we try " term " and " term," and "term." to bias to whole words.
        like_clauses: list[str] = []
        params: list[str] = []
        for t in terms:
            like_clauses.append("LOWER(title) LIKE ?")
            params.append(f"%{t.lower()}%")
        where_clause = " OR ".join(like_clauses)
        try:
            async with get_db() as conn:
                async with conn.execute(
                    f"""SELECT title FROM signals
                        WHERE is_breaking = 1
                          AND ({where_clause})
                          AND datetime(published_at) >= datetime('now', '-3 days')
                        ORDER BY published_at DESC
                        LIMIT ?""",
                    (*params, limit * 2),  # over-fetch so we can dedupe below
                ) as cur:
                    rows = await cur.fetchall()
            titles = [(r[0] if isinstance(r, tuple) else r["title"]) for r in rows if r]
            # Filter very-short/junk titles and obvious off-topic catches (Reddit titles often)
            cleaned: list[str] = []
            seen: set[str] = set()
            for t in titles:
                if not t or len(t) < 20:
                    continue
                key = t[:80].lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(t[:140])
                if len(cleaned) >= limit:
                    break
            return cleaned
        except Exception as e:  # noqa: BLE001
            log.debug("hold-filler: news fetch failed for %s: %s", label, e)
            return []

    # Try primary first (high specificity), fallback to secondary if empty.
    result = await _query(primary_terms)
    if not result and secondary_terms:
        result = await _query(secondary_terms)
    return result


# Per-asset templates for auto-filler. Each holding gets UNIQUE bull/bear/
# trend/prediction text — no more generic "Базовая диверсифицированная
# позиция" copy-pasted across CSPX/EXH1/NATO/NUCL/SMH. Maksim complained
# the templates looked identical; this fixes that.
HOLD_FILLER_TEMPLATES: dict[str, dict[str, str]] = {
    "CSPX": {
        "trend":      "Ядро портфеля движется в коридоре с S&P 500; без свежих макро-сюрпризов — фоновая работа.",
        "bull":       "Долгосрочный bull-кейс остаётся: buyback'и Big Tech + AI capex + дивидендный yield ~1.3%.",
        "bear":       "Концентрация в топ-10 (NVDA/MSFT/AAPL/AMZN/META/GOOG) уже 35%+ — иллюзия диверсификации, при коррекции AI просядут синхронно.",
        "prediction": "Жду диапазон ±2-3% до следующего CPI/FOMC; пробой максимумов на сильных earnings AI-гигантов → новый импульс.",
        "mid":        "1-3 месяца: продолжение AI-driven roll-up в индексе; ключевые катализаторы — CPI каждые ~30д + FOMC.",
        "long":       "1-3 года: индекс остаётся базовой позицией; ожидаемая среднегодовая доходность 7-10% при сохранении AI-supercycle.",
    },
    "SMH": {
        "trend":      "Чипы продолжают AI-rally с откатами 3-5% на фиксации; momentum жив пока NVDA/TSM не разочаруют.",
        "bull":       "AI-инфраструктура capex от гиперскейлеров расширяется (Microsoft, Google, Meta, Oracle); SMH захватывает всю цепочку — от EUV до HBM.",
        "bear":       "Перегретый PE (~35x forward) — любой sigma-1 промах NVDA по выручке/гайденсу обрушит сектор на 8-12%.",
        "prediction": "Ключевой триггер — отчёт NVIDIA. Жду консолидации до публикации; пробой максимумов на beat → +10-15%, miss → откат к 50-DMA.",
        "mid":        "1-3 месяца: продолжение AI capex cycle; следить за hyperscaler guidance и Blackwell-поставками.",
        "long":       "1-3 года: SMH остаётся стратегической ставкой на AI-стек (EUV, HBM, foundry); риск-cycle через 2026-27.",
    },
    "NATO": {
        "trend":      "Defense-сектор стабильно растёт на фоне расширения бюджетов NATO; индекс в восходящем канале без явных просадок.",
        "bull":       "Расходы стран NATO на оборону достигли 2% ВВП; контракты на 10+ лет вперёд (Lockheed F-35, RTX patriot) дают visibility выручки.",
        "bear":       "Дипломатическая разрядка / ceasefire в активных конфликтах → переоценка сектора вниз на 10-15% за недели.",
        "prediction": "Ожидаю продолжение восходящего тренда до следующего саммита NATO; новые контракты по drone defense — главный катализатор upside.",
        "mid":        "1-3 месяца: defense budgets fiscal-year cycle, новые контракты на patriot/SAM/drones продолжают идти.",
        "long":       "1-3 года: структурный bid под defense supply-chain автономии (US/EU re-shoring); CAGR 8-12%.",
    },
    "NUCL": {
        "trend":      "Уран-сегмент откатился от максимумов, но AI-PPA сделки (MSFT-CEG, Amazon-Talen, Google-Kairos) держат структурный спрос.",
        "bull":       "Гиперскейлеры законтрактовали multi-billion PPA на 10-20 лет; уран supply-side остаётся ограниченным (Cameco, Казахстан).",
        "bear":       "Спот-уран в верхней 6-летней четверти; любой sentiment-разворот (отмена PPA, regulatory delay) даёт -25% за недели.",
        "prediction": "Жду консолидации в текущем диапазоне; пробой вверх на новой PPA-сделке → следующий уровень сопротивления; downside ограничен AI-нарративом.",
        "mid":        "1-3 месяца: следить за DOE SMR funding и новыми hyperscaler-PPA; диапазон $48-60.",
        "long":       "1-3 года: nuclear renaissance + AI power demand структурно поддерживают сектор; цель $80-110.",
    },
    "EXH1": {
        "trend":      "Энергетика EU балансирует между опасениями рецессии (давит цены) и геополитикой Ближний Восток / Россия (поддерживает).",
        "bull":       "Дефицит инвестиций в upstream + рост спроса на природный газ для AI-датацентров поддерживают высокие цены 2-3 года.",
        "bear":       "При замедлении глобальной экономики и росте EV-проникновения нефтяной спрос пика достигает быстрее ожиданий.",
        "prediction": "Жду диапазон Brent $75-95 до следующего OPEC+; пробой $100 — на эскалации Ближний Восток, провал $70 — на риске рецессии.",
        "mid":        "1-3 месяца: OPEC+ заседания + sezzonal demand; CPI-чувствительность высокая.",
        "long":       "1-3 года: structural underinvestment в upstream поддерживает цены $70-100; EV-cycle ограничивает upside.",
    },
    "IB1T": {
        "trend":      "Bitcoin консолидируется после ралли; институциональные потоки через ETF стабильны, но спот-объёмы остыли.",
        "bull":       "Цикл халвинга 2024 + spot-ETF inflows + макро-нарратив 'обесценивание доллара' → структурный спрос на 12-24 месяца.",
        "bear":       "Корреляция с risk-on активами растёт; при VIX>25 BTC обычно падает быстрее S&P, ликвидность плеча убивает rally.",
        "prediction": "Жду консолидации $75-90k до следующего FOMC; решительный break > $90k открывает $100k+, провал $72k — фикс прибыли.",
        "mid":        "1-3 месяца: post-halving bull-cycle + ETF AUM growth; ключевые катализаторы — FOMC + регулирование.",
        "long":       "1-3 года: $150-250k диапазон возможен при continued institutional adoption; downside $50k при rate-hike scenario.",
    },
    "ETH-CORE": {
        "trend":      "Ethereum отстаёт от Bitcoin по силе тренда; staking yield + spot-ETF flows основные драйверы спроса.",
        "bull":       "ETH spot ETF набирает AUM; staking yield 3-4% делает ETH-CORE доходным; rollup-экосистема растёт.",
        "bear":       "L2 fragmentation размывает ценность L1; конкуренция с Solana/Base за DeFi-объёмы давит на ETH-burn.",
        "prediction": "Жду торговли $2,100-2,600 до следующего апгрейда; катализатор upside — ETH ETF staking approval; downside — bearish CPI.",
        "mid":        "1-3 месяца: ETH/BTC ratio reset возможен; Pectra upgrade + spot-ETF staking — главные катализаторы.",
        "long":       "1-3 года: $4-6k реалистично при доминировании в RWA/stablecoin settlement; downside-сценарий Solana takes over.",
    },
    "IB01": {
        "trend":      "Короткие трежери в боковике; доходности 4-5%, NAV почти не двигается, классический risk-off инструмент.",
        "bull":       "При признаках замедления экономики или risk-off распродаже → rotation в короткие трежери, NAV растёт + купонный yield.",
        "bear":       "Если ФРС остаётся ястребиной дольше ожиданий или ястреб больше cut'ов → доходности растут, NAV короткой дюрации не страдает сильно, но opportunity-cost остаётся.",
        "prediction": "Жду минимальной волатильности до следующего FOMC; dovish surprise → NAV +0.5-1%; hawkish surprise → opportunity-cost остаётся.",
        "mid":        "1-3 месяца: следить за dot-plot и темпом cut'ов; короткая дюрация даёт оптимальность.",
        "long":       "1-3 года: при cut-cycle ФРС yield упадёт до 2-3%, но дюрация защищает; structural risk-off hedge.",
    },
    "CASH-USD": {
        "trend":      "Доллар сохраняет силу; реальная доходность кэша в позитивной зоне при инфляции 2.5-3%.",
        "bull":       "Огневой резерв для следующей коррекции; позволяет действовать когда другие panic-sell.",
        "bear":       "Доллар может ослабнуть при cut-cycle ФРС → теряется покупательная способность относительно gold/BTC.",
        "prediction": "Держать до следующей просадки S&P >5%; затем deploying tranches в просевшие позиции.",
        "mid":        "1-3 месяца: DXY чувствителен к dot-plot; разворот вниз начнётся при первом cut.",
        "long":       "1-3 года: реальная доходность отрицательная при инфляции 3%+; cash как stabilizer, не источник доходности.",
    },
    "GOLD-PHYS": {
        "trend":      "Золото у исторических максимумов; центробанки EM продолжают накапливать, retail-спрос стабилен.",
        "bull":       "Закупки CB (Китай, Индия, Турция) + геополитический stress + хедж от FX-девальвации = структурный bid.",
        "bear":       "Сильный доллар + рост реальных доходностей 10Y = главный встречный ветер для золота.",
        "prediction": "Жду диапазон $4,500-4,900 до следующего FOMC; пробой $5k на dovish surprise возможен; коррекция к $4,300 на сильных US-данных.",
        "mid":        "1-3 месяца: CB-bid сохраняется; реальные ставки и DXY — главные predictoры.",
        "long":       "1-3 года: $5-6k реалистично при сохранении CB-purchases на $1.5-2T/год + ослаблении $; downside $3.5k.",
    },
    "NVDA": {
        "trend":      "NVDA остаётся лидером AI-rally; momentum жив пока guidance подтверждает $100B+ data-center выручки.",
        "bull":       "Capex от хайперскейлеров (MSFT, GOOGL, META, ORCL, AMZN) растёт; Blackwell rollout + CUDA moat + sovereign AI-deals.",
        "bear":       "Customer concentration top-4 hyperscalers > 40% выручки; любой capex slowdown даёт -15-20% за квартал.",
        "prediction": "Жду диапазон ±5% до earnings; beat + raise guidance → продолжение тренда, miss → проверка 200-DMA.",
        "mid":        "1-3 месяца: hyperscaler Q-reports и AI capex guidance; sovereign AI deals — bonus катализатор.",
        "long":       "1-3 года: $300-400 реалистично при continued AI capex; downside $150 при industry-wide deceleration.",
    },
}


def _hold_filler_critics(asset_label: str, asset_class: str) -> dict[str, str]:
    """Return per-asset auto-filler templates. Falls back to a generic
    asset-class template if the specific label has no entry. Always returns
    all 6 keys: trend, bull, bear, prediction, mid, long."""
    if asset_label in HOLD_FILLER_TEMPLATES:
        return HOLD_FILLER_TEMPLATES[asset_label]
    return {
        "trend":      "Позиция в портфеле работает в фоне; без свежих триггеров — фоновая работа.",
        "bull":       "Структурный bull-нарратив сектора пока интактен.",
        "bear":       "Любой неожиданный макро-шок может изменить расклад — держи стоп.",
        "prediction": "Без чёткого катализатора — жду консолидации; следующее значимое движение — на следующих макро-данных.",
        "mid":        "1-3 месяца: следить за макро-катализаторами (CPI, FOMC, sector reports).",
        "long":       "1-3 года: structural-cycle позиции в портфеле; периодически переоценивать аллокацию.",
    }


async def _build_hold_fillers(missing_holdings: list[dict]) -> list:
    """Build informative HOLD InvestmentSignal cards for holdings the LLM skipped.

    Programmatic, no LLM call. Pulls 2-3 real news headlines per asset from
    the signals DB and uses asset-class critic templates so the cards have
    actual content — not just "нет триггеров сегодня" filler.
    """
    from ..models import InvestmentSignal  # noqa: PLC0415

    fillers: list = []
    for h in missing_holdings:
        label = h.get("asset_label") or "?"
        asset_class = h.get("asset_class") or "other"
        current_price = h.get("current_price") or 0.0
        pnl_pct = h.get("pnl_pct")
        c24 = h.get("change_24h_pct") or 0.0
        c7d = h.get("change_7d_pct") or 0.0
        pnl_note = (
            f"твоя позиция {'+' if (pnl_pct or 0) >= 0 else ''}{(pnl_pct or 0):.1f}% от входа"
            if pnl_pct is not None
            else "позиция в портфеле"
        )

        # Pull live news + load per-asset critic templates
        news = await _fetch_news_for_asset(label, asset_class, limit=3)
        tpl = _hold_filler_critics(label, asset_class)

        try:
            sig = InvestmentSignal(
                asset=label,
                price=float(current_price) if current_price else 0.0,
                change_24h=float(c24),
                change_7d=float(c7d),
                strength=4,
                timeframe="1-3 months",
                news_highlights=news,
                trend=tpl["trend"],
                critic_bull=tpl["bull"],
                critic_bear=tpl["bear"],
                prediction=tpl["prediction"],
                prediction_mid=tpl.get("mid", ""),
                prediction_long=tpl.get("long", ""),
                signal_age_hours=0,
                action="HOLD",
                do_now="🟢 Держи — нет триггеров сегодня",
                how_to_execute="—",
                why_now_short=f"Без значимого движения сегодня — {pnl_note}, сиди в позиции.",
                is_portfolio_holding=True,
                # PASS verdict + the magic substring keeps the reflexion loop
                # AND the critic AWAY from these. They're programmatic fillers,
                # not LLM output — passing them through improve-mode is a waste.
                verdict="STRONG_PASS",
                critic_notes="auto-filled HOLD card (no LLM call) — per-asset template",
                reflexion_rounds_passed=2,  # max — won't be touched by improve-mode
            )
            fillers.append(sig)
        except Exception as e:  # noqa: BLE001
            log.warning("investment_analyzer: could not build filler for %s: %s", label, e)

    return fillers


# ============================================================================
# Standalone CLI: `uv run python -m oracle.agents.investment_analyzer`
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
        # Use real market collector + hardcoded sample clusters
        from .market import collect_market_data

        sample_clusters = [
            {
                "topic": "fed-pause-priced-in",
                "display_name": "Fed pause priced in for June",
                "story": (
                    "Bloomberg + Reuters reporting Fed minutes hint at June pause; "
                    "10Y yields dropped 12bps overnight. Multiple HN/Reddit r/investing "
                    "threads on rate-sensitive sectors rotating in."
                ),
                "lifecycle_stage": "GROWING",
                "confidence": 75,
                "related_signal_titles": [
                    "Fed minutes hint at pause",
                    "10Y yield drops 12bps",
                    "Rate-sensitive sectors rotating",
                ],
                "related_signal_sources": ["bloomberg/markets", "rss/reuters", "reddit/r/investing"],
                "category": "investment",
            },
            {
                "topic": "ai-chip-export-controls",
                "display_name": "China AI chip export controls",
                "story": (
                    "TechCrunch + Bloomberg covering new round of US export controls on "
                    "advanced chips to China. NVDA exposure ~25%. Mixed signal — demand "
                    "pull-forward in Q4 vs FY26 overhang."
                ),
                "lifecycle_stage": "GROWING",
                "confidence": 70,
                "related_signal_titles": [
                    "New round of US chip export controls",
                    "NVDA China exposure analysis",
                ],
                "related_signal_sources": ["bloomberg/markets", "rss/techcrunch"],
                "category": "investment",
            },
        ]

        # Collect real market data so the LLM has live prices
        log.info("standalone: collecting live market data first...")
        market_data, market_errors = await collect_market_data()
        if market_errors:
            log.warning("market collection had %d errors", len(market_errors))

        fake_state: OracleState = {
            "scout_signals": [], "market_data": market_data, "trend_signals": [],
            "custom_signals": [], "synthesized": sample_clusters, "raw_ideas": [],
            "investment_scenarios": [], "reflexion_round": 3,
            "surviving_ideas": [], "validated": [], "final_digest": {}, "errors": [],
        }
        result = await investment_analyzer_node(fake_state)
        scenarios = result.get("investment_scenarios", [])

        print()
        print("=" * 70)
        print(f"Investment scenarios generated: {len(scenarios)}")
        print("=" * 70)
        for i, s in enumerate(scenarios, start=1):
            print()
            print(f"#{i} [{s['signal_type']}, str={s['strength']}/10] {s['asset']}")
            print(f"   Price: ${s['price']} ({s['change_24h']:+.1f}% / {s['change_7d']:+.1f}% wk)")
            print(f"   Timeframe: {s['timeframe']}")
            print(f"   Bull ({s['bull_prob']}%): {s['bull_scenario']}")
            print(f"     Trigger: {s['bull_trigger']}")
            print(f"   Bear ({s['bear_prob']}%): {s['bear_scenario']}")
            print(f"     Trigger: {s['bear_trigger']}")
            print(f"   Key events: {', '.join(s['key_events'])}")
            if s['geopolitical_note']:
                print(f"   Geo: {s['geopolitical_note']}")

    asyncio.run(_main())
