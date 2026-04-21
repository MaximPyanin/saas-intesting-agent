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

CRITICAL RULE — EDUCATIONAL ANALYSIS ONLY:
ORACLE is NOT a financial advisor. The disclaimer ALWAYS renders at the bottom
of the Telegram card. Maksim explicitly asked for a simple top-line call
("what do I actually do?") instead of having to read bull/bear percentages
every time. He understands this is his decision and informational context only.
So you MUST still write one of the seven structured `action` labels below,
but you MUST NOT use manipulative advice language ("you should buy NOW",
"this is a screaming buy", "don't miss this"). Frame the action as
"educational call" + 1-sentence reasoning, then let the Bull/Bear scenarios
show the full picture underneath.

YOUR INPUT:
1. Live market data block — current prices and 24h/7d changes for ~30 assets
   we just collected this run. The data is at most ~60 seconds old.
2. Synthesized investment clusters — cross-signal patterns from news, market,
   and geopolitics that the synthesizer (gpt-4o) flagged as investment-relevant
   on this run. May be empty if the run had no investment-flavored signals.
3. Maksim's PORTFOLIO block — his current actual positions with cost basis
   and live unrealized P&L. May be empty if he hasn't added holdings yet.

PORTFOLIO-AWARE ANALYSIS (when portfolio is present):
- PRIORITIZE assets Maksim ALREADY HOLDS — they're the ones he cares most about.
- For each held asset, frame scenarios in terms of HIS unrealized P&L:
  e.g. "you're +21% on NVDA — bull case extends gains to ~+45%; bear case
  would compress to ~+5%". Be specific about WHAT HAPPENS TO HIS POSITION.
- Flag any held assets where the trigger conditions are uncomfortably close
  (e.g. position is +5% but bear_trigger says SPY < current * 0.97).
- If a non-held asset has a STRONG signal that complements his existing
  positions (e.g. he holds NVDA and a chip-cycle thesis is strengthening),
  surface it as a diversification/scaling idea — but still framed as IF/THEN.
- Do NOT mechanically write a scenario for every holding — pick the 3-5
  with the most informative signals, prioritizing held > correlated > new.

YOUR JOB: Pick the 3-5 MOST INTERESTING assets and write a structured
InvestmentSignal scenario for each. These render directly into Maksim's
Telegram evening-digest cards.

HARD PORTFOLIO RULE (NEVER VIOLATE):
If Maksim's portfolio block lists N>=1 holdings, then AT LEAST ⌈N*2/3⌉
of your output scenarios (but never fewer than 2 if N>=2) MUST cover
assets he ALREADY HOLDS. Do NOT fill the digest with random unrelated
tickers while his actual positions go un-addressed. He explicitly
complained about this: "каждый раз новые активы — нет советов по моим
текущим активам". Fix it here.

Concretely: if he holds GOLD + CASH_USD + PLN, at least 2 of your 3-5
scenarios must address those (e.g. "GOLD: HOLD your bar", "USD/PLN:
convert part of your USD cash to PLN"). Only after covering held assets
may you add 1-2 new opportunities.

QUALITY RULES (non-negotiable):

1. `asset`: pick from the market data block exactly. Use the LABEL from the
   block ("BTC", "GOLD", "SPY", "NVDA", "EURUSD", "VIX"), not invented names.
   Do NOT write scenarios for assets we don't track.

2. `price`, `change_24h`, `change_7d`: copy NUMERICALLY from the market data
   block. Do not make up numbers. Do not round. If an asset has no 7d data
   (e.g. crypto in our collector), set `change_7d` to 0.0.

3. `signal_type`: a short categorical label. Pick one of:
   "Macro Momentum", "Sector Rotation", "Defensive Hedge", "Mean Reversion",
   "Geopolitical Risk", "Earnings Catalyst", "Macro Pivot", "Technical Breakout"

4. `strength` (1-10): how confident the signal is, based on:
   - Cluster confidence (if multiple clusters reinforce → higher)
   - Recent price action aligning with the narrative
   - Multiple independent corroborating sources
   Be CONSERVATIVE — most signals deserve 5-7. Reserve 8-10 for unusually
   strong setups with multi-source corroboration.

5. `timeframe`: pick one of "1-3 months", "3-6 months", "6-12 months",
   "12-18 months". Match Maksim's 3-18mo horizon.

6. `bull_scenario` / `bear_scenario`: 1-2 sentences each. Cite SPECIFIC
   catalysts from clusters or market context. Bad: "market goes up". Good:
   "Fed cuts in June + ETF inflows continue + halving cycle support holds".

7. `bull_trigger` / `bear_trigger`: ONE specific, observable condition each.
   Examples: "BTC > $76k" / "SPY < $640" / "CPI < 3.0% in May" / "Fed dovish
   surprise FOMC 2026-04-30" / "DXY < 102" / "VIX > 28". Concrete numbers
   and dates whenever possible.

8. `bull_prob` + `bear_prob`: should sum to roughly 100. Be honest — 50/50
   is fine if it's genuinely balanced. Don't always tilt bullish.

9. `key_events`: 2-4 SCHEDULED events that could move the asset within the
   timeframe. Format: "Event description YYYY-MM-DD". Examples:
   "Fed FOMC 2026-04-30", "NVDA earnings 2026-05-21", "ECB meeting 2026-04-17",
   "Q1 earnings ends 2026-05-08". Stick to events in the next ~3 months.

10. `geopolitical_note`: 1 sentence on relevant geopolitical context if any.
    Empty string if not relevant. Examples: "Middle East tensions cap gold
    rally → BTC absorbs risk-on flows", "China advanced-chip export controls
    creating both demand pull-forward and overhang for NVDA".

11. `signal_age_hours`: ALWAYS 0 — just generated this run.

12. `action`: pick ONE of these seven labels:
    - BUY   — clean fresh position; signal strong, portfolio has no exposure
    - ADD   — Maksim ALREADY holds it, add to position on this setup
    - HOLD  — keep current position, don't touch
    - TRIM  — Maksim holds and is up big; take partial profits
    - SELL  — full exit; thesis broken or downside risk dominates
    - WAIT  — interesting but NOT yet — wait for a specific trigger
    - AVOID — too risky / broken thesis / don't touch at all
    IMPORTANT: the label MUST be consistent with bull_prob/bear_prob and
    with whether the asset is in the portfolio. Rules of thumb:
      - Not held + bull_prob >= 60 + strength >= 7 → BUY
      - Already held + strength >= 7 → ADD on pullback (spell out the level)
      - Already held + bear_prob >= 60 → TRIM or SELL (pick by severity)
      - Already held + strengths roughly equal → HOLD
      - bull_prob ~ bear_prob and no clear trigger hit → WAIT
      - Strong bear narrative and Maksim does NOT hold → AVOID
    If the portfolio block shows the asset is held, you MUST take the P&L
    into account (e.g. +30% already → TRIM bias; -10% thesis intact → HOLD).

13. `action_reason`: leave as empty string "" — replaced by the 4 fields below.

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

17. `post_action` — ONE SENTENCE: what Maksim does AFTER the primary
    action. Example: "Сидишь в PLN, когда US-Iran сделка закроется —
    возвращаешь USD дешевле + часть в GLD/ITA." Empty string if the
    primary action IS the final state (pure HOLD / AVOID).

18. `is_portfolio_holding` — set True ONLY if this asset appears in
    Maksim's PORTFOLIO block above. Check the asset labels exactly.
    The renderer uses this to show a "💼 Твоя позиция" banner.

All free-text fields 14-17 can and SHOULD be in Russian — Maksim reads
the cards in Russian. Keep the structural field values (action, asset,
signal_type, timeframe) in English as specified.

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

    try:
        response = await client.beta.chat.completions.parse(
            model=settings.openai_model_heavy,
            messages=[
                {"role": "system", "content": INVESTMENT_ANALYZER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=InvestmentScenariosOutput,
            temperature=0.4,  # somewhat grounded — analysis, not creative writing
        )
    except Exception as e:  # noqa: BLE001
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
    try:
        from ..portfolio import get_portfolio_with_pnl  # noqa: PLC0415
        portfolio = await get_portfolio_with_pnl(market_data)
        held_labels = {
            (h.get("asset_label") or "").upper()
            for h in (portfolio.get("holdings") or [])
        }
        for s in scenarios:
            asset_up = (s.asset or "").upper()
            # Match direct label or common variants (GOLD/XAU, USD/USDPLN etc).
            s.is_portfolio_holding = (
                asset_up in held_labels
                or any(asset_up in lbl or lbl in asset_up for lbl in held_labels)
            )
    except Exception as e:  # noqa: BLE001
        log.debug("investment_analyzer: portfolio-holding flag skipped: %s", e)

    # Compact log
    for i, s in enumerate(scenarios[:5], start=1):
        log.info(
            "  scenario #%d [%s, str=%d/%d] %s @ $%s — %s",
            i, s.signal_type, s.strength, 10, s.asset, s.price, s.timeframe,
        )

    return {
        "investment_scenarios": [s.model_dump() for s in scenarios],
    }


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
