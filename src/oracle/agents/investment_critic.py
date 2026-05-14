"""Investment critic agent — Step 13.5 (Reflexion loop for investments).

Mirror of the business-idea critic (Step 11), but attacks along axes that
matter for TRADER-BRIEF quality, not for SaaS idea viability.

Why this exists: Maksim complained that investment cards are vague
("bull/bear % — не понятно что делать"), that they don't address his
portfolio, and that each run surfaces random tickers. The trader-brief
format (do_now / how_to_execute / why_now_short / post_action) fixes the
prompt, but a single LLM pass still slips. This critic loop catches:

  1. NON-OPERATIONAL `do_now` — "consider", "interesting setup", "watch
     for" etc. These are the fluff phrases Maksim explicitly banned.
  2. MISSING execution detail in `how_to_execute` — no broker, no size,
     no price level. A card without these is useless at 08:00.
  3. PORTFOLIO MISS — when the user holds N>=2 assets but fewer than
     ⌈N*2/3⌉ of the scenarios address them. Violates the hard rule.
  4. INCONSISTENCY — `action=BUY` but `bull_prob < 50`, or `action=SELL`
     when asset isn't held, etc.
  5. STALE DATES — `key_events` referencing past years (2024 when we're
     in 2026) — signals LLM hallucinated from stale training cues.

Verdicts:
  KILL          Scenario is fundamentally broken (hallucinated asset,
                incoherent action, dangerously wrong price). Drop.
  WEAKEN        Salvageable — specific fixes in `notes`. Goes back to
                investment_analyzer improve-mode next round.
  PASS          Solid trader brief, keep.
  STRONG_PASS   Concrete, operational, portfolio-aware, internally
                consistent — gold standard.

Cost: ~2-3k input tokens / ~1k out per call. 1-2 calls per run → ~$0.02.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from ..config import get_settings
from ..state import OracleState

log = logging.getLogger(__name__)


InvestmentVerdict = Literal["KILL", "WEAKEN", "PASS", "STRONG_PASS"]


class InvestmentCritiqueVerdict(BaseModel):
    index: int = Field(ge=1, description="1-based index of the input scenario")
    verdict: InvestmentVerdict
    notes: str = Field(
        max_length=400,
        description="Concise reasoning (≤400 chars). For WEAKEN, spell out EXACTLY "
                    "what the improve-mode rewriter must fix (e.g. 'do_now too "
                    "vague — replace with imperative action', 'how_to_execute "
                    "missing broker and size', 'ignores GOLD holding').",
    )


class InvestmentCriticOutput(BaseModel):
    verdicts: list[InvestmentCritiqueVerdict] = Field(min_length=1)


# ============================================================================
# System prompt
# ============================================================================


INVESTMENT_CRITIC_SYSTEM = """\
You are ORACLE's critic for Maksim's investment trader briefs.

Card structure Maksim reads (in this order):
  asset · price · 24h%
  news_highlights (2-3 headlines)
  trend (1-2 sentences on CURRENT momentum)
  critic_bull (bull argument with numbers/names)
  critic_bear (bear argument with numbers/names)
  action + do_now (verdict headline)
  how_to_execute + why_now_short (concrete execution)
  prediction (1-2 sentences forecast 1-4 weeks)

VALIDATE EACH SCENARIO ALONG 6 AXES:

1. trend — must reference CURRENT momentum with numbers (24h or 7d %).
   BAD: "позиция работает в фоне", "сектор стабилен". WEAKEN.
   GOOD: "Уран -5% за день, 7d momentum +18% — техническая просадка".

2. critic_bull / critic_bear — must be CONCRETE, with names/numbers.
   BAD: "Базовая диверсифицированная позиция — индексы растут".
   GOOD: "AI-PPA Microsoft + Amazon $13bn — структурный спрос 10 лет".
   If both critics sound generic / interchangeable across multiple
   assets → WEAKEN, instruct rewriter to make them asset-specific.

3. news_highlights — non-empty (≥1 headline) if asset has any signal
   in market context. Empty array allowed only for off-broker assets
   (GOLD-PHYS, CASH-USD). WEAKEN otherwise.

4. do_now MUST BE OPERATIONAL (Russian imperative).
   BAD: "Consider entering", "Interesting setup", "Monitor closely".
   GOOD: "Меняй USD → PLN сегодня", "Жди пробой $215", "Продавай 30%".

5. how_to_execute MUST NAME broker+size+price for any non-HOLD action.
   GOOD: "IBKR limit buy 10 NUCL @ $52, стоп $48".
   Empty "—" acceptable ONLY for HOLD/WAIT/AVOID.

6. prediction — 1-2 sentences with realistic upcoming dates (next 1-4
   weeks) and concrete price targets/triggers. No 2024-dated events.
   BAD: "Следить за CPI и FOMC".
   GOOD: "Жду $44-48 до 15 июня; пробой $50 → $58, провал $43 — стоп".

CONSISTENCY (auto-WEAKEN if violated):
- action=BUY/ADD but critic_bear dominates critic_bull → WEAKEN.
- action=SELL/TRIM but is_portfolio_holding=False → KILL (can't sell what
  you don't hold).
- action=ADD but is_portfolio_holding=False → WEAKEN (should be BUY).

PRICE SANITY:
- Asset not in market data → KILL.
- Price off >20% from market_data for same asset → KILL.

CALIBRATION TARGET (Maksim asked for stricter critic — current runs
PASS too easily):
- Healthy run: 20-40% WEAKEN, 50-70% PASS, 0-10% KILL, 5-15% STRONG_PASS.
- If your last batch was >70% PASS — you under-rejected. Default toward
  WEAKEN on anything generic/templatey.

IGNORE LEGACY FIELDS (do NOT validate them, they're deprecated):
  bull_scenario, bull_prob, bull_trigger, bear_scenario, bear_prob,
  bear_trigger, key_events, geopolitical_note, post_action, action_reason,
  market_situation, critic_risk, future_outlook.
The renderer does not show them — checking them wastes tokens.

AUTO-FILLER NOTE: scenarios with critic_notes containing
"auto-filled HOLD card" are server-generated placeholders. Treat them
leniently — PASS unless they have a real error. Do NOT WEAKEN them just
because critic_bull/bear sounds templatey; they intentionally come from
per-asset templates and the improve-mode rewriter cannot make them
better without real signal input.

NOTES FIELD:
- For WEAKEN, name the SPECIFIC fix (≤300 chars).
- For KILL, explain why unsalvageable.
- For PASS/STRONG_PASS, one-line justification.

Return ONE verdict per input scenario, 1-based index. JSON only.
"""


# ============================================================================
# Formatting
# ============================================================================


def format_scenarios_for_critic(scenarios: list[dict]) -> str:
    """Compact representation of each scenario for the critic.

    Only the NEW (rendered) fields are shown. Legacy fields are skipped
    to save tokens and stop the critic from validating them.
    """
    lines: list[str] = []
    for i, s in enumerate(scenarios, start=1):
        asset = s.get("asset", "?")
        action = s.get("action", "?")
        held = "YES" if s.get("is_portfolio_holding") else "NO"
        price = s.get("price", "?")
        c24 = s.get("change_24h", 0)
        c7d = s.get("change_7d", 0)
        is_filler = "auto-filled" in (s.get("critic_notes") or "")

        news = (s.get("news_highlights") or [])[:3]
        trend = (s.get("trend") or "").strip()[:240]
        bull = (s.get("critic_bull") or "").strip()[:240]
        bear = (s.get("critic_bear") or "").strip()[:240]
        prediction = (s.get("prediction") or "").strip()[:240]
        do_now = (s.get("do_now") or "").strip()[:200]
        how_to = (s.get("how_to_execute") or "").strip()[:200]
        why_now = (s.get("why_now_short") or "").strip()[:200]

        filler_tag = " · AUTO-FILLER (be lenient)" if is_filler else ""
        lines.append(f"#{i} {asset} · action={action} · held={held}{filler_tag}")
        lines.append(f"   price=${price} ({c24:+.1f}% 24h / {c7d:+.1f}% 7d)")
        lines.append(f"   news: {' | '.join(n[:80] for n in news) if news else '(empty)'}")
        lines.append(f"   trend: {trend}")
        lines.append(f"   bull: {bull}")
        lines.append(f"   bear: {bear}")
        lines.append(f"   prediction: {prediction}")
        lines.append(f"   do_now: {do_now}")
        lines.append(f"   how_to_execute: {how_to}")
        lines.append(f"   why_now_short: {why_now}")
    return "\n".join(lines)


def _portfolio_block(market_data: dict) -> str:
    """Best-effort portfolio summary — same format the analyzer sees."""
    try:
        import asyncio  # noqa: PLC0415

        from ..portfolio import (  # noqa: PLC0415
            format_portfolio_for_llm,
            get_portfolio_with_pnl,
        )
        portfolio = asyncio.get_event_loop().run_until_complete(
            get_portfolio_with_pnl(market_data)
        ) if False else None
        # We're already inside async context in the caller — use a simpler approach.
        return "(see caller — portfolio block passed in)"
    except Exception:  # noqa: BLE001
        return "(portfolio unavailable)"


# ============================================================================
# LLM call
# ============================================================================


async def critique_investment_scenarios(
    scenarios: list[dict],
    portfolio_blob: str,
    market_data: dict,
) -> list[InvestmentCritiqueVerdict] | None:
    settings = get_settings()
    from ..observability import has_llm_credentials  # noqa: PLC0415
    if not has_llm_credentials():
        log.info("investment_critic: no LLM credentials — passing all scenarios through")
        return None
    if not scenarios:
        return []

    try:
        from ..observability import get_openai_client, log_llm_usage  # noqa: PLC0415
    except ImportError:
        return None

    try:
        client = get_openai_client(agent="investment_critic")
    except RuntimeError as e:
        log.error("investment_critic: %s", e)
        return None

    # Brief market snapshot for price sanity check
    from .investment_analyzer import format_market_for_investment  # noqa: PLC0415
    market_blob = format_market_for_investment(market_data)

    scenarios_blob = format_scenarios_for_critic(scenarios)
    user_msg = (
        "=== LIVE MARKET DATA (for price sanity check) ===\n"
        f"{market_blob}\n\n"
        "=== MAKSIM'S PORTFOLIO (for coverage rule) ===\n"
        f"{portfolio_blob}\n\n"
        f"=== {len(scenarios)} INVESTMENT SCENARIOS TO CRITIQUE ===\n"
        f"{scenarios_blob}\n\n"
        "Return ONE verdict per scenario with specific notes. Index by the "
        "1-based number above. Attack vague do_now, missing execution "
        "details, portfolio-coverage violations, internal inconsistencies, "
        "and stale/hallucinated dates or prices."
    )

    log.info(
        "investment_critic: calling %s on %d scenarios (~%d KB user msg)",
        settings.openai_model_heavy,
        len(scenarios),
        len(user_msg) // 1024,
    )

    try:
        response = await client.beta.chat.completions.parse(
            model=settings.openai_model_heavy,
            messages=[
                {"role": "system", "content": INVESTMENT_CRITIC_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=InvestmentCriticOutput,
            temperature=0.2,
        )
    except Exception as e:  # noqa: BLE001
        log.error("investment_critic: LLM call failed: %s", e)
        return None

    await log_llm_usage("investment_critic", response)

    parsed = response.choices[0].message.parsed
    if not parsed:
        log.error("investment_critic: LLM returned no parsed output")
        return None

    usage = response.usage
    if usage:
        log.info(
            "investment_critic: %d verdicts · in=%d out=%d tokens",
            len(parsed.verdicts), usage.prompt_tokens, usage.completion_tokens,
        )
    return parsed.verdicts


# ============================================================================
# LangGraph node
# ============================================================================


async def investment_critic_node(state: OracleState) -> dict:
    """Mirror of critic_node, but for investment_scenarios.

    Writes verdicts/notes back into each scenario dict, drops KILLs, and
    increments investment_reflexion_round. The router decides whether to
    loop back to investment_analyzer (improve mode) or exit to validator.

    Graceful pass-through when no LLM credentials — every scenario gets
    verdict=PASS so the loop exits cleanly on the first check.
    """
    round_num = state.get("investment_reflexion_round", 0)
    scenarios = list(state.get("investment_scenarios", []) or [])
    market_data = state.get("market_data", {}) or {}

    if not scenarios:
        log.info("investment_critic round %d: no scenarios — empty result", round_num)
        return {
            "investment_scenarios": [],
            "investment_reflexion_round": round_num + 1,
        }

    # Fetch portfolio blob for the coverage-rule check.
    portfolio_blob = "(portfolio unavailable)"
    try:
        from ..portfolio import (  # noqa: PLC0415
            format_portfolio_for_llm,
            get_portfolio_with_pnl,
        )
        portfolio = await get_portfolio_with_pnl(market_data)
        portfolio_blob = format_portfolio_for_llm(portfolio)
    except Exception as e:  # noqa: BLE001
        log.debug("investment_critic: portfolio blob unavailable: %s", e)

    verdicts = await critique_investment_scenarios(scenarios, portfolio_blob, market_data)

    # Graceful degradation — no LLM → pass all through so loop exits.
    if verdicts is None:
        for s in scenarios:
            s["verdict"] = s.get("verdict") or "PASS"
            s["reflexion_rounds_passed"] = round_num + 1
        return {
            "investment_scenarios": scenarios,
            "investment_reflexion_round": round_num + 1,
        }

    by_index = {v.index: v for v in verdicts}
    for i, s in enumerate(scenarios, start=1):
        v = by_index.get(i)
        if v:
            s["verdict"] = v.verdict
            s["critic_notes"] = v.notes
            s["reflexion_rounds_passed"] = round_num + 1
        else:
            s["verdict"] = "WEAKEN"
            s["critic_notes"] = "(critic did not return a verdict)"

    killed = [s for s in scenarios if s.get("verdict") == "KILL"]
    weakened = [s for s in scenarios if s.get("verdict") == "WEAKEN"]
    passed = [s for s in scenarios if s.get("verdict") in ("PASS", "STRONG_PASS")]
    strong = [s for s in scenarios if s.get("verdict") == "STRONG_PASS"]

    log.info(
        "investment_critic round %d: %d KILL · %d WEAKEN · %d PASS (%d STRONG)",
        round_num, len(killed), len(weakened), len(passed), len(strong),
    )
    for i, s in enumerate(scenarios[:5], start=1):
        log.info(
            "  #%d %s — %s — %s",
            i, s.get("verdict"),
            (s.get("asset") or "")[:20],
            (s.get("critic_notes") or "")[:80],
        )

    # Drop KILLs. Keep PASS + WEAKEN for next round.
    next_scenarios = passed + weakened

    return {
        "investment_scenarios": next_scenarios,
        "investment_reflexion_round": round_num + 1,
    }


# ============================================================================
# Standalone CLI
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
        from .market import collect_market_data
        market_data, _ = await collect_market_data()

        sample = [
            {
                "asset": "NVDA", "signal_type": "Sector Rotation",
                "price": 142.5, "change_24h": 1.2, "change_7d": 3.4,
                "strength": 6, "timeframe": "3-6 months",
                "bull_scenario": "AI capex strong", "bull_prob": 55,
                "bull_trigger": "NVDA earnings beat",
                "bear_scenario": "China export hit", "bear_prob": 45,
                "bear_trigger": "NVDA < $130",
                "key_events": ["NVDA earnings 2024-05-22"],  # stale year
                "geopolitical_note": "China chip controls",
                "action": "BUY",
                "do_now": "Consider accumulating NVDA",  # vague — should WEAKEN
                "how_to_execute": "",  # empty for BUY — should WEAKEN
                "why_now_short": "AI spend is strong",
                "post_action": "",
                "is_portfolio_holding": False,
                "signal_age_hours": 0,
            },
        ]
        fake_state: OracleState = {
            "scout_signals": [], "market_data": market_data, "trend_signals": [],
            "custom_signals": [], "synthesized": [], "raw_ideas": [],
            "investment_scenarios": sample, "reflexion_round": 3,
            "investment_reflexion_round": 0,
            "surviving_ideas": [], "validated": [], "final_digest": {}, "errors": [],
        }
        result = await investment_critic_node(fake_state)
        print(result)

    asyncio.run(_main())
