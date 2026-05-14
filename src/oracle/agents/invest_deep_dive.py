"""Investment deep-dive — invoked when Maksim taps 📊 Full analysis on a
signal card. ONE focused gpt-5.4-mini call that produces:

  - technical_levels: support / resistance / 200-DMA with concrete prices
  - upcoming_catalysts: 2-4 SCHEDULED events in next 3 months (real dates)
  - historical_analogues: similar setup that played out in the past +
    what happened (educational, not prediction)
  - sizing_recommendation: % of portfolio for this position given his
    current allocation and risk profile
  - downside_risk: max realistic drawdown + the 1 trigger that causes it

Cost ~$0.01. Tight prompt designed for gpt-5.4-mini structured output.
Gracefully no-ops without LLM credentials.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ..config import get_settings

log = logging.getLogger(__name__)


class InvestDeepDive(BaseModel):
    technical_levels: str = Field(
        description=(
            "2-3 sentences in Russian on price structure: support, resistance, "
            "200-DMA, volume profile. CONCRETE levels (e.g. '$48 support, "
            "$58 resistance, 200-DMA $52'). NOT 'consult chart' style."
        )
    )
    upcoming_catalysts: list[str] = Field(
        description=(
            "2-4 SCHEDULED events in the next 3 months that move this asset. "
            "Each item: 'YYYY-MM-DD — event description'. ONLY events you are "
            "confident actually happen on that date (earnings, FOMC, OPEC+, "
            "elections, CPI release). NO 'around this time' fluff. If you "
            "don't know specific dates, give fewer items rather than guess."
        ),
        min_length=1,
        max_length=4,
    )
    historical_analogues: str = Field(
        description=(
            "2-3 sentences in Russian. ONE similar setup from past 5-10 years "
            "that played out — describe both setup and outcome. Educational "
            "framing. Example: 'Уран 2023: рост на AI-PPA narrative до $107, "
            "затем коррекция -35% на Казахстан supply news. Текущий setup "
            "напоминает август 2023 — рост на PPA-сделках MSFT-CEG.'"
        )
    )
    sizing_recommendation: str = Field(
        description=(
            "2-3 sentences in Russian on what % of portfolio is reasonable for "
            "this position given Maksim's current allocation (passed in input). "
            "Reference his actual holdings + risk profile. Example: 'У тебя "
            "уже 9% в NUCL — добор до 12-13% оправдан на просадке к $48. "
            "Дальше — концентрация выше комфортного для $20k портфеля.'"
        )
    )
    downside_risk: str = Field(
        description=(
            "1-2 sentences in Russian. Realistic max drawdown % from current "
            "price + ONE concrete trigger that causes it. Example: "
            "'-25% к $36 при остановке Казахстан-поставок или отмене MSFT-PPA.'"
        )
    )


SYSTEM_PROMPT = """\
ORACLE investment deep-dive analyst for Maksim's $20k portfolio.

Input: ONE InvestmentSignal + his current portfolio allocation.

Output 5 fields in Russian. Be CONCRETE — actual prices, dates, names,
percentages. Generic phrases ("monitor closely", "around this time") = fail.

Anchor to Maksim's actual allocation (sizing_recommendation must reference
the % he currently has in this and related positions).

Educational analysis only — no buy/sell language. Frame everything as
"setup", "scenario", "what would have to happen for X".

JSON only.
"""


async def run_invest_deep_dive(
    signal_dict: dict[str, Any],
    portfolio_summary: str,
) -> dict[str, Any] | None:
    """One LLM call — returns structured deep-dive or None on failure."""
    from ..observability import has_llm_credentials  # noqa: PLC0415
    if not has_llm_credentials():
        return None

    try:
        from ..observability import get_openai_client, log_llm_usage  # noqa: PLC0415
        client = get_openai_client(agent="invest_deep_dive")
    except Exception as e:  # noqa: BLE001
        log.warning("invest_deep_dive: client init failed: %s", e)
        return None

    settings = get_settings()

    user_msg = (
        f"=== ASSET ===\n"
        f"Asset: {signal_dict.get('asset', '')}\n"
        f"Price: ${signal_dict.get('price', 0):,.2f}\n"
        f"24h: {signal_dict.get('change_24h', 0):+.1f}%   7d: {signal_dict.get('change_7d', 0):+.1f}%\n"
        f"Held in portfolio: {signal_dict.get('is_portfolio_holding')}\n"
        f"Current trend (from analyzer): {(signal_dict.get('trend') or '')[:300]}\n"
        f"Bull view: {(signal_dict.get('critic_bull') or '')[:200]}\n"
        f"Bear view: {(signal_dict.get('critic_bear') or '')[:200]}\n"
        f"Recent news: {' | '.join((signal_dict.get('news_highlights') or [])[:3])}\n\n"
        f"=== PORTFOLIO CONTEXT ===\n"
        f"{portfolio_summary[:2500]}\n\n"
        f"Produce the 5-section deep-dive JSON per schema. Anchor sizing "
        f"to the actual portfolio allocation above. Educational only."
    )

    log.info("invest_deep_dive: calling %s for %s",
             settings.openai_model_light, signal_dict.get("asset", "?"))
    try:
        response = await client.beta.chat.completions.parse(
            model=settings.openai_model_light,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format=InvestDeepDive,
            temperature=0.3,
        )
    except Exception as e:  # noqa: BLE001
        log.error("invest_deep_dive: LLM call failed: %s", e)
        return None

    await log_llm_usage("invest_deep_dive", response)
    parsed = response.choices[0].message.parsed
    if not parsed:
        return None
    return parsed.model_dump()
