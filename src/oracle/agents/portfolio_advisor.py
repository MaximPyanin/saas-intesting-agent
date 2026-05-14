"""Morning portfolio advisor — Step 20.5 (Maksim's request).

A single cheap LLM call that produces ONE-LINE advice for each holding in
the portfolio. Designed for the 07:00 Warsaw morning brief — not a deep
analysis like the evening digest, just a quick "что делать сегодня" per
position.

INPUT: portfolio dict with live P&L (from portfolio.get_portfolio_with_pnl).
OUTPUT: list of `PortfolioAdvice` items, one per holding.

Cost target: ~$0.02-0.05 per morning (gpt-4o-mini, ~2000 input tokens,
~800 output tokens). Cheaper than running the full evening_digest just to
get HOLD verdicts.

Gracefully no-ops if no LLM credentials — returns deterministic HOLD
advice for every position so the morning brief always has SOMETHING to
render.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ..config import get_settings
from ..portfolio import format_portfolio_for_llm

log = logging.getLogger(__name__)


# ============================================================================
# Output schemas
# ============================================================================


class PortfolioAdvice(BaseModel):
    """One-line advice for a single holding."""

    asset: str = Field(description="Asset label, e.g. 'CSPX', 'IB1T'")
    action: str = Field(
        description=(
            "One of: HOLD | TRIM | ADD | WATCH | REBALANCE | SELL. "
            "Pick HOLD when nothing notable today (most positions, most days)."
        )
    )
    advice_short: str = Field(
        description=(
            "1-sentence Russian advice for today. Imperative mood. "
            "Examples: "
            "'Базовый рост работает — не трогай', "
            "'Перегрев AI-чипов — готовь TRIM на +8%', "
            "'Risk-off растёт — добирай при просадке к $97'."
        )
    )


class PortfolioMorningAdvice(BaseModel):
    """Per-holding advice array."""

    advice: list[PortfolioAdvice] = Field(
        description="One PortfolioAdvice per holding in the input portfolio.",
    )


# ============================================================================
# System prompt
# ============================================================================


SYSTEM_PROMPT = """\
ORACLE morning portfolio advisor for Maksim. Output = "💼 Совет по портфелю"
section of the 07:00 Warsaw brief.

INPUT: portfolio block. Each holding shows cost basis, live USD price,
live 24h% and 7d% changes, P&L from entry.

ANCHOR every advice line on the LIVE 24h/7d numbers, NOT P&L-from-entry.
Severity tiers (use these to calibrate action label):

  24h move           7d move          → recommended action
  ──────────────     ──────────────   ──────────────────────────────
  >+5%               any              → TRIM (strong impulse, take some)
  +3% to +5%         any              → WATCH (готовь TRIM на +X%)
  +1% to +3%         any              → HOLD with momentum note
  -1% to +1%         any              → HOLD with calm note
  -3% to -1%         any              → HOLD with dip note
  -5% to -3%         any              → ADD (просадка, добирай)
  <-5%               any              → ADD (сильная просадка) or WATCH
                                         (если фундаментал сломался)
  any                7d > +10%        → consider TRIM if held big
  any                7d < -10%        → consider ADD if conviction high

RULES:
1. ONE PortfolioAdvice per holding. NO skip, NO duplicate.
2. `action` ∈ {HOLD, TRIM, ADD, WATCH, REBALANCE, SELL}.
3. `advice_short`: ONE Russian sentence, ≤20 words, imperative.
   MUST reference 24h or 7d number when available. Examples:
   - "+1.2% за сутки — держи, без триггеров."
   - "Чипы -2.8% сегодня — добирай при просадке к $84."
   - "AI-бум +6% за неделю — готовь TRIM на следующих +3%."
   - "Risk-off растёт, бонды +0.4% — добирай при просадке."
   - "Огневой резерв, держи." (для cash)
   - "Физ. золото — хедж, не трогай." (для GOLD-PHYS)
4. NO IDENTICAL advice across positions. Each must reference its own
   numbers / sector.
5. Conservative on action labels: 60-80% of positions HOLD on a typical
   day. Aggressive labels (TRIM/SELL) only when severity tier triggers.
6. Cash positions: HOLD usually; REBALANCE only when ≥1 other position
   had 24h < -3% (real sell-off → deploy cash).
7. Physical gold: HOLD/REBALANCE only (off-broker, not tradable).

OUTPUT: `advice` array, one PortfolioAdvice per holding. JSON only.
"""


# ============================================================================
# Main entry point
# ============================================================================


async def generate_morning_portfolio_advice(
    portfolio: dict[str, Any],
) -> list[PortfolioAdvice]:
    """Single LLM call producing one advice line per holding.

    Returns a deterministic fallback (all HOLD) if no LLM creds.
    Returns empty list if portfolio has no holdings.
    """
    holdings = portfolio.get("holdings") or []
    if not holdings:
        return []

    # Fallback: no LLM → trivial HOLD for every holding
    from ..observability import has_llm_credentials  # noqa: PLC0415
    if not has_llm_credentials():
        log.info("portfolio_advisor: no LLM creds — returning trivial HOLD advice")
        return [
            PortfolioAdvice(
                asset=h.get("asset_label") or "?",
                action="HOLD",
                advice_short="Держи — нет анализа сегодня (LLM недоступен).",
            )
            for h in holdings
        ]

    try:
        from ..observability import get_openai_client, log_llm_usage  # noqa: PLC0415
        client = get_openai_client(agent="portfolio_advisor")
    except Exception as e:  # noqa: BLE001
        log.warning("portfolio_advisor: client init failed — fallback HOLD: %s", e)
        return [
            PortfolioAdvice(
                asset=h.get("asset_label") or "?",
                action="HOLD",
                advice_short="Держи — анализ временно недоступен.",
            )
            for h in holdings
        ]

    settings = get_settings()
    portfolio_blob = format_portfolio_for_llm(portfolio)

    user_msg = (
        f"=== Portfolio snapshot (this morning) ===\n"
        f"{portfolio_blob}\n\n"
        f"Generate one PortfolioAdvice per holding above. "
        f"Total {len(holdings)} advice items expected."
    )

    log.info(
        "portfolio_advisor: calling %s for %d holdings",
        settings.openai_model_light, len(holdings),
    )

    try:
        response = await client.beta.chat.completions.parse(
            model=settings.openai_model_light,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format=PortfolioMorningAdvice,
            temperature=0.3,  # not creative — disciplined advice
        )
    except Exception as e:  # noqa: BLE001
        log.error("portfolio_advisor: LLM call failed: %s — fallback HOLD", e)
        return [
            PortfolioAdvice(
                asset=h.get("asset_label") or "?",
                action="HOLD",
                advice_short="Держи — анализ упал, проверь логи.",
            )
            for h in holdings
        ]

    await log_llm_usage("portfolio_advisor", response)
    parsed = response.choices[0].message.parsed
    if not parsed or not parsed.advice:
        return [
            PortfolioAdvice(
                asset=h.get("asset_label") or "?",
                action="HOLD",
                advice_short="Держи — LLM вернул пусто, проверь логи.",
            )
            for h in holdings
        ]

    # Make sure every holding is covered (LLM might forget one). Auto-fill HOLD.
    covered = {a.asset.upper() for a in parsed.advice}
    for h in holdings:
        label = (h.get("asset_label") or "").upper()
        if label and label not in covered:
            parsed.advice.append(
                PortfolioAdvice(
                    asset=h["asset_label"],
                    action="HOLD",
                    advice_short="Держи — без триггеров.",
                )
            )

    return parsed.advice
