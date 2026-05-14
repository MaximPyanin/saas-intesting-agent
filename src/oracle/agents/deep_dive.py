"""Deep-dive agent — invoked when Maksim taps 🔍 Go deeper on an idea card.

One focused LLM call (gpt-5.4-mini) that produces:
  - pricing_analysis: 2-3 sentences on price points the target customer pays today
  - cac_estimate: realistic CAC range + main acquisition channels with cost
  - competitor_landscape: top 3 actual competitors with what differentiates each
  - next_steps: 3 concrete actions for next 14 days
  - risk_callouts: 2-3 deal-breaking risks to validate FIRST

Cost: ~$0.005-0.015 per call. Designed to be cheap enough to call freely.

Gracefully no-ops without LLM credentials — returns None and the caller
falls back to the static placeholder.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ..config import get_settings

log = logging.getLogger(__name__)


class DeepDiveOutput(BaseModel):
    """Structured deep-dive analysis for one BusinessIdea."""

    pricing_analysis: str = Field(
        description=(
            "2-3 sentences in Russian. What do similar products charge TODAY? "
            "Cite real price points ($X/mo, $Y annual). If unknown for this "
            "exact niche, give analogous-niche benchmarks."
        )
    )
    cac_estimate: str = Field(
        description=(
            "2-3 sentences in Russian. Realistic CAC range for THIS target "
            "customer + the 1-2 cheapest acquisition channels with estimated "
            "cost-per-customer. Examples: 'Reddit cold posts CAC $5-15', "
            "'cold-LinkedIn outreach $30-80', 'ProductHunt launch $0 + 1 day'."
        )
    )
    competitor_landscape: str = Field(
        description=(
            "3-4 sentences in Russian listing top 3-4 actual competitors WITH "
            "their differentiator. Example: 'Notion AI — broad knowledge mgmt, "
            "weak vertical. Reflect — daily journal narrow but mature. None "
            "focused on Maksim's exact niche.'"
        )
    )
    next_steps: list[str] = Field(
        description=(
            "Exactly 3 concrete actions Maksim should do in next 14 days, "
            "in Russian. Each: 1 imperative sentence + a clear outcome. "
            "Examples: "
            "'Запостить waitlist-лендинг на Carrd за 2 часа, цель 30 имейлов "
            "до выходных.' "
            "'Сделать 10 cold-DM на r/[niche] с прямым вопросом про pain — "
            "минимум 5 ответов.' "
            "'Опубликовать demo-видео на ProductHunt в четверг 8am PT.'"
        ),
        min_length=3,
        max_length=3,
    )
    risk_callouts: list[str] = Field(
        description=(
            "2-3 deal-breaking risks Maksim must validate BEFORE building. "
            "Each 1 sentence in Russian, framed as a question or assumption "
            "to test. Examples: "
            "'Готовы ли клиенты платить $X/mo, а не ждать бесплатной альтернативы?' "
            "'Не уберёт ли OpenAI/Anthropic эту фичу в следующем релизе?'"
        ),
        min_length=2,
        max_length=3,
    )


SYSTEM_PROMPT = """\
You are ORACLE's deep-dive analyst for Maksim, a solo Python AI engineer
considering whether to build a specific business idea. Your job: produce
a SHARP, actionable analysis that helps him decide YES/NO in 60 seconds
of reading.

INPUT: one BusinessIdea (title, problem, solution, target_customer,
revenue_model, mvp_weeks, mvp_stack, unfair_advantage, signal_sources,
competitors, validation_notes).

OUTPUT (Russian, concrete, no fluff):
1. pricing_analysis — what similar products charge TODAY. Real numbers.
2. cac_estimate — realistic CAC + cheapest acquisition channels.
3. competitor_landscape — top 3-4 real competitors with their angle.
4. next_steps — exactly 3 actions for the next 14 days. Imperative,
   timeboxed, with measurable outcome.
5. risk_callouts — 2-3 deal-breaking risks to validate BEFORE building.

QUALITY RULES:
- Use REAL company/product names. Don't say 'similar SaaS tools' — say
  'Notion, Mem, Reflect'. Don't say 'social media ads' — say
  'r/indiehackers cold post, $50 Meta Ads to lookalike of Linear users'.
- ZERO generic advice. 'Talk to customers' is banned. Always give the
  channel + the cost + the expected outcome.
- If you don't know the exact niche pricing, give the closest analogous
  niche and SAY it's an analogy.
- Concise. Each field ≤ 3 sentences (lists ≤ 3 items, 1 sentence each).
- Russian throughout. Keep brand names in original (English).

Respond ONLY with valid JSON matching the schema.
"""


async def run_idea_deep_dive(idea_dict: dict[str, Any]) -> dict[str, Any] | None:
    """One LLM call — returns structured deep-dive or None on failure."""
    from ..observability import has_llm_credentials  # noqa: PLC0415
    if not has_llm_credentials():
        log.info("deep_dive: no LLM creds — skipping")
        return None

    try:
        from ..observability import get_openai_client, log_llm_usage  # noqa: PLC0415
        client = get_openai_client(agent="deep_dive")
    except Exception as e:  # noqa: BLE001
        log.warning("deep_dive: client init failed: %s", e)
        return None

    settings = get_settings()

    # Compact idea representation for the prompt
    user_msg = (
        f"=== IDEA TO DEEP-DIVE ===\n"
        f"Title: {idea_dict.get('title', '')}\n"
        f"Problem: {idea_dict.get('problem', '')}\n"
        f"Solution: {idea_dict.get('solution', '')}\n"
        f"Target customer: {idea_dict.get('target_customer', '')}\n"
        f"Revenue model: {idea_dict.get('revenue_model', '')}\n"
        f"MVP: {idea_dict.get('mvp_weeks', '?')} weeks, "
        f"stack={', '.join((idea_dict.get('mvp_stack') or [])[:6])}\n"
        f"Unfair advantage: {idea_dict.get('unfair_advantage', '')}\n"
        f"Critic-listed competitors: "
        f"{', '.join((idea_dict.get('competitors') or [])[:5]) or '(none)'}\n"
        f"Validator notes: {idea_dict.get('validation_notes', '')}\n\n"
        f"Produce the 5-section deep-dive analysis in JSON per the schema."
    )

    log.info("deep_dive: calling %s for '%s'",
             settings.openai_model_light, idea_dict.get("title", "?")[:60])

    try:
        response = await client.beta.chat.completions.parse(
            model=settings.openai_model_light,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format=DeepDiveOutput,
            temperature=0.4,  # consistent + grounded, not creative
        )
    except Exception as e:  # noqa: BLE001
        log.error("deep_dive: LLM call failed: %s", e)
        return None

    await log_llm_usage("deep_dive", response)
    parsed = response.choices[0].message.parsed
    if not parsed:
        return None

    return parsed.model_dump()
