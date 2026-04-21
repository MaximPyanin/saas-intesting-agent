"""Critic agent — Step 11 (heart of the Reflexion loop).

Third LLM-using agent. Reads `state["raw_ideas"]` (drafts from
idea_generator), assigns one of four verdicts to each, and lets the
Reflexion loop iterate until ideas are either approved or killed.

Verdict schema (per user spec):
  KILL          fatal flaw — dominant competitor / OpenAI will subsume /
                no first-customer path / technically infeasible
  WEAKEN        salvageable but has issues — idea_generator will rewrite
                it on the next round, addressing the critic's notes
  PASS          solid but not exceptional — accepted as-is
  STRONG_PASS   no dominant competitor in specific niche AND clear path to
                first 10 customers AND timing tied to recent signal AND
                Maksim's RAG/LLM skills give a real advantage

After critique:
  - KILLs are dropped permanently (never re-evaluated)
  - PASS / STRONG_PASS are kept untouched in raw_ideas
  - WEAKEN go back to idea_generator's improve mode on the next round
  - When max rounds OR no WEAKEN remain, surviving_ideas is set to the
    current PASS/STRONG_PASS pool and the loop exits to the validator

The router (`should_continue_reflexion` in nodes.py) decides when to exit.
This module handles the critique itself.

Cost: ~3-4k input tokens (system + ideas) + ~1-2k output (verdicts) per
call, ~$0.02-0.04. Up to 3 calls per Reflexion loop = ~$0.06-0.12 per run.

Per the user spec ("ruthless critic, 3 great beat 10 mediocre"), this
agent is intentionally aggressive — defaults toward KILL/WEAKEN unless
the idea is genuinely strong. Quality over quantity.

Gracefully no-ops without OPENAI_API_KEY: passes ALL ideas through with
a warning. Downstream nodes still run.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from ..config import get_settings
from ..state import OracleState
from .synthesizer import format_market_for_llm

log = logging.getLogger(__name__)


# ============================================================================
# Output schema
# ============================================================================

Verdict = Literal["KILL", "WEAKEN", "PASS", "STRONG_PASS"]


class CritiqueVerdict(BaseModel):
    """One verdict per input idea, identified by 1-based index."""

    index: int = Field(
        ge=1,
        description="1-based index matching the input idea list order",
    )
    verdict: Verdict
    notes: str = Field(
        max_length=400,
        description="Concise reasoning (≤400 chars). For WEAKEN, this is what "
                    "idea_generator will use to rewrite the idea next round.",
    )


class CriticOutput(BaseModel):
    verdicts: list[CritiqueVerdict] = Field(min_length=1)


# ============================================================================
# System prompt — verbatim from user spec, expanded
# ============================================================================


CRITIC_SYSTEM = """\
You are ORACLE's ruthless critic for Maksim, a Python AI engineer who can ship
solo MVPs in 2-6 weeks. Your job: KILL weak business ideas. Be specific — name
real competitors, name real failure modes.

For each idea, attack along these 5 axes:

1. COMPETITORS — "This already exists as [X.com] and [Y.com]." Try to recall
   actual products in the same niche. If a dominant direct competitor exists
   in the exact same niche with the exact same buyer, KILL.

2. CUSTOMER PATH — "Who exactly pays $X/month? Name 3 real companies."
   If Maksim cannot realistically reach the first 100 paying customers in
   3 months without burning paid-ads budget, KILL or WEAKEN.

3. OPENAI / ANTHROPIC RISK — "OpenAI or Anthropic will add this as a free
   feature in 6 months." If the idea is a thin LLM wrapper that a frontier
   lab could add to ChatGPT / Claude in a single feature drop, KILL.

4. TECHNICAL FEASIBILITY — "Can Maksim build this in 6 weeks solo with the
   listed stack?" If the MVP estimate is wildly optimistic (e.g. building a
   vector DB from scratch, training a custom LLM, complex distributed system),
   KILL or WEAKEN.

5. MOAT / DEFENSIBILITY — "What stops the next solo dev from cloning this in
   4 weeks?" If there's no defensible moat (proprietary data, deep workflow
   embedding, network effect, niche expertise), KILL.

VERDICTS (one per idea, MUST be one of these four):

  KILL          Fatal flaw above. No path to viability. Drop permanently.
  WEAKEN        Salvageable but has specific issues. The idea generator will
                rewrite this idea on the next round using your `notes` as the
                instruction. Be SPECIFIC in notes — name what to fix.
  PASS          Solid. Acceptable. Not exceptional but worth shipping.
  STRONG_PASS   No dominant direct competitor in the SPECIFIC niche AND clear
                path to first 10 customers AND timing tied to a recent signal
                AND Maksim's RAG/LangGraph/LLM skills give a real advantage.

Per the user's spec: "3 great ideas beat 10 mediocre ones." Be RUTHLESS.
Default toward KILL or WEAKEN unless the idea is genuinely strong. The
goal is to filter ~10 raw drafts down to ~3 strong survivors.

Notes field (≤400 chars): for KILL, explain WHY it dies (which competitor,
which OpenAI feature, which customer-path failure). For WEAKEN, give SPECIFIC
instructions for the rewriter ("Narrow target customer to dental clinics
only — generic 'SMBs' is too broad. Add a vertical-specific workflow as the
moat."). For PASS/STRONG_PASS, brief justification.

Index every verdict by the 1-based index of the idea in the input list.
Return one verdict PER input idea. Respond ONLY with valid JSON.
"""


# ============================================================================
# Idea formatting for LLM input
# ============================================================================


def format_ideas_for_critic(ideas: list[dict]) -> str:
    """Compress ideas into ~250 tokens each for the critic to evaluate."""
    lines: list[str] = []
    for i, idea in enumerate(ideas, start=1):
        title = idea.get("title", "(untitled)")
        conf = idea.get("confidence", 0)
        stage = idea.get("lifecycle_stage", "?")
        problem = (idea.get("problem") or "").strip()[:200]
        solution = (idea.get("solution") or "").strip()[:200]
        customer = (idea.get("target_customer") or "").strip()[:160]
        why_now = (idea.get("why_now") or "").strip()[:200]
        revenue = (idea.get("revenue_model") or "").strip()[:120]
        weeks = idea.get("mvp_weeks", "?")
        stack = ", ".join((idea.get("mvp_stack") or [])[:8])
        advantage = (idea.get("unfair_advantage") or "").strip()[:160]
        competitors = ", ".join((idea.get("competitors") or [])[:5]) or "(none listed)"
        sources = ", ".join((idea.get("signal_sources") or [])[:4]) or "(none)"

        lines.append(f"#{i} [{stage}, conf={conf}] {title}")
        lines.append(f"   Problem: {problem}")
        lines.append(f"   Solution: {solution}")
        lines.append(f"   Customer: {customer}")
        lines.append(f"   Why now: {why_now}")
        lines.append(f"   Revenue: {revenue}")
        lines.append(f"   MVP: {weeks}w, stack={stack}")
        lines.append(f"   Advantage: {advantage}")
        lines.append(f"   Competitors listed: {competitors}")
        lines.append(f"   Signal sources: {sources}")
    return "\n".join(lines)


# ============================================================================
# LLM call
# ============================================================================


async def critique_ideas(
    raw_ideas: list[dict],
    market_data: dict,
) -> list[CritiqueVerdict] | None:
    """Single OpenAI call. Returns None on error or no key."""
    settings = get_settings()
    from ..observability import has_llm_credentials  # noqa: PLC0415
    if not has_llm_credentials():
        log.info("critic: no LLM credentials — skipping (passing all ideas)")
        return None
    if not raw_ideas:
        return []

    try:
        from ..observability import get_openai_client, log_llm_usage  # noqa: PLC0415
    except ImportError:
        log.warning("critic: observability module unavailable")
        return None

    try:
        client = get_openai_client(agent="critic")
    except RuntimeError as e:
        log.error("critic: %s", e)
        return None

    market_line = format_market_for_llm(market_data)
    ideas_blob = format_ideas_for_critic(raw_ideas)
    user_msg = (
        f"{market_line}\n\n"
        f"=== {len(raw_ideas)} ideas to critique ===\n"
        f"{ideas_blob}\n\n"
        f"For EACH idea above, return ONE verdict (KILL / WEAKEN / PASS / "
        f"STRONG_PASS) with concise notes. Index by the 1-based number above. "
        f"Be ruthless — kill weak ideas, weaken salvageable ones, only "
        f"PASS/STRONG_PASS the genuinely strong."
    )

    log.info(
        "critic: calling %s with %d ideas (~%d KB user msg)",
        settings.openai_model_heavy,
        len(raw_ideas),
        len(user_msg) // 1024,
    )

    try:
        response = await client.beta.chat.completions.parse(
            model=settings.openai_model_heavy,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=CriticOutput,
            temperature=0.2,  # critical reasoning, low temperature
        )
    except Exception as e:  # noqa: BLE001
        log.error("critic: LLM call failed: %s", e)
        return None

    await log_llm_usage("critic", response)

    parsed = response.choices[0].message.parsed
    if not parsed:
        log.error("critic: LLM returned no parsed output")
        return None

    usage = response.usage
    if usage:
        log.info(
            "critic: %d verdicts · in=%d out=%d tokens",
            len(parsed.verdicts),
            usage.prompt_tokens,
            usage.completion_tokens,
        )

    return parsed.verdicts


# ============================================================================
# LangGraph node
# ============================================================================


async def critic_node(state: OracleState) -> dict:
    """Step 11 — replaces the Step 1 placeholder.

    Reads raw_ideas, critiques each, returns:
      - raw_ideas: PASS/STRONG_PASS + WEAKEN (KILLs dropped)
                   These feed back into idea_generator next round.
      - surviving_ideas: just PASS/STRONG_PASS (current best)
                         Read by validator after the loop exits.
      - reflexion_round: incremented

    Without OpenAI key, passes all ideas through as PASS so the graph
    still produces output (no LLM filtering).
    """
    round_num = state.get("reflexion_round", 0)
    raw_ideas = list(state.get("raw_ideas", []) or [])

    if not raw_ideas:
        log.info("critic round %d: no raw ideas — empty result", round_num)
        return {
            "raw_ideas": [],
            "surviving_ideas": [],
            "reflexion_round": round_num + 1,
        }

    market_data = state.get("market_data", {}) or {}
    verdicts = await critique_ideas(raw_ideas, market_data)

    # Graceful degradation: no LLM → pass all through unchanged
    if verdicts is None:
        log.info("critic round %d: graceful pass-through (%d ideas)", round_num, len(raw_ideas))
        for idea in raw_ideas:
            idea["verdict"] = idea.get("verdict") or "PASS"
        return {
            "raw_ideas": raw_ideas,
            "surviving_ideas": raw_ideas,  # all considered surviving
            "reflexion_round": round_num + 1,
        }

    # Map verdicts back to ideas by 1-based index
    verdict_by_index: dict[int, CritiqueVerdict] = {v.index: v for v in verdicts}
    for i, idea in enumerate(raw_ideas, start=1):
        v = verdict_by_index.get(i)
        if v:
            idea["verdict"] = v.verdict
            idea["critic_notes"] = v.notes
            idea["reflexion_rounds_passed"] = round_num + 1
        else:
            # No verdict for this idea — default to WEAKEN so it gets reviewed
            idea["verdict"] = "WEAKEN"
            idea["critic_notes"] = "(critic did not return a verdict for this idea)"

    killed = [i for i in raw_ideas if i.get("verdict") == "KILL"]
    weakened = [i for i in raw_ideas if i.get("verdict") == "WEAKEN"]
    passed = [i for i in raw_ideas if i.get("verdict") in ("PASS", "STRONG_PASS")]
    strong = [i for i in raw_ideas if i.get("verdict") == "STRONG_PASS"]

    log.info(
        "critic round %d: %d KILL · %d WEAKEN · %d PASS (%d STRONG)",
        round_num, len(killed), len(weakened), len(passed), len(strong),
    )

    # Compact log of verdicts
    for i, idea in enumerate(raw_ideas[:5], start=1):
        log.info("  #%d %s — %s — %s",
                 i, idea.get("verdict"),
                 (idea.get("title") or "")[:50],
                 (idea.get("critic_notes") or "")[:80])

    # Drop KILLs immediately. Survivors for next round = PASS + WEAKEN.
    next_raw = passed + weakened

    return {
        "raw_ideas": next_raw,            # for next idea_generator iteration
        "surviving_ideas": passed,        # current best (read by validator on exit)
        "reflexion_round": round_num + 1,
    }


# ============================================================================
# Standalone CLI: `uv run python -m oracle.agents.critic`
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
        # Hardcoded sample ideas to test the critic in isolation
        sample_ideas: list[dict] = [
            {
                "title": "ChatGPT for Lawyers",
                "one_liner": "AI legal assistant for solo lawyers",
                "problem": "Solo lawyers spend hours drafting routine contracts",
                "solution": "Upload context, get a draft contract in seconds",
                "target_customer": "Solo lawyers and small firms",
                "why_now": "GPT-4 is good enough at legal text",
                "revenue_model": "$99/mo per seat",
                "mvp_weeks": 6,
                "mvp_stack": ["Python", "FastAPI", "OpenAI"],
                "confidence": 60,
                "lifecycle_stage": "PEAK",
                "window_months": 6,
                "competitors": ["Harvey", "Clio Duo", "LawDroid"],
                "unfair_advantage": "RAG over case law",
                "signal_sources": ["hn", "techcrunch"],
            },
            {
                "title": "Self-hosted LangGraph Observability Dashboard",
                "one_liner": "Drop-in Python decorator that traces every node + cost in production agents",
                "problem": "Teams ship LangGraph agents to prod but have zero visibility into per-node accuracy/latency/cost",
                "solution": "@oracle_trace decorator → DuckDB-backed dashboard, no cloud dependency",
                "target_customer": "AI engineers spending $1k-100k/mo on OpenAI, found on r/MachineLearning and Twitter",
                "why_now": "LangGraph 1.x just hit GA in March 2026, agent stacks are hitting prod at scale, Langfuse pushes cloud-only",
                "revenue_model": "OSS core free + $99/mo for hosted dashboard with team features",
                "mvp_weeks": 5,
                "mvp_stack": ["Python", "FastAPI", "DuckDB", "OpenTelemetry", "Next.js"],
                "confidence": 82,
                "lifecycle_stage": "GROWING",
                "window_months": 10,
                "competitors": ["Langfuse (cloud-first)", "Helicone", "Arize Phoenix"],
                "unfair_advantage": "Self-hosted-first + LangGraph-native trace adapters Maksim wrote",
                "signal_sources": ["hn", "reddit/r/LocalLLaMA", "github/python"],
            },
        ]
        fake_state: OracleState = {
            "scout_signals": [], "market_data": {}, "trend_signals": [],
            "custom_signals": [], "synthesized": [], "raw_ideas": sample_ideas,
            "investment_scenarios": [], "reflexion_round": 0,
            "surviving_ideas": [], "validated": [], "final_digest": {}, "errors": [],
        }
        result = await critic_node(fake_state)
        print()
        print("=" * 70)
        print(f"Critic verdicts:")
        print("=" * 70)
        for i, idea in enumerate(sample_ideas, start=1):
            print()
            print(f"#{i} [{idea.get('verdict', '?')}] {idea['title']}")
            print(f"   notes: {idea.get('critic_notes', '(no notes)')[:200]}")
        print()
        print(f"Surviving (PASS/STRONG): {len(result.get('surviving_ideas', []))}")
        print(f"For next round (PASS+WEAKEN): {len(result.get('raw_ideas', []))}")

    asyncio.run(_main())
