"""Idea generator agent — Step 10.

Second LLM-using agent. Reads `state["synthesized"]` (insight clusters from
Step 9), filters to business_idea clusters, and asks gpt-4o to draft 5-10
specific business ideas in the BusinessIdea schema (same one used by the
Telegram cards in Step 3).

The drafts then go into the Reflexion loop (Step 11 critic) which kills
weak ones and asks idea_generator to improve any flagged WEAKEN. Step 10
implements only the FRESH generation path; the round-aware "improve"
branch lights up in Step 11 once critic sets real verdicts.

Per the user's repeated 70/30 priority — this agent is the heart of
ORACLE's primary value: spotting SaaS / AI / agentic ideas before the
hype peaks. Prompt is heavily tuned for Maksim's profile (Python AI eng,
solo MVPs in 2-6 weeks, RAG/LLM expertise as competitive edge).

Cost: ~5-10 ideas * gpt-4o = ~$0.03-0.05 per call. Up to 3 calls per
Reflexion loop = ~$0.10-0.15 per run.

Gracefully no-ops without OPENAI_API_KEY — returns empty raw_ideas list,
graph still runs.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from ..config import get_settings
from ..models import BusinessIdea
from ..state import OracleState
from .synthesizer import format_market_for_llm

log = logging.getLogger(__name__)


# ============================================================================
# Output schema — wrapper around BusinessIdea for OpenAI structured output
# ============================================================================


class IdeasOutput(BaseModel):
    """LLM response: a list of 5-10 raw business ideas, sorted by confidence."""

    ideas: list[BusinessIdea] = Field(
        min_length=1,
        max_length=10,
        description="5-10 business ideas, sorted by confidence DESC",
    )


# ============================================================================
# System prompt — heavily tuned for Maksim's profile
# ============================================================================


IDEA_GEN_SYSTEM = """\
You are ORACLE's business idea generator for Maksim.

USER PROFILE (critical — every idea must fit):
- Python AI engineer, ships solo MVPs in 2-6 weeks
- Stack: Python, FastAPI, LangGraph, RAG, LLMs, Docker, Azure
- Sweet spot: AI-native tools where LLM/RAG sits AT THE CORE (not bolt-on)
- This is Maksim's competitive moat — RAG/agentic depth that competitors lack
- MVP budget: under $5,000, solo buildable, time to first paying customer < 3 months
- Available after 15:00 Warsaw local

AVOID at all costs (these are dealbreakers):
- Physical products of any kind
- Enterprise sales cycles longer than 3 months
- Mobile-only apps (he's a backend/web specialist)
- Heavily regulated verticals (medical, legal, finance compliance)
- Pure thin LLM wrappers with no defensible moat
- Hype-cycle generic AI features ("ChatGPT for X")

YOUR INPUT: 5-10 cross-signal insight clusters (category=business_idea) from
the synthesizer. Each cluster represents a real cross-signal pattern from
this run's collected signals (HN, Reddit r/SaaS / r/indiehackers /
r/MachineLearning, GitHub trending, ProductHunt, VC deal flow, news RSS).

YOUR JOB: produce 5-10 SPECIFIC business idea drafts. Each idea will then go
through 3 rounds of ruthless critique. Quality over quantity — better to
produce 5 strong drafts than 10 mediocre ones.

QUALITY RULES (non-negotiable):
1. Each idea MUST be grounded in one or more input clusters. Cite their
   source IDs in `signal_sources`.
2. `why_now` MUST explain SPECIFIC market timing — what changed in the past
   weeks/months that makes this possible NOW (and was NOT possible 6 months
   ago). Reference the cluster signals.
3. `target_customer` must be CONCRETE — name a job title, company size band,
   and where to find them. Not "businesses" or "developers" — say "AI
   engineers shipping production agents at $1k-100k/mo OpenAI spend on
   Twitter and r/MachineLearning".
4. `unfair_advantage` must explain Maksim's specific edge for THIS idea. RAG
   expertise, LangGraph fluency, niche understanding, etc. If you can't
   articulate the moat, the idea is weak.
5. `mvp_weeks` must be 1-12. Aim for 4-6. Reject mentally any idea that
   would actually take 6+ months for a solo dev.
6. `mvp_stack` should reuse Maksim's stack: Python, FastAPI, LangGraph,
   pandas, Postgres/SQLite, Docker. Add ML/RAG pieces as needed.
7. `revenue_model` must be specific — name a price point and tier. Not
   "subscription" — say "$29/mo per seat with 5-file free tier".
8. `lifecycle_stage`: copy from the cluster (EMERGING / GROWING / PEAK /
   DECLINING). Prefer EMERGING/GROWING — that's where Maksim has time.
9. `confidence` 0-100: how strong the underlying signals are.
10. `competitors`: list 2-5 actual competitor names (not "many companies").
11. `verdict`: leave as "PASS" — the critic in the next stage will set this.
12. `reflexion_rounds_passed`: leave as 0 — handled downstream.
13. `signal_sources`: 2-5 short labels naming the contributing source IDs.
14. `similar_projects`: 2-4 CLOSEST existing projects/products Maksim should
    study. Format EACH entry as "Name — 1-line what-they-do (url if known)".
    NOT the same as competitors: these are reference points for pricing/UX/
    feature-gap study. Example: "Langfuse — open-source LLM observability
    (langfuse.com)". If genuinely none exist, return an empty list.
15. `estimated_cost_usd`: MVP build + first-2-month run cost as a short
    string. Break it down. Example: "$300-600 (infra $40/mo + OpenAI API
    ~$80/mo + landing page $0 Carrd + domain $12/yr)". Be honest — include
    API costs at realistic usage.
16. `launch_steps`: 4-7 ordered CONCRETE steps from zero to paying users.
    Each step is 1 line, imperative, with a timebox. Must cover:
      (a) VALIDATE BEFORE BUILDING — waitlist landing + small paid ad test
      (b) build MVP (core feature only)
      (c) first-cohort recruitment (how to get 10 users)
      (d) monetization trigger (when/how first paid tier is turned on)
      (e) launch event (ProductHunt/Show HN/Reddit post with specific sub)
    Example item: "Week 1: Carrd landing + $50 Meta Ads to r/indiehackers
    lookalike — goal 50 waitlist signups before writing a line of code".
17. `marketing_channels`: 2-5 SPECIFIC channels where THIS target_customer
    actually hangs out. Be specific: subreddit names, exact Slack/Discord
    communities, Twitter cohorts, conferences/meetups. Example:
    "r/LocalLLaMA", "Indie Hackers Slack #stuck", "AI Engineer meetup
    Warsaw", "Hacker News Show HN Tuesday 8am PT". Not "social media".

OUTPUT: 5-10 ideas, sorted by confidence DESC. Respond ONLY with valid JSON
matching the schema.
"""


# ============================================================================
# Cluster formatting for the LLM input
# ============================================================================


def format_clusters_for_llm(clusters: list[dict]) -> str:
    """Compress synthesized clusters into a token-efficient block."""
    lines: list[str] = []
    for i, c in enumerate(clusters, start=1):
        stage = c.get("lifecycle_stage", "UNKNOWN")
        conf = c.get("confidence", 0)
        name = c.get("display_name", c.get("topic", "?"))
        story = (c.get("story") or "").replace("\n", " ").strip()
        sources = ", ".join((c.get("related_signal_sources") or [])[:5])
        signals_titles = (c.get("related_signal_titles") or [])[:5]

        lines.append(f"#{i} [{stage}, conf={conf}] {name}")
        lines.append(f"   Story: {story}")
        if sources:
            lines.append(f"   Sources: {sources}")
        if signals_titles:
            lines.append(f"   Signals: {' | '.join(t[:60] for t in signals_titles)}")
    return "\n".join(lines)


# ============================================================================
# LLM call — fresh generation
# ============================================================================


async def _build_system_prompt() -> str:
    """Compose the idea_generator system prompt with the current learned
    preferences (Step 14) appended at the bottom.

    Falls back to the default prompt if no calibration has run yet.
    """
    base = IDEA_GEN_SYSTEM
    try:
        from ..learning import get_prompt_injection_ideas  # noqa: PLC0415 — lazy
        injection = await get_prompt_injection_ideas()
    except Exception as e:  # noqa: BLE001
        log.debug("idea_generator: could not read learning weights: %s", e)
        injection = ""

    if injection:
        return base + (
            "\n\n----- LEARNED PREFERENCES (from feedback calibration, "
            "Step 14) -----\n"
            f"{injection}\n"
            "Weight these strongly — they reflect Maksim's actual past "
            "behavior on idea cards. Honor them when generating new ideas."
        )
    return base


async def generate_business_ideas(
    clusters: list[dict],
    market_data: dict,
) -> list[BusinessIdea]:
    """Single OpenAI call. Returns empty list on error."""
    settings = get_settings()
    from ..observability import has_llm_credentials  # noqa: PLC0415
    if not has_llm_credentials():
        log.info("idea_generator: no LLM credentials — empty result")
        return []

    if not clusters:
        log.info("idea_generator: no business_idea clusters — empty result")
        return []

    try:
        from ..observability import get_openai_client, log_llm_usage  # noqa: PLC0415
    except ImportError:
        log.warning("idea_generator: observability module unavailable")
        return []

    try:
        client = get_openai_client(agent="idea_generator")
    except RuntimeError as e:
        log.error("idea_generator: %s", e)
        return []

    cluster_blob = format_clusters_for_llm(clusters)
    market_line = format_market_for_llm(market_data)
    user_msg = (
        f"{market_line}\n\n"
        f"=== {len(clusters)} business idea clusters from this run ===\n"
        f"{cluster_blob}\n\n"
        f"Generate 5-10 specific BusinessIdea drafts grounded in these clusters."
    )

    system_prompt = await _build_system_prompt()
    log.info(
        "idea_generator: calling %s with %d clusters (~%d KB user msg, "
        "%s learned prefs)",
        settings.openai_model_heavy,
        len(clusters),
        len(user_msg) // 1024,
        "with" if len(system_prompt) > len(IDEA_GEN_SYSTEM) else "no",
    )

    try:
        response = await client.beta.chat.completions.parse(
            model=settings.openai_model_heavy,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            response_format=IdeasOutput,
            temperature=0.7,  # creative — ideation needs imagination
        )
    except Exception as e:  # noqa: BLE001
        log.error("idea_generator: LLM call failed: %s", e)
        return []

    await log_llm_usage("idea_generator", response)

    parsed = response.choices[0].message.parsed
    if not parsed:
        log.error("idea_generator: LLM returned no parsed output")
        return []

    usage = response.usage
    if usage:
        log.info(
            "idea_generator: %d ideas · in=%d out=%d tokens",
            len(parsed.ideas),
            usage.prompt_tokens,
            usage.completion_tokens,
        )

    return parsed.ideas


# ============================================================================
# Improve mode (Step 11) — rewrite WEAKEN ideas using critic notes
# ============================================================================


IDEA_IMPROVE_SYSTEM = """\
You are ORACLE's idea generator in IMPROVE MODE.

The critic flagged the ideas below as WEAKEN — salvageable, but with specific
issues. Each idea has `critic_notes` explaining what is wrong.

YOUR JOB: rewrite each idea to FIX the critic's concerns while preserving the
core insight. The improved versions go BACK through the critic next round.

RULES (one rewrite per input idea, in the same order):
1. KEEP the core idea but fix the SPECIFIC concerns in critic_notes.
2. If critic flagged competitors → narrow to a more defensible niche or pivot
   the angle (different buyer, different workflow integration).
3. If critic flagged customer path → get MORE SPECIFIC. Name the exact
   buyer (job title, company size band) and where to reach them.
4. If critic flagged OpenAI/Anthropic risk → add a defensible moat:
   proprietary data, deep workflow embedding, network effect, or vertical
   knowledge that ChatGPT can't trivially replicate.
5. If critic flagged technical feasibility → simplify the MVP scope to
   something Maksim can actually ship in ≤6 weeks solo.
6. If critic flagged moat → add a structural defensibility (data network,
   workflow lock-in, vertical depth).
7. Update the affected fields: title, problem, solution, target_customer,
   why_now, revenue_model, unfair_advantage, mvp_stack, mvp_weeks, competitors.
8. Reset `verdict` to "PASS" — the critic will re-evaluate next round.
9. Reset `critic_notes` to "" — the next round will fill it fresh.
10. Keep `lifecycle_stage` and `signal_sources` from the original.
11. ALSO update the practical-execution fields if the pivot changes them:
    - `similar_projects`: 2-4 closest products ("Name — what-they-do (url)")
    - `estimated_cost_usd`: MVP cost breakdown string
    - `launch_steps`: 4-7 ordered steps (validate → build → recruit → monetize → launch)
    - `marketing_channels`: 2-5 specific places the new target_customer actually hangs out
    If the pivot didn't change the buyer, you may keep them — but NEVER leave
    them empty. They are what Maksim acts on.

Maksim's profile (unchanged from fresh-generation mode):
- Python AI engineer, ships solo MVPs in 2-6 weeks
- Stack: Python, FastAPI, LangGraph, RAG, LLMs, Docker, Azure
- AI-native tools with LLM/RAG core are his sweet spot
- Avoid: physical products, mobile-only, regulated verticals, thin LLM wrappers

Output: same number of improved ideas as input, in the same order.
Respond ONLY with valid JSON matching the schema.
"""


def _format_weakened_for_improve(ideas: list[dict]) -> str:
    """Format WEAKEN ideas + their critic notes for the rewriter."""
    lines: list[str] = []
    for i, idea in enumerate(ideas, start=1):
        title = idea.get("title", "(untitled)")
        problem = (idea.get("problem") or "").strip()[:200]
        solution = (idea.get("solution") or "").strip()[:200]
        customer = (idea.get("target_customer") or "").strip()[:160]
        why_now = (idea.get("why_now") or "").strip()[:200]
        revenue = (idea.get("revenue_model") or "").strip()[:120]
        weeks = idea.get("mvp_weeks", "?")
        stack = ", ".join((idea.get("mvp_stack") or [])[:8])
        advantage = (idea.get("unfair_advantage") or "").strip()[:160]
        competitors = ", ".join((idea.get("competitors") or [])[:5]) or "(none)"
        lifecycle = idea.get("lifecycle_stage", "?")
        critic_notes = (idea.get("critic_notes") or "").strip()

        lines.append(f"#{i} [{lifecycle}] {title}")
        lines.append(f"   Problem: {problem}")
        lines.append(f"   Solution: {solution}")
        lines.append(f"   Customer: {customer}")
        lines.append(f"   Why now: {why_now}")
        lines.append(f"   Revenue: {revenue}")
        lines.append(f"   MVP: {weeks}w, stack={stack}")
        lines.append(f"   Advantage: {advantage}")
        lines.append(f"   Competitors: {competitors}")
        lines.append(f"   >>> CRITIC NOTES: {critic_notes}")
    return "\n".join(lines)


async def improve_business_ideas(weakened: list[dict]) -> list[BusinessIdea]:
    """Rewrite WEAKEN ideas based on critic notes. Returns same count, same order."""
    settings = get_settings()
    from ..observability import has_llm_credentials  # noqa: PLC0415
    if not has_llm_credentials():
        log.info("idea_generator: no LLM credentials — improve mode skipped")
        return []
    if not weakened:
        return []

    try:
        from ..observability import get_openai_client, log_llm_usage  # noqa: PLC0415
    except ImportError:
        log.warning("idea_generator: observability module unavailable")
        return []

    try:
        client = get_openai_client(agent="idea_generator_improve")
    except RuntimeError as e:
        log.error("idea_generator: %s", e)
        return []

    blob = _format_weakened_for_improve(weakened)
    user_msg = (
        f"=== {len(weakened)} WEAKEN ideas to rewrite ===\n"
        f"{blob}\n\n"
        f"Rewrite EACH idea above to fix the >>> CRITIC NOTES <<<. Return "
        f"the same number of ideas in the same order."
    )

    # Append learned preferences to the improve prompt too — refines should
    # honor the same calibrated preferences as fresh generation.
    improve_system = IDEA_IMPROVE_SYSTEM
    try:
        from ..learning import get_prompt_injection_ideas  # noqa: PLC0415
        injection = await get_prompt_injection_ideas()
        if injection:
            improve_system += (
                "\n\n----- LEARNED PREFERENCES (Step 14) -----\n"
                f"{injection}\n"
                "Honor these when rewriting weakened ideas."
            )
    except Exception:  # noqa: BLE001
        pass

    log.info(
        "idea_generator: improve mode — calling %s on %d weakened ideas",
        settings.openai_model_heavy, len(weakened),
    )

    try:
        response = await client.beta.chat.completions.parse(
            model=settings.openai_model_heavy,
            messages=[
                {"role": "system", "content": improve_system},
                {"role": "user", "content": user_msg},
            ],
            response_format=IdeasOutput,
            temperature=0.6,  # slightly less creative — we're refining, not inventing
        )
    except Exception as e:  # noqa: BLE001
        log.error("idea_generator: improve LLM call failed: %s", e)
        return []

    await log_llm_usage("idea_generator_improve", response)

    parsed = response.choices[0].message.parsed
    if not parsed:
        return []

    usage = response.usage
    if usage:
        log.info(
            "idea_generator: improve %d → %d ideas · in=%d out=%d tokens",
            len(weakened), len(parsed.ideas),
            usage.prompt_tokens, usage.completion_tokens,
        )

    return parsed.ideas


# ============================================================================
# LangGraph node
# ============================================================================


async def idea_generator_node(state: OracleState) -> dict:
    """Step 10/11 — fresh generation on round 0, improve mode on round 1+.

    Round 0:
      - Reads synthesized business_idea clusters
      - Generates 5-10 fresh BusinessIdea drafts via gpt-4o
      - Returns raw_ideas with verdict='PASS' (critic will re-evaluate)

    Round 1+:
      - Reads raw_ideas which now contain critic verdicts
      - Splits into PASS (kept untouched) + WEAKEN (rewritten)
      - Returns raw_ideas = passed + improved
      - The improved ideas have verdict='PASS' so the critic re-evaluates

    Returns empty list gracefully if no synthesized clusters or no OpenAI key.
    """
    round_num = state.get("reflexion_round", 0)
    synthesized = state.get("synthesized", []) or []

    business_clusters = [
        c for c in synthesized if c.get("category") == "business_idea"
    ]

    log.info(
        "idea_generator: round %d · %d/%d clusters are business_idea",
        round_num, len(business_clusters), len(synthesized),
    )

    # ---------- Round 0: fresh generation ----------
    if round_num == 0:
        ideas = await generate_business_ideas(
            business_clusters, state.get("market_data", {}) or {}
        )
        for idea in ideas:
            idea.verdict = "PASS"
            idea.reflexion_rounds_passed = 1
            idea.critic_notes = ""

        for i, idea in enumerate(ideas[:5], start=1):
            log.info(
                "  idea #%d [%s, conf=%d, %dw] %s",
                i, idea.lifecycle_stage, idea.confidence, idea.mvp_weeks, idea.title,
            )
        return {"raw_ideas": [i.model_dump() for i in ideas]}

    # ---------- Round 1+: improve mode ----------
    raw_ideas = state.get("raw_ideas", []) or []
    weakened = [i for i in raw_ideas if i.get("verdict") == "WEAKEN"]
    passed = [i for i in raw_ideas if i.get("verdict") in ("PASS", "STRONG_PASS")]

    if not weakened:
        log.info(
            "idea_generator: round %d — no WEAKEN ideas (%d PASS), pass-through",
            round_num, len(passed),
        )
        return {"raw_ideas": passed}

    log.info(
        "idea_generator: round %d — improving %d WEAKEN ideas (keeping %d PASS untouched)",
        round_num, len(weakened), len(passed),
    )

    improved = await improve_business_ideas(weakened)
    for idea in improved:
        idea.verdict = "PASS"
        idea.critic_notes = ""
        idea.reflexion_rounds_passed = round_num + 1

    if not improved:
        # Improvement failed — keep weakened as-is so loop can exit naturally
        log.warning("idea_generator: improve returned empty, keeping originals")
        return {"raw_ideas": passed + weakened}

    return {"raw_ideas": passed + [i.model_dump() for i in improved]}


# ============================================================================
# Standalone CLI: `uv run python -m oracle.agents.idea_generator`
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
        # For CLI testing without running the full graph: take a hardcoded
        # sample cluster (representing what synthesizer would emit)
        sample_clusters = [
            {
                "topic": "agentic-rag-production",
                "display_name": "Agentic RAG hitting production",
                "story": (
                    "Three independent signals converge: HN front page on agentic "
                    "RAG patterns, r/MachineLearning + r/LocalLLaMA discussions on "
                    "production deployment pain, and 5 GitHub trending repos for "
                    "LangGraph extensions this week."
                ),
                "lifecycle_stage": "GROWING",
                "confidence": 85,
                "related_signal_titles": [
                    "How to make agentic RAG actually work in production",
                    "Show HN: Self-hosted agent observability",
                    "trending: NousResearch/hermes-agent",
                ],
                "related_signal_sources": [
                    "hn", "reddit/r/MachineLearning", "github/python",
                ],
                "category": "business_idea",
            },
            {
                "topic": "voice-vertical-saas",
                "display_name": "Voice agents for vertical SMBs",
                "story": (
                    "Vapi+ElevenLabs Turbo dropped real-time voice cost to $0.10/min "
                    "this quarter. r/SaaS has 4 posts this week from solo founders "
                    "discussing vertical voice agents (dental, vet, plumbing). a16z "
                    "funded a16z-portfolio voice startup."
                ),
                "lifecycle_stage": "EMERGING",
                "confidence": 78,
                "related_signal_titles": [
                    "I built a voice agent for dental clinics, here's what I learned",
                    "Vapi pricing dropped, finally viable for SMB",
                    "a16z leads $5M in voice agent vertical startup",
                ],
                "related_signal_sources": [
                    "reddit/r/SaaS", "hn", "rss/techcrunch",
                ],
                "category": "business_idea",
            },
        ]

        fake_state: OracleState = {
            "scout_signals": [], "market_data": {}, "trend_signals": [],
            "custom_signals": [], "synthesized": sample_clusters,
            "raw_ideas": [], "investment_scenarios": [], "reflexion_round": 0,
            "surviving_ideas": [], "validated": [], "final_digest": {}, "errors": [],
        }
        result = await idea_generator_node(fake_state)
        ideas = result.get("raw_ideas", [])
        print()
        print("=" * 70)
        print(f"Idea generator output: {len(ideas)} ideas")
        print("=" * 70)
        for i, idea in enumerate(ideas, start=1):
            print()
            print(f"#{i} [{idea['lifecycle_stage']}, conf={idea['confidence']}, {idea['mvp_weeks']}w]")
            print(f"   Title: {idea['title']}")
            print(f"   One-liner: {idea['one_liner']}")
            print(f"   Why now: {idea['why_now']}")
            print(f"   Customer: {idea['target_customer']}")
            print(f"   Revenue: {idea['revenue_model']}")
            print(f"   Stack: {', '.join(idea['mvp_stack'])}")
            print(f"   Advantage: {idea['unfair_advantage']}")
            print(f"   Competitors: {', '.join(idea['competitors']) or '(none)'}")

    asyncio.run(_main())
