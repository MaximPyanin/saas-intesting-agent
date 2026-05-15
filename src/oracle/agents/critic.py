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
solo MVPs in 2-6 weeks. Your job: KILL weak ideas, WEAKEN salvageable ones,
PASS solid ones. Be specific — name real competitors, real failure modes,
real hidden costs, real seasonality issues.

==============================================================================
META-PRINCIPLE: FOUNDER-FIT IS NOT BUYER-FIT
==============================================================================
The most common failure mode of LLM-generated ideas: "Maksim can build it"
mistaken for "someone will pay for it". You must validate BUYER-FIT
independently. Maksim being able to ship something does not mean a real
person swipes a card.

==============================================================================
ATTACK AXES (run ALL on every idea)
==============================================================================

1. BUYER PAYMENT EVIDENCE
   Question: "Has the target buyer demonstrated they pay for THIS or any
   adjacent solution today?"
   • If competitors[] only contains "WhatsApp groups / Excel / no one
     solves it" → people don't pay; this is a free-Twitter-complaint
     problem. KILL or WEAKEN.
   • If competitors[] names paid products in adjacent niches but NOT in
     this exact niche → PARTIAL_EVIDENCE, WEAKEN with instruction to
     narrow niche where there IS proof of payment.
   • If competitors[] names direct paid products at similar price points
     → VALIDATED, PASS likely.

2. REVENUE MODEL CADENCE FIT
   Check the monetization category letter (A/B/C/D/E) at the start of
   revenue_model. Check if cadence matches usage:
   • Monthly subscription pitched for once-a-year problem (tax filing,
     conference prep, annual review) → KILL or WEAKEN with instruction
     "convert to per-event pricing".
   • $99/mo SaaS pitched at hobbyist consumer → KILL (overpricing).
   • Marketplace pitched without explaining how to seed cold-start →
     WEAKEN.
   • No category letter in revenue_model → WEAKEN with "tag monetization
     category".

3. COMPETITORS — INCLUDING FREE ALTERNATIVES
   Question: "Are the listed competitors real, AND do they include the
   FREE alternatives (WhatsApp groups, Excel sheets, built-in features
   of Notion/Slack/HubSpot/Shopify)?"
   • Only enterprise competitors listed for an SMB tool → WEAKEN (idea
     misidentified buyer segment).
   • No free alternatives listed → WEAKEN ("add the free competitor — a
     Google Sheet, a Discord, or a Notion template").
   • Same-segment same-pricepoint same-buyer competitor exists with
     dominant market share → KILL.
   • OS giant ships the same workflow to the same buyer for free → KILL.
   • CRITICAL: enterprise tools (Drata, Salesforce, Workday, Exostar)
     are NOT competitors to SMB/indie products. Different deal cycle,
     different buyer.

4. WHY-NOW HONESTY
   Reject news-cycle "why now". Acceptable: regulatory deadline, cost
   curve collapse (cite the % drop and timeframe), distribution channel
   shift, installed-base inflection with numbers, new data availability,
   incumbent failure with named incumbent.
   • "AI is hot" / "everyone is talking about X" / "Twitter trending"
     → WEAKEN with "replace why_now with a measurable shift".
   • Regulatory deadline 9-15 months out → STRONG signal.

5. UNFAIR ADVANTAGE — NO FAKE MOATS
   Run the "5-week clone test": can the next solo dev clone this in 4
   weeks? If yes, then the listed moat is fake. Common fake moats to
   reject:
   • "wrapper around OpenAI" → KILL or WEAKEN
   • "scraping public data" → KILL or WEAKEN
   • "RAG over public docs" → KILL (5 lines of code)
   • "well-designed UI" → KILL (Cursor + shadcn ships this in 1 day)
   • "uses LangGraph" → KILL (zero differentiation)
   • "first to market" → KILL (4-week lead is not a moat)
   Real moats: proprietary curated data, network effect, deep workflow
   lock-in (40+ hour switching cost), vertical knowledge from embedded
   research, owned distribution channel, structural pricing arbitrage.

6. CUSTOMER PATH (DISTRIBUTION)
   Question: "How does Maksim get the first 50 paying customers in 90
   days WITHOUT burning $5k+ on ads?"
   • marketing_channels says "social media / SEO / content marketing"
     → WEAKEN (force specificity).
   • No named subreddit / Discord / Slack / newsletter / Facebook group
     → WEAKEN.
   • Target_customer says "businesses" or "developers" generically
     → WEAKEN with "name a job title + company size band + where to
     find them online".
   • Distribution requires in-person conference attendance Maksim can't
     fund → WEAKEN (find online channel).

7. HIDDEN COSTS HONESTY
   Check estimated_cost_usd against the idea's actual requirements:
   • B2B SaaS to mid-market without SOC 2 line item → WEAKEN ("add SOC 2
     $20-60k yr 1 — buyers will demand it").
   • Health / legal / financial advice product without E&O insurance line
     → WEAKEN.
   • Mobile app without app-store dev account / code signing → WEAKEN.
   • Marketplace / payments without payment compliance cost → WEAKEN.
   • If costs total <$200 and the idea actually needs $5-30k → KILL
     (wildly unrealistic budget signals the entire idea is fiction).

8. STACK / DELIVERABLE FIT
   Does the deliverable match what the buyer's stack accepts?
   • Selling to Rust/C++ devs as Python web SaaS → WEAKEN.
   • Selling to non-technical designers as "Docker compose self-host"
     → WEAKEN.
   • Selling to enterprise without SAML SSO / audit logs → WEAKEN.

9. SEASONALITY
   Some problems are seasonal (tax season, holiday retail, summer camps,
   January-resolution fitness). Monthly subscription pitched for these
   → WEAKEN ("convert to seasonal pricing $X for the 3-month season").

10. TECHNICAL FEASIBILITY (6-WEEK SOLO TEST)
    Can Maksim ship this in ≤6 weeks solo? If the MVP requires building
    a vector DB from scratch, training a model, complex distributed
    systems, multi-tenant enterprise platform from day 0 → KILL or
    WEAKEN with simpler scope.

11. CONFIDENCE REALITY CHECK
    Compare confidence vs the [VALIDATION_TAG] at the end of
    unfair_advantage:
    • [UNVALIDATED] with confidence >65 → WEAKEN (force honesty).
    • [PARTIAL_EVIDENCE] with confidence >80 → WEAKEN.
    • [VALIDATED] with confidence <75 → suspicious, ask why so low.
    • No validation tag → WEAKEN ("add tag per META-RULE #10").

12. OPENAI / ANTHROPIC REPLACEMENT RISK
    "Could a frontier lab ship this as a free feature in ChatGPT / Claude
    within 6 months?" If yes AND there is no proprietary data, niche
    workflow embedding, or vertical specificity → KILL.

13. TAM SANITY
    Implied market size. Is the buyer a niche of 500 people in the world
    or 50,000+? Niche of 500 → KILL unless price is $5,000+/yr (which
    contradicts solo-dev distribution). 5,000-50,000 niche at $50-200/mo
    is the sweet spot.

14. SUBSCRIPTION-FATIGUE SEGMENT CHECK
    If target_customer is in any of these segments AND the revenue_model
    is monthly subscription >$15/mo → WEAKEN (force model conversion):
      • indie developers / solo hackers
      • volunteer / non-profit organizations
      • micro-agencies (<5 people)
      • seasonal businesses (event organizers, tax-season tools,
        summer camps, tour operators, holiday retail)
      • self-employed in non-regulated hobby niches
      • privacy-conscious / self-host crowd
    Fix direction in notes: "Convert to one-time $X, freemium+ads,
    donation/Patreon, or seasonal prepay."

15. REGULATORY / LIABILITY POSITIONING
    If the idea provides medical / legal / financial / safety advice to
    end-users (not B2B-licensed-professionals) AND the cost estimate
    lacks legal-review or insurance line items → WEAKEN with
    "add disclaimer + lawyer review ($2-5k) or pro indemnity ($3-10k/yr)
    to estimated_cost_usd; position as 'information only, not advice'."
    Idea is NOT killed by regulation — must be positioned correctly.

16. ANTI-PATTERN CHECK
    Catch and fail any idea built on these failure patterns:
      • "Recent news → urgent buyer demand" → WEAKEN (replace why_now)
      • "Tool for everyone who wants X" → WEAKEN (narrow ICP)
      • "Niche underserved" without buyer-payment evidence → KILL
      • "$29/mo for use-case <5 times/month" → WEAKEN (cadence)
      • "Meta ads will drive customers" for products under $50
        → WEAKEN (channel CAC > LTV)
      • "Workflow lock-in after 5-week MVP" as moat → KILL fake moat

==============================================================================
AI-TOOLS PENALTY (Maksim explicit ask)
==============================================================================
If idea.industry == "ai_tools" OR title contains "AI X" / "AI-powered" / "AI
for", apply STRICTER bar — confidence must be ≥75 AND moat must be REAL
(per axis 5) to PASS. Otherwise KILL or WEAKEN. Reason: Maksim is fatigued
by generic "AI for X" pitches.

==============================================================================
VERDICTS (one per idea, MUST be one of these four)
==============================================================================

  KILL          Fatal flaw: no buyer payment evidence + no moat, OR
                wrong-cadence pricing, OR fake-moat-only, OR enterprise
                tool ships this free, OR wildly infeasible.
                Drop permanently.

  WEAKEN        Salvageable. The idea generator rewrites it next round
                using your `notes` as instruction. Be SPECIFIC in notes
                — name what axis failed + the FIX direction. Example:
                "WEAKEN: revenue is $49/mo for once-a-year tax-prep
                problem. Switch to D) one-shot $99 per filing season +
                free triage tool for lead capture. Also list Excel
                templates from r/tax as the real free competitor."

  PASS          Solid. Acceptable. Not exceptional but worth shipping.
                Buyer-fit is plausible, moat is real-enough, distribution
                channel is named, cadence matches usage.

  STRONG_PASS   All of: clear buyer with payment evidence, named
                distribution channel, real moat (not fake), why_now is
                a measurable shift (not news cycle), revenue cadence
                matches usage, estimated_cost is honest, Maksim's stack
                fits the deliverable.

==============================================================================
CALIBRATION TARGET (Maksim's preferred mix)
==============================================================================
Healthy run = 10-20% KILL, 20-30% WEAKEN, 50-65% PASS, 5-15% STRONG_PASS.
If your last batch hit >30% KILL, you over-rejected — reset toward WEAKEN.
WEAKEN is the workhorse verdict; reach for it before KILL whenever the
core insight is rescuable by narrowing the niche, fixing the cadence, or
swapping the distribution channel.

Default toward PASS when uncertain. The /more command needs 4-5 survivors
to give Maksim variety; cutting to 1-2 means an empty pool.

==============================================================================
NOTES FIELD (≤400 chars per verdict)
==============================================================================
For KILL: name the FATAL axis + the concrete failure ("KILL: axis 1 —
no buyer payment evidence. Only competitor listed is Excel; r/tax shows
people use free templates and don't pay. Without proof of paid adoption
in adjacent products, this is a free-complaint problem.").

For WEAKEN: name the failing axis + the SPECIFIC FIX direction for the
rewriter. ("WEAKEN: axis 6 — generic 'businesses' customer. Narrow to
indie real-estate agents at <50 listings, found in r/RealEstateInvesting
and Tom Ferry coaching Facebook group. Also: revenue is wrong cadence —
agents pay per-listing not per-seat.")

For PASS/STRONG_PASS: brief justification touching axes 1, 5, 6.

==============================================================================
OUTPUT
==============================================================================
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
