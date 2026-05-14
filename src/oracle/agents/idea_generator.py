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

Cost: ~8-12 ideas * gpt-4o = ~$0.03-0.05 per call. Up to 3 calls per
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
        max_length=12,
        description="8-12 business ideas, sorted by confidence DESC (Maksim "
                    "wants more variety so /more pool isn't empty after top-3)",
    )


# ============================================================================
# System prompt — heavily tuned for Maksim's profile
# ============================================================================


IDEA_GEN_SYSTEM = """\
You are ORACLE's business idea generator for Maksim.

USER PROFILE (critical — every idea must fit):
- Python AI engineer, ships ONLINE / web / SaaS products as solo MVPs in 2-6 weeks
- Stack: Python, FastAPI, LangGraph, RAG, LLMs, Docker, Azure, Postgres/SQLite
- MVP budget: under $5,000, solo buildable, time to first paying customer < 3 months
- Available after 15:00 Warsaw local
- AI/RAG is in his toolkit but NOT a required ingredient. Plain CRUD-SaaS,
  scraping aggregators, marketplaces, dashboards, automation tools — all fair
  game. Pick AI when AI is the right tool, not by reflex.

WHAT'S IN SCOPE (industries — go BROAD, all corners of life):
- Health & wellness: trackers, telehealth, mental health, sleep, nutrition
- Fitness & sport: workout planners, club management, coach platforms
- Education: tutoring, online courses, exam prep, language learning
- Marketing & adtech: SEO, social, ad ops, attribution, creator tools
- Fintech consumer: budgeting, taxes, expense tracking, crypto tooling
- Insurance: claim triage, broker tools, underwriting helpers
- E-commerce & retail: niche storefronts, dropship ops, supplier discovery
- Entertainment & media: video tools, podcasts, fan platforms
- Gaming: indie-game analytics, Discord bots, tournament platforms,
  game-design SaaS, streamer ops
- Productivity: notes, scheduling, knowledge mgmt, inbox triage
- Dev tools: CLIs, observability, deployment, debug helpers
- Creator economy: newsletter ops, Patreon-style tools, content scheduling
- B2B services: CRM, ATS, HR-tech, recruiting, sales engagement, legaltech
- Travel, hospitality, real-estate, food/beverage
- INDUSTRIAL verticals (SOFTWARE only, not hardware):
  · Energy: utility dashboards, grid analytics, ESG reporting,
    energy-trading helpers, RAG-search over EIA/IEA reports
  · Transport & mobility: fleet management, driver scheduling, EV-charging
    network analytics, last-mile delivery ops
  · Agritech: crop pricing predictors, farm management SaaS, supply chain
    for small farmers
  · Logistics & supply chain: freight broker scoring, warehouse analytics,
    customs paperwork automation
  · Aerospace: OSINT for launches, satellite-data SaaS, supplier intel
  · Defense-tech: procurement intelligence, OSINT dashboards, training
    simulation, supply-chain compliance for defense contractors,
    drone-fleet management SOFTWARE (not the drones themselves)

WHAT IS OUT OF SCOPE (hard dealbreakers):
- Physical products of any kind (hardware, manufactured goods, drugs,
  vehicles, weapons themselves — even if the surrounding industry is in
  scope; you build SOFTWARE for these verticals, never the hardware)
- Food delivery requiring drivers (the courier ops, not delivery software)
- Enterprise sales cycles longer than 3 months (no Fortune-500 procurement)
- Pure offline services (consulting, agencies, in-person therapy, gym buildouts)
- Idea that is literally "ChatGPT for X" with no defensible workflow or data moat
- Any vertical requiring multi-month legal/medical certification BEFORE day-one
  revenue (e.g. FDA-cleared medical device, banking license, ITAR-controlled
  weapons export licence). Light compliance is fine — ITAR-adjacent OSINT or
  unclassified procurement data is fine; building actual munitions is not.

WHAT IS FINE (any online/web product Maksim can ship as solo dev):
- Web SaaS, dashboards, CRMs, marketplaces, analytics platforms
- Browser extensions (Chrome, Firefox), bookmarklets
- Telegram bots, Discord bots, Slack apps
- API-first products / developer tools
- AI agents, RAG-search platforms, scrapers
- PWA / responsive web apps (work on mobile via browser — that's fine)
- Telegram Mini Apps, Slack apps, Notion integrations
- Newsletter platforms, content tools, no-code helpers
- Anything else that lives ON THE INTERNET — sites, web products, APIs.
The unifying constraint: ship as a SOLO Python/web developer in 2-6 weeks.

DIVERSITY RULE (HARD CONSTRAINT — Maksim's #1 repeated complaint):
He is SICK of getting tech-only batches (AI dashboard, dev tool, SaaS X,
AI agent Y...). NO TECH BIAS. Every industry from the list below has
EQUAL probability of being picked — your job is to surface the BEST
opportunity from each domain this run, not to default to tech.

EQUAL-WEIGHT INDUSTRY QUOTA for a batch of 8-12 ideas:
- AT MOST 1 idea total from the TECH-bucket: {saas, ai_tools, dev_tools}
  → If you pick saas, you cannot also pick ai_tools. Pick the SINGLE
    strongest tech opportunity from this run; no second tech idea allowed.
- AT LEAST 1 idea from HEALTH/WELLNESS: {health, fitness_sport}
- AT LEAST 1 idea from CONSUMER: {marketing_adtech, creator_economy,
  ecommerce_retail, entertainment_media, food_beverage, gaming}
- AT LEAST 1 idea from PROFESSIONAL: {education, fintech, insurance,
  b2b_services, hr_recruiting, legaltech, real_estate, productivity}
- AT LEAST 1 idea from INDUSTRIAL: {energy, agritech, logistics_supply,
  aerospace, defense_tech}  ← SOFTWARE only
- AT LEAST 1 idea from TRAVEL/MOBILITY: {travel_hospitality, transport_mobility}
  ← Maksim says these are CONSISTENTLY MISSING. Examples: hotel-tech /
  tour-operator software, AirBnB host analytics, Booking.com partner tools,
  EV-charging fleet management, ride-share routing, last-mile delivery
  scheduling, parking SaaS, car-rental ops. STILL produce one stretch
  idea here even if signals are weak (confidence 55-65 ok).
- AT LEAST 1 idea from LIFESTYLE/CONTENT: {sport_content, gambling_igaming}
  → sport_content = sports media / fantasy / analytics / fan tools
  → gambling_igaming = 18+ casino/sportsbook/poker/betting SOFTWARE
    (compliance, odds analytics, NOT running a casino itself)

If a required non-tech bucket has weak signal this run, INVENT a plausible
online MVP from adjacent signals (Reddit health/sport/casino subs, marketing
trends, edu requests, defense news, agritech VC deals). Lower confidence
to 55-65 for these stretch ideas — let critic decide. NEVER omit a
required bucket because tech looks "stronger". Tech ALREADY won 1 slot —
the other 4-9 are for the rest of the world.

INDUSTRIAL VERTICALS — SOFTWARE ONLY:
For energy/transport/agritech/logistics/aerospace/defense ideas, you MUST
propose a SOFTWARE product (web dashboard, RAG-search, CRM, scheduling,
OSINT, simulation, claims processing, fleet telemetry, supplier discovery,
compliance automation, etc.) — never a physical hardware product. Examples:
- ENERGY: "RAG-search over EIA + IEA reports for energy analysts"
- TRANSPORT: "Driver scheduling SaaS for last-mile delivery fleets"
- AGRITECH: "Crop pricing predictor dashboard for small US farmers"
- LOGISTICS: "Browser extension scoring freight broker reliability"
- AEROSPACE: "OSINT dashboard tracking satellite launches + manifests"
- DEFENSE: "Procurement intelligence platform for defense contractors"
- GAMING: "Discord-bot tracking indie game launch metrics"
- INSURANCE: "AI claims-triage SaaS for small auto-insurance brokers"

ALLOWED `industry` VALUES (use EXACTLY one of these strings, lowercase
with underscores — DO NOT invent custom values like "Biotech/AI" — pick
the closest match from this list):
  health, fitness_sport, education, marketing_adtech, fintech,
  ecommerce_retail, entertainment_media, productivity, dev_tools,
  creator_economy, b2b_services, saas, ai_tools, hr_recruiting,
  legaltech, travel_hospitality, real_estate, food_beverage,
  energy, transport_mobility, agritech, gaming,
  logistics_supply, insurance, aerospace, defense_tech, other

YOUR INPUT: 5-10 cross-signal insight clusters (category=business_idea) from
the synthesizer. Each cluster represents a real cross-signal pattern from
this run's collected signals (HN, Reddit across many subs, GitHub trending,
ProductHunt, VC deal flow, news RSS).

YOUR JOB: produce 5-10 SPECIFIC business idea drafts spanning ≥3 industries.
Each idea will then go through 3 rounds of ruthless critique. Quality over
quantity — better to produce 5 strong drafts than 10 mediocre ones.

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
    communities, Twitter cohorts, conferences/meetups, niche newsletters,
    industry Facebook groups. Example for a fitness idea: "r/Fitness",
    "r/loseit", "Lean Gains Facebook group", "Running Magazine newsletter
    sponsor". For a marketing tool: "r/marketing", "Demand Curve Slack",
    "AppSumo deals page", "GrowthHackers community". Not "social media".
18. `industry`: ONE of the allowed industry values listed above. Pick the
    closest fit. Generate ideas across DIFFERENT industries — the downstream
    selector enforces 3 unique industries in the final delivery.

EXAMPLES of well-tagged ideas (varied industries):
- "AI form-filler for clinical trial participant intake" → industry=health
- "Coach-marketplace for triathlon training plans"      → industry=fitness_sport
- "Browser extension scoring Shopify supplier reviews"  → industry=ecommerce_retail
- "Discord-bot for newsletter sponsor matchmaking"      → industry=creator_economy
- "RAG-search over legal contract precedents"           → industry=legaltech

OUTPUT: 8-12 ideas, sorted by confidence DESC, spanning ≥3 industries.
Respond ONLY with valid JSON matching the schema.
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
    """Compose the idea_generator system prompt with TWO injection slots:
    1. MANUAL preferences (set via /preferences command) — highest authority
    2. LEARNED preferences (auto-calibrated from feedback) — second-tier

    Both are appended to the base prompt. Manual ALWAYS wins over learned
    when they conflict — explicit user wishes beat inferred patterns.
    """
    base = IDEA_GEN_SYSTEM
    manual = ""
    learned = ""
    try:
        from ..learning import get_manual_preferences, get_prompt_injection_ideas  # noqa: PLC0415
        manual = await get_manual_preferences()
        learned = await get_prompt_injection_ideas()
    except Exception as e:  # noqa: BLE001
        log.debug("idea_generator: could not read learning weights: %s", e)

    suffix = ""
    if manual:
        suffix += (
            "\n\n----- MAKSIM'S MANUAL PREFERENCES (set via /preferences) -----\n"
            f"{manual}\n"
            "These are EXPLICIT user wishes. Honor them ABOVE all other "
            "rules. If they conflict with quotas, his manual preference wins."
        )
    if learned:
        suffix += (
            "\n\n----- AUTO-LEARNED PREFERENCES (from feedback calibration) -----\n"
            f"{learned}\n"
            "Weight these strongly when generating ideas — they reflect "
            "Maksim's actual past clicks. If they conflict with manual "
            "preferences above, manual wins."
        )
    return base + suffix


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
        f"Generate 8-12 specific BusinessIdea drafts grounded in these clusters."
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
- Python AI engineer, ships ONLINE/web/SaaS solo MVPs in 2-6 weeks
- Stack: Python, FastAPI, LangGraph, RAG (optional), Docker, Azure, Postgres
- Industries: ANY (health, fitness, education, marketing, fintech, e-comm,
  productivity, dev-tools, b2b services, energy, transport, agritech, gaming,
  logistics, insurance, aerospace, defense-tech SOFTWARE, etc.) — any field
  of human activity is fair game as long as the deliverable is an online/web
  product.
- Fine to ship: web SaaS, browser extensions, Telegram/Discord/Slack bots,
  PWAs, Telegram Mini Apps, RAG agents, scrapers, dashboards. No artificial
  ban on responsive web that works on mobile — only avoid native iOS/Android-
  ONLY products that REQUIRE App-Store gating from day one.
- Avoid: physical products / hardware, enterprise multi-month sales,
  heavy regulatory pre-day-one (FDA, banking licenses), pure offline services.
- Preserve the `industry` tag if already set; only change it if the pivot
  fundamentally moves the idea into a different vertical.

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
# Industry diversity selector — picks 3 ideas across 3 unique industries
# ============================================================================


# Canonical industry vocabulary. Anything outside this list is normalized
# to "other" before the diversity selector runs — otherwise the LLM can
# invent unique-looking tags like "Biotech / Data Infrastructure" and game
# the diversity check.
ALLOWED_INDUSTRIES: set[str] = {
    "health", "fitness_sport", "education", "marketing_adtech",
    "fintech", "ecommerce_retail", "entertainment_media", "productivity",
    "dev_tools", "creator_economy", "b2b_services", "saas", "ai_tools",
    "hr_recruiting", "legaltech", "travel_hospitality", "real_estate",
    "food_beverage",
    # 2026-05 additions (broader-than-mainstream + sport + gambling):
    "energy", "transport_mobility", "agritech", "gaming",
    "logistics_supply", "insurance", "aerospace", "defense_tech",
    "sport_content",       # sports media, fantasy, analytics platforms, fan tools
    "gambling_igaming",    # 18+ casino, sportsbook, poker, betting analytics
    "other",
}

# Tech-heavy industries — diversity selector treats them as ONE bucket
# so the LLM can't game it by tagging 3 ideas as 3 different tech tags.
TECH_BUCKET: set[str] = {"saas", "ai_tools", "dev_tools"}


def _normalize_industry(raw: str | None) -> str:
    """Coerce an LLM-emitted industry tag to the canonical lowercase form.

    Examples of normalization:
      "Biotech / Data Infrastructure" → "health"  (contains 'bio')
      "Marketing & AdTech"             → "marketing_adtech"
      "ai-tools"                       → "ai_tools"
      None                              → "other"
    """
    if not raw:
        return "other"
    t = raw.strip().lower().replace(" / ", "_").replace("/", "_").replace("&", "_").replace("-", "_")
    t = "_".join(p for p in t.split() if p)  # collapse whitespace
    if t in ALLOWED_INDUSTRIES:
        return t
    # Heuristic mapping for common LLM phrasings
    if any(k in t for k in ("bio", "health", "medical", "telemed", "wellness", "mental")):
        return "health"
    if any(k in t for k in ("fitness", "sport", "workout", "running", "gym")):
        return "fitness_sport"
    if any(k in t for k in ("market", "adtech", "seo", "ad_ops", "growth")):
        return "marketing_adtech"
    if any(k in t for k in ("creator", "influencer", "newsletter", "patreon")):
        return "creator_economy"
    if any(k in t for k in ("edu", "tutor", "course", "learn", "teacher")):
        return "education"
    if any(k in t for k in ("fin", "bank", "tax", "budget", "crypto", "payment")):
        return "fintech"
    if any(k in t for k in ("ecom", "retail", "shop", "store", "dropship")):
        return "ecommerce_retail"
    if any(k in t for k in ("entertain", "media", "podcast", "video", "stream")):
        return "entertainment_media"
    if any(k in t for k in ("hr", "recruit", "hiring", "ats")):
        return "hr_recruiting"
    if any(k in t for k in ("legal", "law", "contract")):
        return "legaltech"
    if any(k in t for k in ("travel", "hotel", "trip", "tourism")):
        return "travel_hospitality"
    if any(k in t for k in ("real_estate", "realestate", "property", "rent")):
        return "real_estate"
    if any(k in t for k in ("food", "restaurant", "cafe", "drink", "bev")):
        return "food_beverage"
    if any(k in t for k in ("dev", "engineer", "infra", "devops", "cli")):
        return "dev_tools"
    if any(k in t for k in ("agent", "llm", "ai", "rag", "gpt")):
        return "ai_tools"
    if any(k in t for k in ("productiv", "note", "todo", "calendar", "kanban")):
        return "productivity"
    if any(k in t for k in ("b2b", "crm", "sales", "ops")):
        return "b2b_services"
    # 2026-05 additions
    if any(k in t for k in ("energy", "utility", "grid", "solar", "oil", "gas", "nuclear", "power")):
        return "energy"
    if any(k in t for k in ("transport", "mobility", "vehicle", "auto", "ev", "fleet", "ride", "taxi", "carshare", "scooter", "bike_share")):
        return "transport_mobility"
    if any(k in t for k in ("agri", "farm", "crop", "livestock", "agtech", "agronom")):
        return "agritech"
    if any(k in t for k in ("game", "gaming", "esport", "studio", "indie_game")):
        return "gaming"
    if any(k in t for k in ("logist", "supply", "freight", "warehouse", "shipping", "cargo")):
        return "logistics_supply"
    if any(k in t for k in ("insur", "underwrit", "claim", "actuar", "risk_pool")):
        return "insurance"
    if any(k in t for k in ("aerospace", "space", "satellite", "launch", "rocket", "orbit")):
        return "aerospace"
    if any(k in t for k in ("defense", "military", "weapon", "armed", "intel", "milspec", "warfare", "drone_combat", "c2")):
        return "defense_tech"
    if any(k in t for k in ("sport_content", "sport_media", "fantasy", "sport_analytic", "fan_platform", "esports_media")):
        return "sport_content"
    if any(k in t for k in ("gambling", "igaming", "casino", "sportsbook", "betting", "poker", "lottery", "wagering")):
        return "gambling_igaming"
    if t == "saas":
        return "saas"
    return "other"


def select_diverse_ideas(
    ideas: list[dict],
    *,
    target_count: int = 3,
) -> list[dict]:
    """Pick `target_count` ideas favoring distinct `industry` values.

    Algorithm (greedy):
      1. Sort all ideas by score DESC (confidence + STRONG_PASS bonus).
      2. Walk in order; pick an idea only if its industry hasn't been picked.
      3. If we run out of unique-industry candidates before reaching target_count,
         fall back to filling with the next-best ideas (allowing duplicates).

    Verdict-aware: KILL'ed ideas are filtered out first.

    Returns the picked subset (NEW list, not mutating input).
    """
    if not ideas:
        return []

    # Normalize industry tag on every idea first — LLM tends to invent
    # non-canonical labels like "Biotech / Data Infrastructure" that
    # otherwise look unique and bypass the diversity check.
    for idea in ideas:
        idea["industry"] = _normalize_industry(idea.get("industry"))

    # Filter out killed ideas
    alive = [i for i in ideas if (i.get("verdict") or "").upper() != "KILL"]
    if not alive:
        return []

    # Min-confidence floor — weak fills (conf<60) are demoted so they only
    # appear when no quality option exists for a required bucket.
    MIN_CONFIDENCE_PREFERRED = 60

    def _score(idea: dict) -> tuple[int, int, int, int]:
        verdict = (idea.get("verdict") or "").upper()
        # STRONG_PASS bonus = 1; LIST: above min confidence = 1 (kills weak fills)
        strong_bonus = 1 if verdict == "STRONG_PASS" else 0
        conf = int(idea.get("confidence") or 0)
        confidence_tier = 1 if conf >= MIN_CONFIDENCE_PREFERRED else 0
        rounds = int(idea.get("reflexion_rounds_passed") or 0)
        return (confidence_tier, strong_bonus, conf, rounds)

    ranked = sorted(alive, key=_score, reverse=True)

    def _bucket(industry: str) -> str:
        """All tech industries collapse to one bucket so LLM can't game
        diversity by emitting saas + ai_tools + dev_tools as 3 'unique' picks."""
        return "TECH" if industry in TECH_BUCKET else industry

    picked: list[dict] = []
    seen_buckets: set[str] = set()

    # Pass 1: pick the BEST idea from each unique bucket (no TECH priority —
    # all buckets compete by score). Greedy order is the global score sort.
    for idea in ranked:
        bucket = _bucket(idea.get("industry") or "other")
        if bucket not in seen_buckets:
            picked.append(idea)
            seen_buckets.add(bucket)
            if len(picked) >= target_count:
                break

    # Pass 2: if still short, fill with next-best regardless of duplicate bucket
    if len(picked) < target_count:
        picked_ids = {id(i) for i in picked}
        for idea in ranked:
            if id(idea) in picked_ids:
                continue
            picked.append(idea)
            if len(picked) >= target_count:
                break

    log.info(
        "idea_diversity: picked %d/%d ideas across industries %s (buckets %s)",
        len(picked),
        target_count,
        sorted({(i.get("industry") or "other") for i in picked}),
        sorted({_bucket(i.get("industry") or "other") for i in picked}),
    )
    return picked


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
