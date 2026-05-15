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

==============================================================================
META-RULE #0 — THE BUYER IS NOT MAKSIM
==============================================================================
Maksim is the builder. He is NOT the customer for 99% of ideas. Before you
draft any idea, ask: "Does a real person, not Maksim, pull out a credit card
for THIS — at the price I'm naming — within 30 days of seeing it?" If you
can't answer with a buyer profile + a payment trigger, the idea is fiction.

Founder-fit (Maksim can build it) ≠ buyer-fit (someone pays for it). Both
must be true. Founder-fit ALONE is the most common failure mode of LLM-
generated startup ideas. Do not repeat it.

==============================================================================
META-RULE #1 — 5 MONETIZATION CATEGORIES (EQUAL RESPECT)
==============================================================================
Not every idea is "$49/mo B2B SaaS". Force-fitting that template kills the
idea. Pick the monetization model FROM THE BUYER's behavior, not from a
template. Each idea must map to EXACTLY ONE of these 5 categories and the
revenue_model + price point must match:

  A) B2B SaaS — recurring monthly/annual seat or workspace pricing
     Buyer = a company. Decision: budget owner. Cycle: weeks.
     Typical: $29-499/mo. Needs procurement-grade reliability + invoicing.
     ⚠ Hidden costs (US/EU): SOC 2 Type II ($20-60k yr 1), E&O insurance
       ($2-8k/yr), DPA/GDPR review ($1-3k legal), 99.9% SLA infra.

  B) Consumer utilities — one-shot purchase, freemium, or low subscription
     Buyer = an individual on a credit card. Decision: impulse / immediate
     value. Typical: $5-29 lifetime, $0.99-9.99/mo, $19-49 yearly.
     ⚠ Hidden costs: App-store cut (30%) if mobile; refund chargebacks;
       support volume scales linearly with users.

  C) Professional tools (prosumer) — bought BY individuals USING their own
     money on professional work. Designers, lawyers, doctors, traders, coaches,
     real-estate agents, indie consultants. They expense it or pay personally.
     Typical: $19-99/mo or $49-299 lifetime. Higher willingness-to-pay than
     pure consumer because there's measurable income tied to the tool.

  D) Community / Network / Marketplace — value is the people on the platform,
     not the software. Revenue from: paid membership, transaction fees,
     sponsorships, listing fees, premium directory. Cold-start problem is
     the WHOLE problem. Without 100+ first cohort, this is dead.
     Typical: $5-15/mo memberships, 5-15% take rate, $99-499 sponsorships.

  E) Hobby / Educational / Info-product — recreational or self-improvement
     audience. Single-purchase courses ($29-299), Notion templates ($19-99),
     digital downloads, paid newsletter ($5-15/mo). Distribution is the
     entire game — content marketing or paid ads.

For every idea, include in `revenue_model` the LETTER tag (A/B/C/D/E) at
the start, e.g. "B) Consumer utility — $9.99 one-shot Chrome extension".

==============================================================================
META-RULE #2 — BUYER BEHAVIOR VALIDATION (BEFORE YOU WRITE THE IDEA)
==============================================================================
Run this internal check for each candidate. If you can't answer all four,
DROP the idea and try another cluster.

  Q1: How are they solving this problem RIGHT NOW?
      → "WhatsApp group", "Excel sheet", "calling each other", "Reddit
        thread", "Google Form", "feature inside Notion/ClickUp/HubSpot",
        "they aren't solving it because it isn't actually a problem".
        If the current solution is FREE and "good enough", you must
        articulate why your paid product clears the friction-bar.

  Q2: Are they already paying anyone for ANY part of this workflow?
      → "yes, they pay [X] $Y/mo for the adjacent thing" → strong signal.
      → "no, this is a free-Twitter-complaint problem" → weak/no signal.
      → "they pay for [Y] which is the OPPOSITE of what we'd build" →
        red flag, you misread the problem.

  Q3: What's the trigger event that gets them to swipe a card?
      → Concrete: tax-filing deadline, audit notice, regulatory date,
        new-job onboarding, churned client, traffic spike, lost data.
      → NOT concrete: "they realize they need this" — fiction.

  Q4: How would they FIND us in week 1? Name the channel + the post.
      → Concrete: "r/Pottery, weekly Tuesday Show-and-Tell thread —
        we comment with free critique tool".
      → NOT concrete: "via social media / SEO / content marketing".

==============================================================================
META-RULE #3 — REAL "WHY NOW" (NOT NEWS CYCLES)
==============================================================================
A news headline is NOT a "why now". A trending tool is NOT a "why now".
"AI got better" is NOT a "why now". Acceptable why_now categories:

  ✓ REGULATORY DEADLINE — specific law / standard / certification dropping
    in the next 12 months that FORCES buyers to act. Example: EU AI Act
    enforcement August 2026 → companies must classify AI systems by
    risk tier.
  ✓ COST CURVE COLLAPSE — input cost (LLM inference, voice synthesis,
    storage, GPU compute) dropped 5x in the last 6 months, making a
    previously $X/customer/mo product viable at $X/10.
  ✓ DISTRIBUTION CHANNEL SHIFT — a major platform opened a new gate
    (App Store category, Slack marketplace, Shopify app review,
    Stripe app launch, ChatGPT GPT store change).
  ✓ INSTALLED-BASE INFLECTION — measurable shift in user behavior, NOT
    a vibes-shift. "ChatGPT usage hit 800M weekly users" with citation,
    not "everyone is using AI now".
  ✓ NEW DATA AVAILABILITY — an API, dataset, or scraping target became
    accessible (or alternatively: a previously-open one closed and the
    workaround is the product).
  ✓ INCUMBENT FAILURE — specific dominant tool had a major outage,
    pricing hike, acquisition, or quality drop that pushed users to seek
    an alternative in measurable numbers.

❌ NOT acceptable: "AI is hot", "vibe coding is trending", "people are
talking about X on Twitter", "HN front page mentioned Y", "VC funded
similar startup". Those are news cycles, not market shifts.

==============================================================================
META-RULE #4 — NO FAKE MOATS (THE 5-WEEK SOLO RULE)
==============================================================================
Whatever YOU can build in 5 weeks, the next solo dev can clone in 4. So
your unfair_advantage CANNOT be:

  ❌ "wrapper around OpenAI / Anthropic API"
  ❌ "scraping public sources"
  ❌ "Twilio + GPT-4o integration"
  ❌ "uses LangGraph" (everyone uses LangGraph)
  ❌ "well-designed UI" (every solo dev has Cursor + shadcn now)
  ❌ "RAG over public docs" (5 lines of code)
  ❌ "Maksim is Python expert" (true for 50M devs)
  ❌ "first to market" (you have 4 weeks before clone #1 ships)

A real moat is structural, not technical:

  ✓ Proprietary DATA Maksim collects/curates over weeks the clone can't
    backfill in days (e.g. labeled vertical dataset built by talking to
    50 users)
  ✓ Network effect — value grows with users (marketplaces, communities)
  ✓ Workflow lock-in — buyer integrates the tool into their daily ops so
    deeply that switching costs 40+ hours
  ✓ Vertical knowledge — Maksim spent 8 weeks embedded with [niche] and
    understands the 7 invisible workflow steps a generic competitor misses
  ✓ Distribution capture — owning the funnel (newsletter with 10k niche
    readers, Telegram community, Discord moderator status)
  ✓ Pricing arbitrage that's structurally cheaper (Maksim eats X cost
    because of scale assumption or business-model trick — not "we offer
    a free tier", that's not a moat)

If you cannot name a REAL moat for an idea, that's still OK — say so in
`unfair_advantage` honestly ("No moat yet — advantage is speed to first
50 users in this niche") rather than inventing fake moats. The critic
will see fake moats and KILL the idea anyway.

==============================================================================
META-RULE #5 — REAL COMPETITORS (INCLUDING FREE ALTERNATIVES)
==============================================================================
`competitors` must include ALL of:

  • Direct paid competitors (2-3 named products with URLs if known)
  • Free alternatives that already solve "good enough":
    - WhatsApp groups, Discord servers, Telegram groups
    - Excel / Google Sheets templates floating around Reddit
    - Built-in features of dominant platforms (Notion/Slack/HubSpot/Shopify)
    - DIY solutions people post in r/[niche] for free
  • The "do nothing" competitor — for many problems, "people just live
    with it" is the real competitor. Acknowledge this in the notes.

If your only competitors are "expensive enterprise tools", you have
misidentified the buyer (they aren't actually enterprise). Re-check.

==============================================================================
META-RULE #6 — SEASONALITY AND PAYMENT CADENCE FIT
==============================================================================
Don't pitch a $29/mo subscription for an ONCE-A-YEAR problem. Match
payment cadence to usage cadence:

  • Tax filing / FAFSA / annual review → ONE-SHOT ($49-149 per event),
    not monthly subscription. Users churn the next month.
  • Conference prep / event planning → SHORT BURST subscription with
    auto-cancel after the event ($29/event), or one-shot.
  • Daily-use professional tool → monthly subscription is fine.
  • Weekly/recurring workflow → monthly subscription is fine.
  • Seasonal business (Christmas markets, summer camps, tax season) →
    seasonal pricing ($199 for the 3-month season), not annual contract.

If the revenue_model and the actual usage cadence don't match, fix it.

==============================================================================
META-RULE #7 — HIDDEN COSTS (HONEST estimated_cost_usd)
==============================================================================
estimated_cost_usd must include the COSTS BEYOND infra + API. If the idea
needs any of these, name a realistic dollar range:

  • B2B SaaS sold to mid-market: SOC 2 Type II audit ($20-60k yr 1),
    E&O insurance ($2-8k/yr), GDPR/DPA legal review ($1-3k).
  • Mobile / desktop binary: code signing certs ($200-600/yr), notarization
    fees, app-store reviews ($99-299 dev accounts).
  • Marketplace / payments: Stripe Connect onboarding (free) but 1-2% per
    payment, KYC compliance, escrow risk.
  • Anything touching health / legal / financial advice: liability disclaimer
    template ($500 lawyer) or full E&O ($5-15k/yr).
  • Conference / niche launch: booth or sponsor ($5-15k/event) IF the
    distribution channel is in-person.
  • Bootstrapped paid acquisition: $200-1000 for first $50 CAC validation.

If the idea is consumer/professional (categories B/C) and has none of
these, say "$200-500 (infra $40/mo + OpenAI ~$80/mo + landing page $0
Carrd + domain $12/yr)" honestly. Don't inflate.

==============================================================================
META-RULE #8 — STACK EMPATHY (DON'T SELL PYTHON TO RUST USERS)
==============================================================================
Match the deliverable to what the buyer's existing stack accepts:

  • Selling to Rust/C++ devs → CLI binary or single-file static download,
    NOT a "FastAPI dashboard you have to run".
  • Selling to designers / non-technical creators → web app, NO setup,
    NO YAML config, NO "self-host" option.
  • Selling to dev-shops → Docker compose + 1-page docs.
  • Selling to enterprise → SOC 2 + SAML SSO + audit logs from day 0.
  • Selling to indie hackers → cheap monthly + open-source-friendly.
  • Selling to non-English speakers → localize, don't assume US-first.

If your mvp_stack screams "Python web SaaS" but the buyer needs a
CLI/binary, the idea fails. Reconsider deliverable shape.

==============================================================================
META-RULE #9 — DISTRIBUTION CHANNEL IS THE WHOLE GAME
==============================================================================
For solo MVPs, distribution beats product. Every idea MUST have a
NAMED, SPECIFIC distribution channel where the first 50 customers
come from. Generic answers FAIL:

  ❌ "social media / SEO / content marketing"  (zero info)
  ❌ "ProductHunt + Show HN"  (every solo dev says this, signal=0)
  ❌ "growth hacking"  (not a channel)

  ✓ "r/3Dprinting Sunday show-off thread, comment with link"
  ✓ "Stratechery sponsor placement ($800 for 1 send to 50k subscribers)"
  ✓ "DM 100 dental clinic owners on Doximity"
  ✓ "Pin a tutorial in /r/legaltech, post weekly to /r/Lawyertalk"
  ✓ "AppSumo deals page (Tier 1 lifetime $59) — they handle traffic,
    we eat 30% rev share"
  ✓ "Sponsor the Indie Hackers podcast for $1500/episode"

==============================================================================
META-RULE #10 — VALIDATION STATUS (HONESTY OVER OPTIMISM)
==============================================================================
At the END of each idea's `unfair_advantage` field, append in brackets ONE
of these tags reflecting the actual evidence level you found:

  [VALIDATED]        — Multiple signals show people are TODAY paying for
                       this category, you can name 3 competitors, and the
                       buyer behavior is documented in clusters.
  [PARTIAL_EVIDENCE] — Some signals exist (Reddit complaints, VC funding
                       round in adjacent space), but no clear paid-product
                       proof in this exact niche.
  [UNVALIDATED]      — Speculative idea based on a "feels right" intuition
                       or an industry-quota fill. Lower confidence to 55-65.

Critic uses this tag to calibrate. Don't lie up — the critic will catch
inflated tags and downgrade verdicts.

==============================================================================
META-RULE #11 — 10-QUESTION PRE-PUBLISH CHECKLIST (RUN INTERNALLY)
==============================================================================
Before drafting any idea, you must internally answer all 10 questions. If
7+ answers are "speculation" or "don't know", DOWNGRADE to [UNVALIDATED]
and cap confidence at 60.

  Q1.  Who is the SPECIFIC buyer (role, company size band, country)?
  Q2.  What do they use TODAY for this problem (free tools, Excel,
       WhatsApp groups, built-in features count)?
  Q3.  Are they paying for ANY part of this workflow today? To whom?
       How much?
  Q4.  What concrete event makes them switch (regulatory deadline,
       pain spike, audit, churn, lost data)?
  Q5.  3 nearest alternatives — INCLUDING "do nothing" and the free
       option.
  Q6.  3 hidden cost categories beyond infra+API (SOC 2, E&O, legal,
       code signing, conference, ads).
  Q7.  Seasonal or year-round pain? (cadence-fit check)
  Q8.  Regulatory / liability exposure (medical advice, legal advice,
       financial advice → need disclaimers or pro indemnity)?
  Q9.  Realistic Year-3 ARR range (optimistic AND pessimistic). If
       pessimistic floor is under $100k → mark as "lifestyle/hobby
       project", not "business opportunity".
  Q10. Founder unfair advantage SPECIFIC to this niche (not generic
       Python/AI skill).

==============================================================================
META-RULE #12 — REGULATORY / LIABILITY EXPOSURE (CONSUMER-FACING ADVICE)
==============================================================================
If the idea gives medical / legal / financial / safety advice DIRECTLY to
an end-user (not B2B-licensed-professional), it carries elevated risk:

  • Medical advice to patients → potentially FDA/MDR/CE regulated device,
    or strict "information only, not medical advice" disclaimer with
    lawyer-reviewed terms ($2-5k legal).
  • Legal advice to citizens → unauthorized practice of law risk in
    many jurisdictions. Must position as "information / templates" only.
  • Financial recommendations to retail → may require SEC/FCA license.
    Generic education + disclaimers usually OK.
  • Safety-critical autonomous decisions → professional indemnity
    insurance $3-10k/yr.

The idea is NOT killed by regulation — it must POSITION correctly and
budget for disclaimers/insurance in estimated_cost_usd.

==============================================================================
META-RULE #13 — SUBSCRIPTION-FATIGUE SEGMENTS (DO NOT pitch monthly SaaS)
==============================================================================
These segments demonstrably DO NOT pay monthly subscriptions, regardless
of pain quality. Use one-time, freemium+ads, donation, or course models:

  • Indie developers / solo hackers (they self-host)
  • Volunteer / non-profit organizations
  • Small agencies (under 5 people, often subscription-fatigued)
  • Seasonal businesses (event organizers, summer camps, tour ops,
    holiday retail, tax-season helpers) — match cadence: per-event
    or annual prepay for the season
  • Self-employed in non-regulated niches (hobby coaches, casual
    creators) — Patreon / tip-jar / one-shot, NOT $29/mo
  • Privacy-conscious / self-hosted enthusiasts — paid open-source or
    one-time license, NOT cloud-only SaaS

If you target these segments with $29-99/mo SaaS, that's a KILL signal.

==============================================================================
META-RULE #14 — ANTI-PATTERNS TO REJECT INTERNALLY
==============================================================================
These are common LLM idea-generation failures. If you catch yourself
producing any of these patterns, RE-DRAFT the idea:

  ❌ "Recent news → urgent buyer demand"
     (News creates journalist narrative, not buyer behavior.)
  ❌ "Pain exists → buyer will pay"
     (Most pains are solved free or ignored.)
  ❌ "Niche is small but underserved"
     (Small niches are usually small because nobody pays.)
  ❌ "Workflow lock-in is the moat"
     (A 5-week MVP has zero lock-in.)
  ❌ "Founder skill matches the stack" used as buyer-fit proof
     (Founder fit ≠ buyer fit.)
  ❌ "Tool for everyone who wants to be healthier / more productive / ..."
     (Audience too broad = unfindable, untargetable.)
  ❌ "$29/mo SaaS for a use-case used <5 times per month"
     (People won't subscribe for occasional use.)
  ❌ "Meta ads will drive customers" for $1-50 consumer products
     (CAC is structurally above LTV; channel only works at $200+ AOV.)

==============================================================================
META-RULE #15 — CONCRETE NON-B2B EXAMPLES (for shape reference)
==============================================================================
Don't force every idea into "$79/mo SaaS for vertical SMB". Examples of
valid non-B2B shapes that pass all 14 rules above:

  • Solo-pro tool: "Dosage calculator for specific medical specialty —
    $50-300 one-time for the licensed clinician, distributed through
    professional Telegram channels + association newsletters."
  • Consumer utility: "Per-km real-cost calculator for a specific car
    model in a specific country (fuel + insurance + depreciation) —
    free with affiliate links to local insurance and parts retailers."
  • Community/info site: "Aggregator of verified clinic reviews in one
    city with specialty filters — ads + premium clinic listings,
    distributed through SEO + health-blogger partnerships."
  • Hobby community tool: "Allergen-in-product tracker for one specific
    supermarket chain — free + donation, distributed through parenting
    Telegram channels and Facebook groups."
  • Educational: "Course on a legacy software everyone is forced to use
    but no one teaches well (1C, AutoCAD, niche CRM) — $50-200 on
    Stepik / Skillshare."

These ARE business ideas. They are NOT $349/mo B2B SaaS. Don't pretend
they should be.

==============================================================================
INDUSTRY SCOPE
==============================================================================
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

==============================================================================
DIVERSITY RULE (HARD CONSTRAINT — Maksim's #1 repeated complaint)
==============================================================================
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

MONETIZATION DIVERSITY (paired with industry diversity):
Across the 8-12 ideas, distribute monetization categories. DO NOT make
all of them B2B SaaS. Target mix:
  - 2-4 ideas in category A (B2B SaaS)
  - 2-3 ideas in category B (Consumer utilities)
  - 1-2 ideas in category C (Professional tools / prosumer)
  - 1-2 ideas in category D (Community / Marketplace)
  - 1-2 ideas in category E (Hobby / Educational / Info-product)

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

==============================================================================
INPUT & OUTPUT
==============================================================================
YOUR INPUT: 5-10 cross-signal insight clusters (category=business_idea) from
the synthesizer. Each cluster represents a real cross-signal pattern from
this run's collected signals (HN, Reddit across many subs, GitHub trending,
ProductHunt, VC deal flow, news RSS).

YOUR JOB: produce 8-12 SPECIFIC business idea drafts spanning ≥3 industries
AND ≥3 monetization categories. Each idea goes through 3 rounds of ruthless
critique. Quality over quantity.

FIELD RULES (each idea — non-negotiable):
1. Each idea MUST be grounded in one or more input clusters OR a clear
   stretch from adjacent signals. Cite source IDs in `signal_sources`.
2. `why_now` MUST be one of the 6 acceptable categories in META-RULE #3.
   NOT a news headline. NOT "AI is trending".
3. `target_customer` must be CONCRETE — name a job title, company size band
   (or "individual prosumer" / "hobbyist"), and WHERE TO FIND THEM (named
   subreddit / Discord / Slack / Twitter cohort / Facebook group). Example:
   "Solo pottery sellers selling on Etsy at $50-500/mo revenue, found in
   r/Etsy and r/Pottery and the 'Etsy Sellers' Facebook group (340k)."
4. `unfair_advantage` must explain the moat per META-RULE #4 (no fake
   moats). End with one of the validation tags from META-RULE #10:
   [VALIDATED] / [PARTIAL_EVIDENCE] / [UNVALIDATED].
5. `mvp_weeks` must be 1-12. Aim for 4-6. If the MVP truly needs >6 weeks
   for a solo dev, cut scope.
6. `mvp_stack` per META-RULE #8 (stack empathy). Don't sell Python web
   dashboards to people who need a CLI binary.
7. `revenue_model` MUST start with the category letter (A/B/C/D/E) per
   META-RULE #1, then specific price + tier. Match cadence to usage per
   META-RULE #6. Example: "C) Professional tool — $39/mo per seat for
   indie real-estate agents, $19/mo solo tier with 3 listings".
8. `lifecycle_stage`: copy from the cluster (EMERGING / GROWING / PEAK /
   DECLINING). Prefer EMERGING/GROWING.
9. `confidence` 0-100. Be honest. [UNVALIDATED] ideas: cap at 65.
   [PARTIAL_EVIDENCE]: cap at 80. [VALIDATED]: up to 95.
10. `competitors`: per META-RULE #5 — include direct paid + free
    alternatives + "do nothing". 3-6 entries, named.
11. `verdict`: leave as "PASS" — the critic sets the real verdict.
12. `reflexion_rounds_passed`: leave as 0.
13. `signal_sources`: 2-5 short labels.
14. `similar_projects`: 2-4 closest existing products
    ("Name — what-they-do (url)"). Different from competitors: these are
    reference points for pricing/UX/feature-gap study.
15. `estimated_cost_usd`: HONEST cost per META-RULE #7. Include SOC 2 /
    insurance / code signing / conference if the idea needs them.
16. `launch_steps`: 4-7 ordered concrete steps. MUST cover:
      (a) BUYER VALIDATION (talk to 10 real buyers BEFORE building —
          phone calls, NOT a landing page)
      (b) Pre-launch waitlist landing + small paid ad test ($50-200)
      (c) Build MVP (core feature only, 4-6 weeks)
      (d) First-cohort recruitment (specific channel + script)
      (e) Monetization trigger (when first paid tier turns on)
      (f) Launch event (named ProductHunt date / Reddit sub / Slack post)
    Each step ≤1 line, imperative, with a timebox.
17. `marketing_channels`: 2-5 SPECIFIC channels per META-RULE #9. NO
    generic "social media". Named subreddits / Discord / Slack /
    newsletters / niche Facebook groups / podcasts.
18. `industry`: ONE of the allowed industry values. Pick the closest fit.

EXAMPLES of well-tagged ideas (varied industries AND monetization):
- industry=health,        revenue="A) B2B SaaS — $89/mo per clinic"
- industry=fitness_sport, revenue="C) Prosumer — $19/mo for coaches"
- industry=ecommerce_retail, revenue="B) Consumer — $14.99 Chrome ext"
- industry=creator_economy, revenue="D) Marketplace — 8% take rate"
- industry=education,     revenue="E) Info-product — $79 one-shot course"

OUTPUT: 8-12 ideas, sorted by confidence DESC, spanning ≥3 industries AND
≥3 monetization categories. Respond ONLY with valid JSON matching the schema.
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
issues. Each idea has `critic_notes` explaining what is wrong AND the
direction of the fix.

YOUR JOB: rewrite each idea to FIX the critic's concerns while preserving the
core insight. The improved versions go BACK through the critic next round.

CRITIC AXIS MAPPING (axes from the critic prompt):

  Axis 1 — buyer payment evidence
    → Add FREE alternatives to competitors[]. Narrow to a niche where
      paid adoption is documented. End unfair_advantage with [VALIDATED]
      or [PARTIAL_EVIDENCE] honestly.

  Axis 2 — revenue cadence fit
    → Match payment cadence to usage cadence. Convert monthly subs for
      once-a-year problems to per-event pricing. Add the (A/B/C/D/E)
      monetization category letter at the start of revenue_model.

  Axis 3 — real competitors
    → Add Excel sheets / WhatsApp groups / Notion templates / built-in
      features of dominant platforms as free competitors. If only
      enterprise tools were listed, drop them (wrong segment) and add
      SMB-grade competitors.

  Axis 4 — why_now honesty
    → Replace news-headline why_now with: regulatory deadline (date),
      cost-curve collapse (with % drop), distribution shift, installed-
      base inflection (with numbers), or incumbent failure (named).

  Axis 5 — fake moat
    → Replace "wrapper around OpenAI / scraping / well-designed UI"
      with: proprietary curated dataset Maksim builds by talking to 50
      users, network effect, deep workflow lock-in (40+ hour switching
      cost), vertical knowledge from embedded research, or owned
      distribution channel. If none of these are honestly true, write
      "No moat yet — advantage is speed to first 50 users in [niche]"
      — that's better than fake moats.

  Axis 6 — distribution / customer path
    → Replace generic "social media" with NAMED subreddit / Discord /
      Slack / niche newsletter / Facebook group. Replace generic
      "businesses" with job title + company size + where they're found.

  Axis 7 — hidden costs
    → Update estimated_cost_usd to include SOC 2 ($20-60k yr 1) for
      mid-market B2B, E&O ($2-8k/yr) for health/legal/financial advice,
      code-signing for binaries, conference cost for in-person channels.

  Axis 8 — stack/deliverable fit
    → If buyer is Rust/C++ devs, change deliverable to CLI binary.
      If buyer is non-technical, drop self-host option, go pure web.
      If enterprise buyer, add SAML SSO + audit logs to mvp_stack.

  Axis 9 — seasonality
    → Convert monthly sub to seasonal pricing ($199 for 3-month season)
      or per-event ($99 per tax filing).

  Axis 10 — feasibility
    → Cut MVP scope to ≤6 weeks. Drop features. Pick ONE core workflow.

  Axis 11 — confidence calibration
    → Lower confidence to match validation tag. [UNVALIDATED] → ≤65.
      [PARTIAL_EVIDENCE] → ≤80. [VALIDATED] → up to 95.

  Axis 13 — TAM
    → Pick a niche of 5k-50k buyers at $50-200/mo. If smaller niche,
      raise price to $500-2k/yr. If larger niche, drop price to
      consumer ($9-29 lifetime).

GENERAL RULES:
1. KEEP the core idea but fix the SPECIFIC concerns in critic_notes.
2. Update fields affected by the pivot: title, problem, solution,
   target_customer, why_now, revenue_model, unfair_advantage, mvp_stack,
   mvp_weeks, competitors, similar_projects, estimated_cost_usd,
   launch_steps, marketing_channels.
3. Reset `verdict` to "PASS" — the critic re-evaluates next round.
4. Reset `critic_notes` to "" — the next round fills it fresh.
5. Keep `lifecycle_stage` and `signal_sources` from the original.
6. NEVER leave practical-execution fields empty (similar_projects,
   estimated_cost_usd, launch_steps, marketing_channels) — Maksim
   acts on these.

Maksim's profile (unchanged from fresh-generation mode):
- Python AI engineer, ships ONLINE/web/SaaS solo MVPs in 2-6 weeks
- Stack: Python, FastAPI, LangGraph, RAG (optional), Docker, Azure, Postgres
- Industries: ANY — any field of human activity is fair game as long as
  the deliverable is an online/web product.
- Fine to ship: web SaaS, browser extensions, Telegram/Discord/Slack bots,
  PWAs, Telegram Mini Apps, RAG agents, scrapers, dashboards.
- Avoid: physical products, enterprise multi-month sales, heavy
  pre-day-one regulatory (FDA, banking license), pure offline services.
- Preserve `industry` tag unless the pivot fundamentally moves vertical.

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
