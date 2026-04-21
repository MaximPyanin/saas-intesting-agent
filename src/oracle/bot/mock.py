"""Mock data generators for ORACLE bot Step 3 testing.

These deterministic fixtures let us validate every card template, every
button callback, and the feedback DB write path BEFORE wiring real LLM
agents in Steps 9-13.

Replace with calls to the real digest pipeline once Step 9+ is online.
"""

from __future__ import annotations

from ..models import BusinessIdea, InvestmentSignal


def mock_business_ideas() -> list[BusinessIdea]:
    """Three deterministic business ideas covering different lifecycle stages."""
    return [
        BusinessIdea(
            title="Agentic CSV Cleaner",
            one_liner="LLM agent that cleans messy CSV files via natural language",
            problem="Data analysts waste 40% of their time fixing CSVs by hand — column drift, mixed types, inconsistent dates.",
            solution="Upload a CSV → describe what's wrong in plain English → agent fixes the file and explains every change.",
            target_customer="Solo data analysts and BI teams in companies under 50 people",
            why_now="GPT-4o-mini cost dropped 80% in Q1 2026; pandas + LLM agents are now cheap enough to run per-file at <$0.05 each.",
            revenue_model="$29/month per seat (5 free files/month free tier)",
            mvp_weeks=4,
            mvp_stack=["Python", "FastAPI", "pandas", "LangChain", "DuckDB"],
            confidence=82,
            lifecycle_stage="GROWING",
            window_months=8,
            competitors=["Akkio", "Rows.com"],
            verdict="STRONG_PASS",
            unfair_advantage="Maksim's RAG expertise → schema-aware fixes that competitors lack",
            signal_sources=["r/dataisbeautiful (12 posts)", "HN front page", "ProductHunt #4"],
            reflexion_rounds_passed=3,
        ),
        BusinessIdea(
            title="Vertical Voice Assistant Builder",
            one_liner="No-code platform to build voice agents for niche professional services",
            problem="Voice AI is mostly horizontal — no ready-made assistants for dental clinics, vet offices, small law firms.",
            solution="Pick a vertical → import workflows from a template library → deploy a phone agent in 1 day.",
            target_customer="SMB owners in vertical service businesses with under 20 staff",
            why_now="ElevenLabs Turbo + Vapi API now enable real-time voice for $0.10/min — economically viable for SMBs.",
            revenue_model="$199/month flat per business + $50 setup fee",
            mvp_weeks=6,
            mvp_stack=["Python", "Vapi", "Twilio", "Postgres", "FastAPI"],
            confidence=71,
            lifecycle_stage="EMERGING",
            window_months=12,
            competitors=["Vapi (platform)", "Bland.ai"],
            verdict="PASS",
            unfair_advantage="Pre-built workflows for 5 chosen verticals beat horizontal players on time-to-deploy",
            signal_sources=["TechCrunch (a16z funding)", "ProductHunt #2 launch"],
            reflexion_rounds_passed=2,
        ),
        BusinessIdea(
            title="LLM Eval Suite for Production Agents",
            one_liner="Self-hosted dashboard tracking accuracy, cost, and latency of every agent step",
            problem="Teams ship LangGraph/CrewAI agents to production but have no observability — flying blind on accuracy and cost.",
            solution="Drop-in Python decorator → metrics + traces + regression tests + alerts.",
            target_customer="AI engineers shipping production agents with $1k-100k/month spend",
            why_now="Agent stacks are hitting production at scale right now — teams just discovered they're flying blind.",
            revenue_model="OSS core (free) + $99/month for hosted dashboard with team features",
            mvp_weeks=5,
            mvp_stack=["Python", "FastAPI", "DuckDB", "OpenTelemetry", "Next.js"],
            confidence=78,
            lifecycle_stage="GROWING",
            window_months=10,
            competitors=["Langfuse", "Helicone", "Arize Phoenix"],
            verdict="STRONG_PASS",
            unfair_advantage="Self-hosted-first (Langfuse pushes cloud) + LangGraph-native trace adapters",
            signal_sources=["3 HN front-page posts on agent observability", "/r/LocalLLaMA discussion"],
            reflexion_rounds_passed=3,
        ),
    ]


def mock_investment_signals() -> list[InvestmentSignal]:
    """Three deterministic investment signals across asset classes."""
    return [
        InvestmentSignal(
            asset="BTC",
            signal_type="Macro Momentum",
            price=72450.0,
            change_24h=2.3,
            change_7d=8.1,
            strength=7,
            timeframe="3-6 months",
            bull_scenario="ETF inflows accelerating, halving cycle support holding, Fed pivot priced in for June.",
            bull_prob=65,
            bull_trigger="BTC > $76k OR daily ETF inflow > $500M",
            bear_scenario="Risk-off shock from geopolitical event drags BTC -15% to $61k.",
            bear_prob=35,
            bear_trigger="VIX > 28 OR major equity pullback (S&P -4% in a week)",
            key_events=["Fed FOMC 2026-04-30", "Q1 earnings ends 2026-05-08"],
            geopolitical_note="Middle East tensions cap gold rally → BTC absorbs risk-on flows",
            signal_age_hours=4,
        ),
        InvestmentSignal(
            asset="Gold (XAU/USD)",
            signal_type="Defensive Hedge",
            price=2730.50,
            change_24h=-0.4,
            change_7d=1.2,
            strength=6,
            timeframe="6-12 months",
            bull_scenario="Central bank buying continues, dollar softens on Fed cuts, real rates fall.",
            bull_prob=55,
            bull_trigger="DXY < 102 OR new central bank disclosed buying",
            bear_scenario="Sharp dollar rally on hawkish surprise; brief 5-7% pullback to $2,540.",
            bear_prob=45,
            bear_trigger="DXY > 107 OR hawkish Fed dot plot revision",
            key_events=["Fed FOMC 2026-04-30", "ECB meeting 2026-04-17"],
            geopolitical_note="Demand from Asia central banks structurally elevated through 2027",
            signal_age_hours=12,
        ),
        InvestmentSignal(
            asset="NVDA",
            signal_type="Sector Rotation",
            price=142.30,
            change_24h=-1.8,
            change_7d=-3.4,
            strength=5,
            timeframe="3-6 months",
            bull_scenario="Q1 earnings beat consensus, hyperscaler datacenter capex stays elevated.",
            bull_prob=50,
            bull_trigger="Q1 EPS > $0.85 OR hyperscaler capex guide raised",
            bear_scenario="Hyperscaler capex normalization rumors → -10-15% pullback to $120s.",
            bear_prob=50,
            bear_trigger="Major hyperscaler cuts FY26 datacenter capex guide",
            key_events=["NVDA earnings 2026-05-21", "GTC keynote 2026-05-15"],
            geopolitical_note="China advanced-chip export controls creating both demand pull-forward and overhang",
            signal_age_hours=8,
        ),
    ]


def mock_morning_data() -> dict:
    """Deterministic morning brief fixture."""
    return {
        "date": "2026-04-07",
        "breaking_emoji": "🔴",
        "top_3_events": (
            "• Fed minutes hint at pause; 10Y yields drop 12bps overnight\n"
            "• ECB holds 4.0%, signals June cut\n"
            "• OpenAI ships GPT-5 mini at 1/3 the cost of GPT-4o"
        ),
        "gold": "2,730",
        "gold_pct": "+0.2%",
        "btc": "72,450",
        "btc_pct": "+2.3%",
        "sp500": "5,420",
        "sp500_pct": "+0.6%",
        "oil": "78.40",
        "oil_pct": "-0.8%",
        "vix": "15.2",
        "key_signal_one_liner": "GPT-5 mini at 1/3 cost → wave of new agentic SaaS opportunities opening",
    }


def mock_sources_data() -> dict:
    """Deterministic sources management fixture."""
    return {
        "total": 5,
        "ideas_count": 3,
        "ideas_sources_list": "• @bensbites\n• @startupoftheday\n• @ycombinator",
        "inv_count": 1,
        "inv_sources_list": "• @markets",
        "geo_count": 1,
        "geo_sources_list": "• @iswnews",
    }
