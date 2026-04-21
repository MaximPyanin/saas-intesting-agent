"""Pydantic output models for ORACLE agents.

These are the structured-output schemas that LLM agents will produce in
Steps 9-13 (synthesizer, idea_generator, investment_analyzer, critic,
validator). They are first NEEDED in Step 3 to give the Telegram bot a
typed contract for rendering cards.

Field names match the user spec literally; rendering-only extras are
clearly marked.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# Signal — a single piece of evidence collected from a source
# ============================================================================


class Signal(BaseModel):
    """One unit of input — a tweet, RSS item, HN story, market datum, etc.

    Producers (Step 4-7): scout / market / trend / custom collectors.
    Consumers: synthesizer (Step 9), topic_timeline tracker.
    """

    model_config = ConfigDict(extra="ignore")

    content: str
    source: str
    published_at: datetime
    freshness_score: int = Field(ge=0, le=100)
    """100=<2h, 75=<24h, 50=<72h, 0=stale"""
    is_breaking: bool = False
    """True if <2h old at collection time"""


# ============================================================================
# BusinessIdea — a SaaS / AI-tool / micro-SaaS opportunity
# ============================================================================


class BusinessIdea(BaseModel):
    """A vetted business opportunity.

    Producer: idea_generator (Step 10), refined by critic (Step 11),
              validated by validator (Step 12).
    Consumer: Telegram bot (Step 3 mock, Step 14 learning system).
    """

    model_config = ConfigDict(extra="ignore")

    # ----- spec literal -----
    title: str
    one_liner: str
    problem: str
    solution: str
    target_customer: str
    why_now: str
    """Must link to today's signals — temporal grounding"""
    revenue_model: str
    mvp_weeks: int = Field(ge=1, le=12)
    mvp_stack: list[str]
    confidence: int = Field(ge=0, le=100)
    lifecycle_stage: str
    """EMERGING | GROWING | PEAK | DECLINING"""
    window_months: int
    """How many months before opportunity closes"""
    competitors: list[str] = Field(default_factory=list)
    """Filled by validator (Step 12)"""
    verdict: str = "PASS"
    """KILL | WEAKEN | PASS | STRONG_PASS — assigned by critic"""

    # ----- practical execution fields (Maksim asked to add these) -----
    similar_projects: list[str] = Field(default_factory=list)
    """2-4 closest existing projects/products. Format per item:
    'Name — 1-line what-they-do (URL if known)'. NOT the same as `competitors`:
    these are reference points to STUDY (their pricing, UX, gaps), not blockers."""
    estimated_cost_usd: str = ""
    """Rough MVP build + first-2-month run cost. Example:
    '$300-600 (infra $40/mo + API $80/mo + landing $0 + domain $12/yr)'."""
    launch_steps: list[str] = Field(default_factory=list)
    """Ordered concrete next-48h-to-4-weeks steps from zero to paying users.
    Example item: 'Week 1: Carrd landing + waitlist + $50 Meta Ads test'.
    Aim for 4-7 steps."""
    marketing_channels: list[str] = Field(default_factory=list)
    """2-5 SPECIFIC channels where THIS target_customer hangs out.
    Example: 'r/indiehackers cold post', 'ProductHunt Tue launch',
    'Indie Hackers Twitter DMs', 'AI Engineer meetup Warsaw'."""

    # ----- rendering extras (used by Telegram cards) -----
    unfair_advantage: str = ""
    """Maksim's specific edge for this idea"""
    signal_sources: list[str] = Field(default_factory=list)
    """Short labels of signals that triggered the idea"""
    reflexion_rounds_passed: int = 0
    """How many critique rounds the idea survived (1..3)"""

    # ----- Reflexion loop scratch space (Step 11) -----
    critic_notes: str = ""
    """Critic's most recent reasoning. Used by idea_generator improve mode
    on the next round. Cleared on KILL/PASS final verdicts."""

    # ----- Validator outputs (Step 12) -----
    validation_notes: str = ""
    """Validator's reasoning after web search — what was confirmed,
    what was contradicted, what new competitors surfaced."""
    market_size_note: str = ""
    """Brief market sizing inferred from web search results."""
    validator_verdict: str = ""
    """VALIDATED | CONCERNS | REJECTED — outside-the-LLM reality check."""


# ============================================================================
# InvestmentSignal — trade-able signal for stocks / crypto / commodities / forex
# ============================================================================


class InvestmentSignal(BaseModel):
    """An asset-level scenario produced by the investment analyzer.

    Producer: investment_analyzer (Step 13).
    Consumer: Telegram bot (Step 3 mock).
    """

    model_config = ConfigDict(extra="ignore")

    asset: str
    """e.g. 'BTC', 'S&P500', 'XAU/USD', 'NVDA'"""
    signal_type: str
    """e.g. 'Macro Momentum', 'Sector Rotation', 'Defensive Hedge'"""
    price: float
    change_24h: float
    """percentage, signed"""
    change_7d: float
    """percentage, signed"""
    strength: int = Field(ge=1, le=10)
    timeframe: str
    """e.g. '3-6 months'"""
    bull_scenario: str
    bull_prob: int = Field(ge=0, le=100)
    bull_trigger: str
    bear_scenario: str
    bear_prob: int = Field(ge=0, le=100)
    bear_trigger: str
    key_events: list[str] = Field(default_factory=list)
    geopolitical_note: str = ""

    # ----- trader-brief fields Maksim explicitly asked for -----
    action: str = "WAIT"
    """One of: BUY | ADD | HOLD | TRIM | SELL | WAIT | AVOID."""
    do_now: str = ""
    """Single imperative headline: what Maksim should do today/this week.
    Written in Russian-friendly imperative, bold-worthy. Examples:
    '💱 Меняй USD → PLN сегодня',
    '🥇 Держишь слиток — не трогаешь',
    '⏸ Жди пока BTC сломает $76k с объёмом — не гонись',
    '🟢 Открывай позицию AVAV после пробоя $215'."""
    how_to_execute: str = ""
    """Concrete execution instruction — broker + venue + size + price.
    Example: 'Wise/Revolut USD→PLN, меняй 5000 USD из 15000 cash по курсу
    3.57-3.60'. For stocks: 'IBKR market buy 10 AVAV shares after open'.
    For crypto: 'Binance limit BTC @ $68,000 size 0.05'. Empty if action
    is WAIT/HOLD/AVOID (then just fill do_now)."""
    why_now_short: str = ""
    """1 sentence chain-of-causation why TODAY not next week. Example:
    'Ормуз открывается → нефть падает → инфляция снижается → PLN укрепляется
    → USD/PLN пойдёт вниз быстро.'"""
    post_action: str = ""
    """What happens after the primary action. Example: 'Сидишь в PLN, ждёшь
    пока сделка США-Иран закроется, потом покупаешь USD обратно дешевле +
    часть в GLD/ITA.' Empty for pure HOLD/AVOID."""
    is_portfolio_holding: bool = False
    """True if this asset is CURRENTLY in Maksim's portfolio. Used by the
    Telegram renderer to show a '💼 Your position' line and by the LLM to
    prioritize held assets (2/3 of signals must cover held assets when
    portfolio is non-empty)."""
    action_reason: str = ""
    """DEPRECATED — was a single reason line; now replaced by do_now +
    why_now_short + post_action. Kept for backward compatibility with any
    existing checkpoints; new runs should leave it empty."""

    # ----- rendering extras -----
    signal_age_hours: int = 0
    """Hours since signal was first surfaced"""

    # ----- Reflexion loop fields (investment critic, Step 13.5) -----
    verdict: str = "PASS"
    """KILL | WEAKEN | PASS | STRONG_PASS — assigned by investment_critic."""
    critic_notes: str = ""
    """Investment critic's reasoning if WEAKEN — used by improve mode on
    next round. Same pattern as BusinessIdea.critic_notes."""
    reflexion_rounds_passed: int = 0
    """How many investment critique rounds this scenario survived (1..2)."""
