"""LangGraph nodes for the ORACLE graph.

Acts as the wiring layer between graph.py and the per-agent implementations
in `oracle.agents`. Real agents are imported here and re-exported under the
name graph.py expects. Placeholders remain for nodes whose implementations
are still ahead in the roadmap.

Replacement progress:
    scout                → Step 5  (Reuters/FT/ISW RSS)             ✅ real
    market               → Step 4  (yfinance + CoinGecko + FRED)    ✅ real
    trend                → Step 6  (HN/PH/Reddit/GitHub/VC RSS)     ✅ real
    custom               → Step 7  (Telethon/YouTube/website RSS)   ✅ real
    synthesizer          → Step 9  (gpt-4o cross-signal patterns)   ✅ real
    idea_generator       → Step 10 (fresh) + Step 11 (improve)      ✅ real
    critic               → Step 11 (Reflexion verdicts)             ✅ real
    validator            → Step 12 (web search + price re-check)    ✅ real
    investment_analyzer  → Step 13 (gpt-4o scenarios per asset)     ✅ real
    formatter            → built up in Steps 9-13                   [placeholder]
"""

from __future__ import annotations

import logging
from typing import Literal

from .agents.critic import critic_node                            # Step 11
from .agents.custom import custom_node                            # Step 7
from .agents.idea_generator import idea_generator_node            # Step 10/11
from .agents.investment_analyzer import investment_analyzer_node  # Step 13 + 13.5
from .agents.investment_critic import investment_critic_node      # Step 13.5
from .agents.market import market_node                            # Step 4
from .agents.scout import scout_node                              # Step 5
from .agents.synthesizer import synthesizer_node                  # Step 9
from .agents.trend import trend_node                              # Step 6
from .agents.validator import validator_node                      # Step 12
from .state import OracleState

log = logging.getLogger(__name__)

# Reflexion loop bound — matches user spec ("Max 3 Reflexion iterations").
# Will move to Settings.reflexion_max_rounds when the bot/scheduler need it.
REFLEXION_MAX_ROUNDS = 3

# Investment Reflexion is cheaper + scenarios are simpler than ideas, so 2
# rounds is enough to catch vague do_now / missing executions / portfolio
# coverage violations without doubling cost.
INVEST_REFLEXION_MAX = 2


# ---------- Layer 1: parallel data collection ----------

# All four collectors are now real:
# scout_node       imported from .agents.scout
# market_node      imported from .agents.market
# trend_node       imported from .agents.trend
# custom_node      imported from .agents.custom


# ---------- Layer 2a: synthesis ----------

# synthesizer_node imported from .agents.synthesizer (Step 9)


# ---------- Layer 2b: reasoning fan-out ----------

# idea_generator_node imported from .agents.idea_generator (Step 10/11)


# investment_analyzer_node imported from .agents.investment_analyzer (Step 13)


# ---------- Layer 2c: Reflexion critic ----------

# critic_node imported from .agents.critic (Step 11)


# ---------- Layer 2d: validation ----------

# validator_node imported from .agents.validator (Step 12)


# ---------- Layer 3: format ----------

async def formatter_node(state: OracleState) -> dict:
    """Final formatter — applies industry-diversity selection on ideas.

    Before this step, `state["validated"]` contains ALL surviving ideas (5-10).
    Maksim wants exactly 3 ideas per digest, each from a DIFFERENT industry.
    But the EXTRA validated ideas (4-7 more) are also useful — they're
    surfaced via the /more command if Maksim wants another round from
    different buckets. So we expose both `ideas` and `ideas_extra`.

    Investment scenarios pass through unchanged (the per-holding rule is
    enforced inside investment_analyzer_node itself).
    """
    from .agents.idea_generator import select_diverse_ideas  # noqa: PLC0415

    validated = state.get("validated", []) or []
    top_3 = select_diverse_ideas(validated, target_count=3)
    # Stable id-based set difference so the extras don't repeat the top-3
    picked_ids = {id(i) for i in top_3}
    extras = [i for i in validated if id(i) not in picked_ids]

    log.info(
        "formatter: %d validated → %d diverse ideas + %d extras; %d invest scenarios",
        len(validated),
        len(top_3),
        len(extras),
        len(state.get("investment_scenarios", []) or []),
    )

    return {
        "final_digest": {
            "ideas": top_3,
            "ideas_extra": extras,
            "investments": state.get("investment_scenarios", []),
        }
    }


# ---------- Reflexion router (pure function) ----------

def should_continue_reflexion(state: OracleState) -> Literal["continue", "exit"]:
    """Pure router: read state, decide whether to loop or exit.

    Reflexion semantics (Step 11):
      - Hard cap at REFLEXION_MAX_ROUNDS critic invocations
      - Exit EARLY if no ideas have verdict 'WEAKEN' — no point looping if
        everything is already PASS/STRONG_PASS or KILLed
      - Exit EARLY if raw_ideas is empty (everything got killed)
      - Counter is incremented INSIDE critic_node (state mutations live in nodes)

    Routing must be a side-effect-free pure function so checkpoint resume picks
    the same branch on replay.
    """
    if state.get("reflexion_round", 0) >= REFLEXION_MAX_ROUNDS:
        return "exit"

    raw_ideas = state.get("raw_ideas", []) or []
    if not raw_ideas:
        return "exit"  # nothing left to refine

    has_weakened = any(i.get("verdict") == "WEAKEN" for i in raw_ideas)
    if not has_weakened:
        return "exit"  # all surviving ideas are PASS/STRONG_PASS

    return "continue"


def should_continue_investment_reflexion(state: OracleState) -> Literal["continue", "exit"]:
    """Investment-side router. Mirror of should_continue_reflexion but for
    investment_scenarios. Exits when:
      - round counter hits INVEST_REFLEXION_MAX
      - no scenarios survive (everything KILLed)
      - no WEAKEN verdicts remain (all PASS/STRONG_PASS)
    """
    if state.get("investment_reflexion_round", 0) >= INVEST_REFLEXION_MAX:
        return "exit"

    scenarios = state.get("investment_scenarios", []) or []
    if not scenarios:
        return "exit"

    has_weakened = any(s.get("verdict") == "WEAKEN" for s in scenarios)
    if not has_weakened:
        return "exit"

    return "continue"
