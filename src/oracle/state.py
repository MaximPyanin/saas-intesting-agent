"""LangGraph state schema for ORACLE.

The state is the contract between every agent in the pipeline. Each field has a
single producer node and one or more consumer nodes. See the table in the Step 1
plan for the full producer/consumer mapping.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class OracleState(TypedDict):
    # Layer 1 outputs — written by collectors (parallel)
    scout_signals: list[dict]
    market_data: dict
    trend_signals: list[dict]
    custom_signals: list[dict]
    # Layer 2 — reasoning
    synthesized: list[dict]
    raw_ideas: list[dict]
    investment_scenarios: list[dict]
    reflexion_round: int          # Reflexion loop counter for ideas (0..REFLEXION_MAX_ROUNDS)
    investment_reflexion_round: int  # Separate counter for investments (0..INVEST_REFLEXION_MAX)
    surviving_ideas: list[dict]
    validated: list[dict]
    # Layer 3 — output
    final_digest: dict
    # Cross-cutting — accumulated by all nodes via operator.add
    errors: Annotated[list[str], operator.add]
