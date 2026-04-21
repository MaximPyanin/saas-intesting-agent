"""LangGraph StateGraph builder for ORACLE.

Topology (Step 1):

    START ──┬─→ scout       ──┐
            ├─→ market      ──┤
            ├─→ trend       ──┼─→ synthesizer ─→ idea_generator ─→ critic ──┐
            └─→ custom      ──┘                       ▲                      │
                                                      │                  (router)
                                                      │                      │
                                                      └────── continue ──────┤
                                                                             │
                                                                            exit
                                                                             │
                                                                             ▼
                                                              investment_analyzer
                                                                             │
                                                                             ▼
                                                                         validator
                                                                             │
                                                                             ▼
                                                                         formatter
                                                                             │
                                                                             ▼
                                                                            END

Fan-in semantics:
  - synthesizer waits for all 4 Layer-1 collectors (single super-step fan-in)
  - investment_analyzer is sequenced AFTER the Reflexion loop exit, NOT in
    parallel with idea_generator. Why: LangGraph fan-in only synchronizes
    parents that fire in the SAME super-step. The Reflexion loop spans many
    super-steps, so a parallel investment branch would race ahead and fire
    validator/formatter twice. Sequencing post-loop is the canonical fix and
    matches the spec's actual reflexion_loop code (which only operates on
    ideas, not investment scenarios).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    critic_node,
    custom_node,
    formatter_node,
    idea_generator_node,
    investment_analyzer_node,
    investment_critic_node,
    market_node,
    scout_node,
    should_continue_investment_reflexion,
    should_continue_reflexion,
    synthesizer_node,
    trend_node,
    validator_node,
)
from .state import OracleState


def build_graph() -> StateGraph:
    """Construct the ORACLE workflow graph (uncompiled).

    Compilation with the checkpointer happens in main.run_once() so the
    AsyncSqliteSaver context manager owns the lifetime of the DB connection.
    """
    g = StateGraph(OracleState)

    # Register all agent nodes
    g.add_node("scout", scout_node)
    g.add_node("market", market_node)
    g.add_node("trend", trend_node)
    g.add_node("custom", custom_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("idea_generator", idea_generator_node)
    g.add_node("investment_analyzer", investment_analyzer_node)
    g.add_node("investment_critic", investment_critic_node)
    g.add_node("critic", critic_node)
    g.add_node("validator", validator_node)
    g.add_node("formatter", formatter_node)

    # Layer 1 — parallel fan-out from START, fan-in to synthesizer
    for collector in ("scout", "market", "trend", "custom"):
        g.add_edge(START, collector)
        g.add_edge(collector, "synthesizer")

    # Layer 2a — synthesizer feeds the ideas branch (Reflexion loop)
    g.add_edge("synthesizer", "idea_generator")
    g.add_edge("idea_generator", "critic")

    # Reflexion loop: critic decides per round whether to refine or exit.
    # On exit we route to investment_analyzer (NOT validator) so the
    # downstream path is a single linear chain — no multi-super-step fan-in.
    g.add_conditional_edges(
        "critic",
        should_continue_reflexion,
        {"continue": "idea_generator", "exit": "investment_analyzer"},
    )

    # Layer 2b — investments run after the ideas loop completes, with their
    # OWN Reflexion loop (max 2 rounds): investment_analyzer → investment_critic
    # → (continue → investment_analyzer improve mode | exit → validator).
    g.add_edge("investment_analyzer", "investment_critic")
    g.add_conditional_edges(
        "investment_critic",
        should_continue_investment_reflexion,
        {"continue": "investment_analyzer", "exit": "validator"},
    )

    # Layer 3 — validator → formatter → END
    g.add_edge("validator", "formatter")
    g.add_edge("formatter", END)

    return g
