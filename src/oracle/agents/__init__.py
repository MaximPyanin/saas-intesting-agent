"""ORACLE real agent implementations.

Each module in this package implements one of the 10 LangGraph nodes from
Step 1. They are wired into the graph by re-export from `oracle.nodes`.

Implementation order matches the user spec:
    market               → Step 4  ← this step
    scout                → Step 5
    trend                → Step 6
    custom               → Step 7
    synthesizer          → Step 9
    idea_generator       → Step 10
    critic               → Step 11
    validator            → Step 12
    investment_analyzer  → Step 13
    formatter            → Step 9-13 (built up incrementally)
"""
