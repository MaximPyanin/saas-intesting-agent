"""Unit test for idea-industry-diversity selector (no LLM, no DB).

Verifies that `select_diverse_ideas` correctly picks 3 ideas from 3 distinct
`industry` values when given a mixed batch.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from oracle.agents.idea_generator import select_diverse_ideas

    # Simulate 8 raw ideas from the validator. Confidence is the tiebreaker.
    sample = [
        {"title": "A1", "industry": "ai_tools",        "confidence": 88, "verdict": "STRONG_PASS", "reflexion_rounds_passed": 2},
        {"title": "A2", "industry": "ai_tools",        "confidence": 85, "verdict": "PASS",        "reflexion_rounds_passed": 1},
        {"title": "B1", "industry": "health",          "confidence": 82, "verdict": "PASS",        "reflexion_rounds_passed": 2},
        {"title": "C1", "industry": "fitness_sport",   "confidence": 80, "verdict": "PASS",        "reflexion_rounds_passed": 1},
        {"title": "D1", "industry": "marketing_adtech","confidence": 78, "verdict": "PASS",        "reflexion_rounds_passed": 1},
        {"title": "E1", "industry": "ai_tools",        "confidence": 75, "verdict": "PASS",        "reflexion_rounds_passed": 1},
        {"title": "F1", "industry": "education",       "confidence": 60, "verdict": "PASS",        "reflexion_rounds_passed": 1},
        {"title": "G1", "industry": "creator_economy", "confidence": 90, "verdict": "KILL",        "reflexion_rounds_passed": 0},  # excluded
    ]

    picked = select_diverse_ideas(sample, target_count=3)
    print("Picked ideas:")
    for i, p in enumerate(picked, 1):
        print(f"  {i}. [{p['industry']}] {p['title']} — conf={p['confidence']}, verdict={p['verdict']}")

    industries = [p["industry"] for p in picked]
    titles = [p["title"] for p in picked]

    assert len(picked) == 3, f"expected 3 picks, got {len(picked)}"
    assert len(set(industries)) == 3, f"expected 3 distinct industries, got {industries}"
    assert "G1" not in titles, "KILL'ed idea should NOT be picked"
    assert picked[0]["title"] == "A1", f"first pick should be A1 (highest conf STRONG_PASS), got {picked[0]['title']}"

    print()
    print("✅ All assertions passed.")
    print(f"   Industries picked: {industries}")


if __name__ == "__main__":
    main()
