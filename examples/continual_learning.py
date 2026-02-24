"""
examples/continual_learning.py
================================
Demonstrates NeSy-Core continual learning:
  - Learn Task A
  - Learn Task B (without forgetting Task A)
  - Verify both tasks still work
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nesy.api.nesy_model import NeSyModel
from nesy.core.types import Predicate, SymbolicRule


def main():
    model = NeSyModel(domain="general")

    # Task A: Animals
    model.learn(SymbolicRule(
        id="bird_flies",
        antecedents=[Predicate("IsBird", ("?x",))],
        consequents=[Predicate("CanFly", ("?x",))],
        weight=0.85,
        description="Birds can fly",
    ))

    # Verify Task A works
    output_a = model.reason({Predicate("IsBird", ("tweety",))})
    print(f"Task A — Bird test: {output_a.answer}")
    assert "CanFly" in output_a.answer

    # Task B: Medical (new task)
    model.learn(SymbolicRule(
        id="fever_needs_test",
        antecedents=[Predicate("HasSymptom", ("?p", "fever"))],
        consequents=[Predicate("RequiresTest", ("?p", "blood_test"))],
        weight=0.80,
        description="Fever requires blood test",
    ))

    # Verify Task B works
    output_b = model.reason({Predicate("HasSymptom", ("patient_1", "fever"))})
    print(f"Task B — Fever test: {output_b.answer}")
    assert "RequiresTest" in output_b.answer

    # Verify Task A still works (no forgetting in symbolic layer)
    output_a2 = model.reason({Predicate("IsBird", ("penguin",))})
    print(f"Task A still works: {output_a2.answer}")
    assert "CanFly" in output_a2.answer

    # Add permanent anchor
    model.learn(SymbolicRule(
        id="alive_needs_oxygen",
        antecedents=[Predicate("IsAlive", ("?x",))],
        consequents=[Predicate("NeedsOxygen", ("?x",))],
        weight=1.0,
        description="All living things need oxygen",
    ), make_anchor=True)

    print(f"\nTotal rules: {model.rule_count}")
    print(f"Anchored:    {model.anchored_rules}")
    print("✓ Continual learning test passed.")


if __name__ == "__main__":
    main()
