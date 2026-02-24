"""
examples/basic_reasoning.py
=============================
Minimal NeSy-Core example — pure symbolic reasoning, no neural backbone.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nesy.api.nesy_model import NeSyModel
from nesy.core.types import Predicate, SymbolicRule


def main():
    model = NeSyModel(domain="general", doubt_threshold=0.50)

    model.add_rule(SymbolicRule(
        id="mortal",
        antecedents=[Predicate("IsHuman", ("?x",))],
        consequents=[Predicate("IsMortal", ("?x",))],
        weight=1.0,
        description="All humans are mortal (Socrates syllogism)",
    ))
    model.add_rule(SymbolicRule(
        id="philosopher",
        antecedents=[Predicate("IsPhilosopher", ("?x",))],
        consequents=[Predicate("IsHuman", ("?x",))],
        weight=1.0,
        description="All philosophers are human",
    ))

    facts = {Predicate("IsPhilosopher", ("socrates",))}
    output = model.reason(facts=facts, neural_confidence=1.0)
    print(model.explain(output))
    assert "IsMortal" in output.answer, "Should derive: Socrates is mortal"
    print("✓ Basic reasoning test passed.")


if __name__ == "__main__":
    main()
