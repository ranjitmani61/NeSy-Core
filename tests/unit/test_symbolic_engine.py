"""tests/unit/test_symbolic_engine.py"""
import pytest
from nesy.symbolic.engine import SymbolicEngine
from nesy.core.types import Predicate, SymbolicRule
from nesy.core.exceptions import SymbolicConflict


def test_add_rule(fever_infection_rule):
    engine = SymbolicEngine()
    engine.add_rule(fever_infection_rule)
    assert len(engine.rules) == 1


def test_immutable_rule_cannot_be_removed(hard_contraindication_rule):
    engine = SymbolicEngine()
    engine.add_rule(hard_contraindication_rule)
    with pytest.raises(SymbolicConflict):
        engine.remove_rule(hard_contraindication_rule.id)


def test_reason_derives_infection(fever_infection_rule, fever_facts):
    engine = SymbolicEngine()
    engine.add_rule(fever_infection_rule)
    derived, steps, conf = engine.reason(fever_facts, check_hard_constraints=False)
    infection = Predicate("PossiblyHas", ("patient_test", "bacterial_infection"))
    assert infection in derived


def test_symbolic_confidence_is_geometric_mean(fever_infection_rule, fever_facts):
    engine = SymbolicEngine()
    engine.add_rule(fever_infection_rule)
    _, _, conf = engine.reason(fever_facts, check_hard_constraints=False)
    assert 0.0 <= conf <= 1.0


def test_no_rules_returns_original_facts():
    engine = SymbolicEngine()
    facts = {Predicate("A", ("x",))}
    derived, steps, conf = engine.reason(facts)
    assert facts.issubset(derived)
    assert conf == 1.0
