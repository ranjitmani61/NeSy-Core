"""
tests/unit/test_logic.py
========================
Tests for symbolic logic primitives: unification, resolution, Betti number.
"""
import pytest
from nesy.symbolic.logic import (
    unify, apply_substitution, resolve_clauses,
    is_satisfiable, betti_0, forward_chain,
)
from nesy.core.types import Predicate, SymbolicRule


class TestUnification:
    def test_unify_identical_ground(self):
        p1 = Predicate("HasSymptom", ("patient_1", "fever"))
        p2 = Predicate("HasSymptom", ("patient_1", "fever"))
        assert unify(p1, p2) == {}

    def test_unify_variable_to_ground(self):
        p1 = Predicate("HasSymptom", ("?p", "fever"))
        p2 = Predicate("HasSymptom", ("patient_1", "fever"))
        theta = unify(p1, p2)
        assert theta == {"?p": "patient_1"}

    def test_unify_ground_conflict(self):
        p1 = Predicate("HasSymptom", ("patient_1", "fever"))
        p2 = Predicate("HasSymptom", ("patient_1", "cough"))
        assert unify(p1, p2) is None

    def test_unify_name_mismatch(self):
        p1 = Predicate("HasSymptom", ("?p", "fever"))
        p2 = Predicate("HasAllergy",  ("?p", "fever"))
        assert unify(p1, p2) is None

    def test_apply_substitution(self):
        p = Predicate("HasSymptom", ("?p", "?s"))
        theta = {"?p": "patient_1", "?s": "fever"}
        result = apply_substitution(p, theta)
        assert result == Predicate("HasSymptom", ("patient_1", "fever"))


class TestResolution:
    def test_resolution_finds_contradiction(self):
        # {P} and {¬P} → empty resolvent (contradiction)
        from nesy.symbolic.logic import negate_predicate
        p = Predicate("HasDisease", ("patient_1",))
        clauses = [
            frozenset([p]),
            frozenset([negate_predicate(p)])
        ]
        assert is_satisfiable(clauses) is False

    def test_satisfiable_independent_clauses(self):
        p1 = Predicate("HasSymptom", ("p1", "fever"))
        p2 = Predicate("HasSymptom", ("p1", "cough"))
        clauses = [frozenset([p1]), frozenset([p2])]
        assert is_satisfiable(clauses) is True


class TestBetti:
    def test_betti_connected(self):
        preds = [
            Predicate("HasSymptom", ("patient_1", "fever")),
            Predicate("RequiresTest", ("patient_1", "blood_test")),
        ]
        assert betti_0(preds) == 1

    def test_betti_disconnected(self):
        preds = [
            Predicate("HasSymptom", ("patient_1", "fever")),
            Predicate("DrugInteraction", ("drug_a", "drug_b")),
        ]
        # Two components: patient_1 cluster and drug cluster
        assert betti_0(preds) >= 2

    def test_betti_empty(self):
        assert betti_0([]) == 0


class TestForwardChaining:
    def test_basic_chain(self):
        rules = [
            SymbolicRule(
                id="r1",
                antecedents=[Predicate("A", ("?x",))],
                consequents=[Predicate("B", ("?x",))],
                weight=1.0,
            )
        ]
        initial = {Predicate("A", ("node_1",))}
        derived, trace = forward_chain(initial, rules)
        assert Predicate("B", ("node_1",)) in derived

    def test_chain_stops_at_fixpoint(self):
        rules = [
            SymbolicRule(
                id="r1",
                antecedents=[Predicate("A", ("?x",))],
                consequents=[Predicate("B", ("?x",))],
                weight=1.0,
            )
        ]
        initial = {Predicate("C", ("x",))}  # C doesn't trigger r1
        derived, trace = forward_chain(initial, rules)
        assert derived == initial
        assert len(trace) == 0
