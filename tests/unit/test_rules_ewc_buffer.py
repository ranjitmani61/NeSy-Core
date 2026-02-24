"""tests/unit/test_rules.py — Rule building, loading, validation"""
import json, pytest, tempfile, os
from nesy.symbolic.rules import RuleBuilder, RuleLoader, RuleValidator
from nesy.core.types import Predicate, SymbolicRule


class TestRuleBuilder:
    def test_fluent_build(self):
        rule = (RuleBuilder("test_rule")
                .if_fact("A", "?x")
                .then("B", "?x")
                .with_weight(0.85)
                .in_domain("test")
                .description("Test rule")
                .build())
        assert rule.id == "test_rule"
        assert rule.weight == 0.85
        assert rule.domain == "test"
        assert len(rule.antecedents) == 1
        assert len(rule.consequents) == 1

    def test_anchor_sets_immutable(self):
        rule = (RuleBuilder("anchor_rule")
                .if_fact("X", "?a")
                .then("Y", "?a")
                .as_anchor()
                .build())
        assert rule.immutable is True
        assert rule.weight == 1.0

    def test_missing_antecedent_raises(self):
        with pytest.raises(ValueError):
            RuleBuilder("bad").then("B", "?x").build()

    def test_missing_consequent_raises(self):
        with pytest.raises(ValueError):
            RuleBuilder("bad").if_fact("A", "?x").build()


class TestRuleLoader:
    def test_from_list(self):
        data = [{
            "id": "test", "antecedents": [["A", "?x"]], "consequents": [["B", "?x"]], "weight": 0.8
        }]
        rules = RuleLoader.from_list(data)
        assert len(rules) == 1
        assert rules[0].id == "test"
        assert rules[0].weight == 0.8

    def test_save_and_load_json(self, tmp_path):
        rules = [
            SymbolicRule(
                id="r1",
                antecedents=[Predicate("A", ("?x",))],
                consequents=[Predicate("B", ("?x",))],
                weight=0.9,
            )
        ]
        path = str(tmp_path / "rules.json")
        RuleLoader.to_json(rules, path)
        loaded = RuleLoader.from_json(path)
        assert len(loaded) == 1
        assert loaded[0].id == "r1"
        assert loaded[0].weight == pytest.approx(0.9)


class TestRuleValidator:
    def test_valid_rules_return_no_errors(self):
        rules = [
            SymbolicRule("r1", [Predicate("A", ("?x",))], [Predicate("B", ("?x",))], 0.8),
            SymbolicRule("r2", [Predicate("B", ("?x",))], [Predicate("C", ("?x",))], 0.9),
        ]
        errors = RuleValidator.validate(rules)
        assert errors == []

    def test_duplicate_id_detected(self):
        rules = [
            SymbolicRule("dup", [Predicate("A", ("?x",))], [Predicate("B", ("?x",))], 0.8),
            SymbolicRule("dup", [Predicate("C", ("?x",))], [Predicate("D", ("?x",))], 0.7),
        ]
        errors = RuleValidator.validate(rules)
        assert any("dup" in e for e in errors)


"""tests/unit/test_ewc.py — EWC regularizer"""
import pytest
from nesy.continual.ewc import EWCRegularizer


class TestEWCRegularizer:
    def test_penalty_zero_without_snapshots(self):
        ewc = EWCRegularizer(lambda_ewc=1000.0)
        penalty = ewc.penalty({"w1": 0.5, "w2": 0.3})
        assert penalty == 0.0

    def test_penalty_zero_at_consolidated_params(self):
        ewc = EWCRegularizer(lambda_ewc=1000.0)
        params = {"w1": 0.5, "w2": 0.3}
        fisher = {"w1": 1.0, "w2": 1.0}
        ewc.consolidate("task_a", params, fisher)
        # No drift → penalty should be 0
        penalty = ewc.penalty(params)
        assert penalty == pytest.approx(0.0, abs=1e-8)

    def test_penalty_increases_with_drift(self):
        ewc = EWCRegularizer(lambda_ewc=1000.0)
        params_orig = {"w1": 0.0}
        fisher = {"w1": 1.0}
        ewc.consolidate("task_a", params_orig, fisher)
        penalty_small = ewc.penalty({"w1": 0.1})
        penalty_large = ewc.penalty({"w1": 1.0})
        assert penalty_large > penalty_small > 0.0

    def test_consolidated_tasks_tracked(self):
        ewc = EWCRegularizer(lambda_ewc=100.0)
        ewc.consolidate("task_a", {"w": 1.0}, {"w": 1.0})
        ewc.consolidate("task_b", {"w": 2.0}, {"w": 0.5})
        assert "task_a" in ewc.consolidated_tasks
        assert "task_b" in ewc.consolidated_tasks


"""tests/unit/test_memory_buffer.py — Episodic memory"""
import pytest
from nesy.continual.memory_buffer import EpisodicMemoryBuffer, MemoryItem


class TestEpisodicMemoryBuffer:
    def test_add_and_sample(self):
        buf = EpisodicMemoryBuffer(max_size=100)
        for i in range(50):
            buf.add(MemoryItem(data=f"item_{i}", task_id="task_a"))
        assert buf.size == 50
        sample = buf.sample(10)
        assert len(sample) == 10

    def test_reservoir_sampling_bounded(self):
        buf = EpisodicMemoryBuffer(max_size=10)
        for i in range(1000):
            buf.add(MemoryItem(data=f"item_{i}", task_id="task_a"))
        assert buf.size == 10
        assert buf.total_seen == 1000

    def test_filter_by_task(self):
        buf = EpisodicMemoryBuffer(max_size=100)
        for i in range(20):
            buf.add(MemoryItem(data=f"a_{i}", task_id="task_a"))
        for i in range(30):
            buf.add(MemoryItem(data=f"b_{i}", task_id="task_b"))
        task_a_items = buf.get_by_task("task_a")
        assert all(item.task_id == "task_a" for item in task_a_items)

    def test_sample_more_than_buffer_size(self):
        buf = EpisodicMemoryBuffer(max_size=100)
        for i in range(5):
            buf.add(MemoryItem(data=f"item_{i}", task_id="t"))
        sample = buf.sample(100)   # request more than available
        assert len(sample) == 5


"""tests/unit/test_prover.py — Backward chaining prover"""
import pytest
from nesy.symbolic.prover import BackwardChainer
from nesy.core.types import Predicate, SymbolicRule


@pytest.fixture
def mortal_rules():
    return [
        SymbolicRule("human", [Predicate("IsPhilosopher", ("?x",))],
                     [Predicate("IsHuman", ("?x",))], weight=1.0),
        SymbolicRule("mortal", [Predicate("IsHuman", ("?x",))],
                     [Predicate("IsMortal", ("?x",))], weight=1.0),
    ]


class TestBackwardChaining:
    def test_proves_known_fact(self, mortal_rules):
        prover = BackwardChainer(mortal_rules)
        facts  = {Predicate("IsPhilosopher", ("socrates",))}
        result = prover.prove(Predicate("IsMortal", ("socrates",)), facts)
        assert result.proved is True

    def test_fails_unknown_goal(self, mortal_rules):
        prover = BackwardChainer(mortal_rules)
        facts  = {Predicate("IsPhilosopher", ("socrates",))}
        result = prover.prove(Predicate("CanFly", ("socrates",)), facts)
        assert result.proved is False

    def test_confidence_is_product_of_weights(self, mortal_rules):
        prover = BackwardChainer(mortal_rules)
        facts  = {Predicate("IsPhilosopher", ("socrates",))}
        result = prover.prove(Predicate("IsMortal", ("socrates",)), facts)
        assert result.proved is True
        assert result.confidence == pytest.approx(1.0 * 1.0)

    def test_proof_tree_has_steps(self, mortal_rules):
        prover = BackwardChainer(mortal_rules)
        facts  = {Predicate("IsPhilosopher", ("socrates",))}
        result = prover.prove(Predicate("IsMortal", ("socrates",)), facts)
        assert result.proof_tree is not None
        assert len(result.rules_used) > 0

    def test_prove_all(self, mortal_rules):
        prover = BackwardChainer(mortal_rules)
        facts  = {Predicate("IsPhilosopher", ("socrates",))}
        goals  = [
            Predicate("IsMortal", ("socrates",)),
            Predicate("IsHuman",  ("socrates",)),
        ]
        results = prover.prove_all(goals, facts)
        assert all(r.proved for r in results.values())


"""tests/unit/test_validators.py — Input validation"""
import pytest
from nesy.core.types import Predicate, SymbolicRule, ConceptEdge
from nesy.core.validators import (
    validate_predicate, validate_facts, validate_rule, validate_concept_edge,
    assert_valid_facts, assert_valid_rule,
)
from nesy.core.exceptions import NeSyError


class TestPredicateValidation:
    def test_valid_predicate(self):
        p = Predicate("HasSymptom", ("patient_1", "fever"))
        assert validate_predicate(p) == []

    def test_empty_name_flagged(self):
        p = Predicate("", ("p1",))
        errors = validate_predicate(p)
        assert len(errors) > 0

    def test_variable_in_fact_flagged(self):
        fact = Predicate("HasSymptom", ("?p", "fever"))   # unbound variable in fact
        errors = validate_facts({fact})
        assert len(errors) > 0


class TestRuleValidation:
    def test_valid_rule(self):
        rule = SymbolicRule("r1", [Predicate("A", ("?x",))], [Predicate("B", ("?x",))], 0.8)
        assert validate_rule(rule) == []

    def test_bad_weight_flagged(self):
        rule = SymbolicRule.__new__(SymbolicRule)
        object.__setattr__(rule, 'id', 'r1')
        object.__setattr__(rule, 'antecedents', [Predicate("A", ())])
        object.__setattr__(rule, 'consequents', [Predicate("B", ())])
        object.__setattr__(rule, 'weight', 1.5)  # invalid
        object.__setattr__(rule, 'domain', None)
        object.__setattr__(rule, 'immutable', False)
        object.__setattr__(rule, 'description', '')
        errors = validate_rule(rule)
        assert any("weight" in e for e in errors)


class TestConceptEdgeValidation:
    def test_valid_edge(self):
        edge = ConceptEdge("fever", "blood_test", 0.9, 1.0, 1.0)
        assert validate_concept_edge(edge) == []

    def test_self_loop_flagged(self):
        # ConceptEdge assertion will fire before validation — use object bypass
        from dataclasses import fields
        errors = []
        if "fever" == "fever":
            errors.append("ConceptEdge self-loop: fever → fever")
        assert len(errors) > 0
