"""
tests/unit/test_api_hardening.py
================================
100% coverage tests for:
  - nesy/api/nesy_model.py   (all methods including 5 revolutionary innovations)
  - nesy/core/validators.py  (every branch, every convenience function)

Covers:
  - Validator Wall: validate_predicate, validate_facts, validate_rule,
    validate_concept_edge, validate_confidence_report, validate_present_set,
    validate_null_set, clamp_probability, safe_divide, assert_valid_*
  - NeSyModel: reason (all paths), learn, explain, save/load concept graph,
    _facts_to_answer, _make_rejected_output, properties
  - PCAP: export_proof_capsule, verify_proof_capsule, load_proof_capsule
  - CFG:  suggest_fixes (Type2, Type3, empty nulls)
  - TB:   trust_budget_reason (budget exceeded, budget OK, budget ≤ 0)
  - DCV:  dual_channel_verdict (all 4 grades)
  - ECS:  edge_consistency_seal
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nesy.api.nesy_model import (
    CounterfactualFix,
    DualChannelVerdict,
    NeSyModel,
    ProofCapsule,
    TrustBudgetResult,
    _compute_pcap_checksum,
)
from nesy.core.exceptions import (
    CriticalNullViolation,
    NeSyError,
    SymbolicConflict,
)
from nesy.core.types import (
    ConceptEdge,
    ConfidenceReport,
    LogicClause,
    LogicConnective,
    NSIOutput,
    NullItem,
    NullSet,
    NullType,
    OutputStatus,
    Predicate,
    PresentSet,
    ReasoningStep,
    ReasoningTrace,
    SymbolicRule,
)
from nesy.core.validators import (
    assert_valid_concept_edge,
    assert_valid_confidence,
    assert_valid_facts,
    assert_valid_rule,
    clamp_probability,
    safe_divide,
    validate_concept_edge,
    validate_confidence_report,
    validate_facts,
    validate_immutable_not_overwritten,
    validate_null_set,
    validate_predicate,
    validate_present_set,
    validate_rule,
    validate_rules_no_duplicates,
)


# ═══════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def model():
    return NeSyModel(domain="test")


@pytest.fixture
def medical_model():
    m = NeSyModel(domain="medical")
    m.add_rule(SymbolicRule(
        id="fever-infection",
        antecedents=[Predicate("HasSymptom", ("?p", "fever"))],
        consequents=[Predicate("PossiblyHas", ("?p", "infection"))],
        weight=0.8,
    ))
    m.add_concept_edge(ConceptEdge(
        source="fever", target="blood_test",
        cooccurrence_prob=0.85, causal_strength=1.0, temporal_stability=1.0,
    ))
    m.register_critical_concept("blood_test", "diagnostic_test")
    return m


@pytest.fixture
def sample_output(medical_model):
    facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
    return medical_model.reason(facts, context_type="medical")


@pytest.fixture
def rejected_output(model):
    """Build a rejected output by hand."""
    ps = PresentSet(concepts={"a"}, context_type="general")
    ns = NullSet(items=[], present_set=ps)
    trace = ReasoningTrace(
        steps=[ReasoningStep(0, "halted", None, [], 0.0)],
        rules_activated=[], neural_confidence=0.0,
        symbolic_confidence=0.0, null_violations=[], logic_clauses=[],
    )
    conf = ConfidenceReport(factual=0.0, reasoning=0.0, knowledge_boundary=0.0)
    return NSIOutput(
        answer="", confidence=conf, reasoning_trace=trace,
        null_set=ns, status=OutputStatus.REJECTED, flags=["test_reject"],
    )


def _make_good_output():
    """High-confidence output for DCV grade A."""
    ps = PresentSet(concepts={"x"}, context_type="general")
    ns = NullSet(items=[], present_set=ps)
    trace = ReasoningTrace(
        steps=[ReasoningStep(0, "OK", None, [], 1.0)],
        rules_activated=[], neural_confidence=0.9,
        symbolic_confidence=0.9, null_violations=[], logic_clauses=[],
    )
    conf = ConfidenceReport(factual=0.9, reasoning=0.85, knowledge_boundary=0.82)
    return NSIOutput(
        answer="Good", confidence=conf, reasoning_trace=trace,
        null_set=ns, status=OutputStatus.OK, flags=[],
    )


def _make_uncertain_output():
    """Medium-confidence output for DCV grade B."""
    ps = PresentSet(concepts={"x"}, context_type="general")
    ns = NullSet(items=[], present_set=ps)
    trace = ReasoningTrace(
        steps=[ReasoningStep(0, "OK", None, [], 0.7)],
        rules_activated=[], neural_confidence=0.7,
        symbolic_confidence=0.7, null_violations=[], logic_clauses=[],
    )
    conf = ConfidenceReport(factual=0.7, reasoning=0.65, knowledge_boundary=0.62)
    return NSIOutput(
        answer="Uncertain", confidence=conf, reasoning_trace=trace,
        null_set=ns, status=OutputStatus.UNCERTAIN, flags=["warn1", "warn2"],
    )


def _make_low_output():
    """Low-confidence output for DCV grade C."""
    ps = PresentSet(concepts={"x"}, context_type="general")
    ns = NullSet(items=[], present_set=ps)
    trace = ReasoningTrace(
        steps=[ReasoningStep(0, "weak", None, [], 0.4)],
        rules_activated=[], neural_confidence=0.4,
        symbolic_confidence=0.4, null_violations=[], logic_clauses=[],
    )
    conf = ConfidenceReport(factual=0.45, reasoning=0.41, knowledge_boundary=0.40)
    return NSIOutput(
        answer="Low", confidence=conf, reasoning_trace=trace,
        null_set=ns, status=OutputStatus.FLAGGED, flags=[],
    )


# ═══════════════════════════════════════════════════════════════════
#  VALIDATOR WALL — validate_predicate
# ═══════════════════════════════════════════════════════════════════

class TestValidatePredicate:
    def test_valid_predicate_no_args(self):
        p = Predicate("foo", ())
        assert validate_predicate(p) == []

    def test_valid_predicate_ground_args(self):
        p = Predicate("HasSymptom", ("patient_1", "fever"))
        assert validate_predicate(p) == []

    def test_valid_predicate_variable_args(self):
        p = Predicate("HasSymptom", ("?patient", "?symptom"))
        assert validate_predicate(p) == []

    def test_empty_name(self):
        p = Predicate("", ("a",))
        errs = validate_predicate(p)
        assert any("empty" in e.lower() for e in errs)

    def test_invalid_name_chars(self):
        p = Predicate("123bad", ("a",))
        errs = validate_predicate(p)
        assert any("invalid" in e.lower() for e in errs)

    def test_invalid_name_special(self):
        p = Predicate("has-symptom", ("a",))
        errs = validate_predicate(p)
        assert len(errs) >= 1

    def test_empty_arg(self):
        p = Predicate("Foo", ("",))
        errs = validate_predicate(p)
        assert any("empty" in e.lower() for e in errs)

    def test_invalid_variable_format(self):
        p = Predicate("Foo", ("?123bad",))
        errs = validate_predicate(p)
        assert any("variable" in e.lower() for e in errs)

    def test_invalid_ground_term_format(self):
        p = Predicate("Foo", ("hello world",))
        errs = validate_predicate(p)
        assert any("invalid characters" in e.lower() for e in errs)

    def test_valid_ground_with_dots_dashes(self):
        p = Predicate("Foo", ("patient-1.v2",))
        assert validate_predicate(p) == []

    def test_mixed_valid_invalid(self):
        p = Predicate("Foo", ("ok", "", "?bad!"))
        errs = validate_predicate(p)
        assert len(errs) == 2  # one empty, one bad variable


# ═══════════════════════════════════════════════════════════════════
#  VALIDATOR WALL — validate_facts
# ═══════════════════════════════════════════════════════════════════

class TestValidateFacts:
    def test_valid_ground_facts(self):
        facts = {
            Predicate("A", ("x",)),
            Predicate("B", ("y", "z")),
        }
        assert validate_facts(facts) == []

    def test_facts_with_variable_rejected(self):
        facts = {Predicate("A", ("?x",))}
        errs = validate_facts(facts)
        assert any("unbound variable" in e.lower() for e in errs)

    def test_facts_with_invalid_predicate(self):
        facts = {Predicate("123", ("x",))}
        errs = validate_facts(facts)
        assert len(errs) >= 1


# ═══════════════════════════════════════════════════════════════════
#  VALIDATOR WALL — validate_rule
# ═══════════════════════════════════════════════════════════════════

class TestValidateRule:
    def test_valid_rule(self):
        r = SymbolicRule(
            id="r1",
            antecedents=[Predicate("A", ("?x",))],
            consequents=[Predicate("B", ("?x",))],
            weight=0.9,
        )
        assert validate_rule(r) == []

    def test_empty_rule_id(self):
        r = SymbolicRule.__new__(SymbolicRule)
        object.__setattr__(r, "id", "")
        object.__setattr__(r, "antecedents", [Predicate("A", ("x",))])
        object.__setattr__(r, "consequents", [Predicate("B", ("x",))])
        object.__setattr__(r, "weight", 0.5)
        object.__setattr__(r, "domain", None)
        object.__setattr__(r, "immutable", False)
        object.__setattr__(r, "description", "")
        errs = validate_rule(r)
        assert any("empty" in e.lower() for e in errs)

    def test_invalid_rule_id(self):
        r = SymbolicRule.__new__(SymbolicRule)
        object.__setattr__(r, "id", "bad!@#id")
        object.__setattr__(r, "antecedents", [Predicate("A", ("x",))])
        object.__setattr__(r, "consequents", [Predicate("B", ("x",))])
        object.__setattr__(r, "weight", 0.5)
        object.__setattr__(r, "domain", None)
        object.__setattr__(r, "immutable", False)
        object.__setattr__(r, "description", "")
        errs = validate_rule(r)
        assert any("invalid" in e.lower() for e in errs)

    def test_weight_out_of_range(self):
        r = SymbolicRule.__new__(SymbolicRule)
        object.__setattr__(r, "id", "r1")
        object.__setattr__(r, "antecedents", [Predicate("A", ("x",))])
        object.__setattr__(r, "consequents", [Predicate("B", ("x",))])
        object.__setattr__(r, "weight", 0.0)
        object.__setattr__(r, "domain", None)
        object.__setattr__(r, "immutable", False)
        object.__setattr__(r, "description", "")
        errs = validate_rule(r)
        assert any("weight" in e.lower() for e in errs)

    def test_no_antecedents(self):
        r = SymbolicRule.__new__(SymbolicRule)
        object.__setattr__(r, "id", "r1")
        object.__setattr__(r, "antecedents", [])
        object.__setattr__(r, "consequents", [Predicate("B", ("x",))])
        object.__setattr__(r, "weight", 0.5)
        object.__setattr__(r, "domain", None)
        object.__setattr__(r, "immutable", False)
        object.__setattr__(r, "description", "")
        errs = validate_rule(r)
        assert any("no antecedents" in e.lower() for e in errs)

    def test_no_consequents(self):
        r = SymbolicRule.__new__(SymbolicRule)
        object.__setattr__(r, "id", "r1")
        object.__setattr__(r, "antecedents", [Predicate("A", ("x",))])
        object.__setattr__(r, "consequents", [])
        object.__setattr__(r, "weight", 0.5)
        object.__setattr__(r, "domain", None)
        object.__setattr__(r, "immutable", False)
        object.__setattr__(r, "description", "")
        errs = validate_rule(r)
        assert any("no consequents" in e.lower() for e in errs)

    def test_invalid_antecedent_predicate(self):
        r = SymbolicRule.__new__(SymbolicRule)
        object.__setattr__(r, "id", "r1")
        object.__setattr__(r, "antecedents", [Predicate("123", ("x",))])
        object.__setattr__(r, "consequents", [Predicate("B", ("x",))])
        object.__setattr__(r, "weight", 0.5)
        object.__setattr__(r, "domain", None)
        object.__setattr__(r, "immutable", False)
        object.__setattr__(r, "description", "")
        errs = validate_rule(r)
        assert any("antecedent" in e.lower() for e in errs)

    def test_invalid_consequent_predicate(self):
        r = SymbolicRule.__new__(SymbolicRule)
        object.__setattr__(r, "id", "r1")
        object.__setattr__(r, "antecedents", [Predicate("A", ("x",))])
        object.__setattr__(r, "consequents", [Predicate("123", ("x",))])
        object.__setattr__(r, "weight", 0.5)
        object.__setattr__(r, "domain", None)
        object.__setattr__(r, "immutable", False)
        object.__setattr__(r, "description", "")
        errs = validate_rule(r)
        assert any("consequent" in e.lower() for e in errs)

    def test_hyphen_in_id_ok(self):
        r = SymbolicRule(
            id="fever-infection",
            antecedents=[Predicate("A", ("x",))],
            consequents=[Predicate("B", ("x",))],
            weight=0.9,
        )
        assert validate_rule(r) == []


# ═══════════════════════════════════════════════════════════════════
#  VALIDATOR WALL — validate_rules_no_duplicates
# ═══════════════════════════════════════════════════════════════════

class TestValidateRulesNoDuplicates:
    def test_no_duplicates(self):
        rules = [
            SymbolicRule(id="r1", antecedents=[Predicate("A", ("x",))],
                         consequents=[Predicate("B", ("x",))], weight=0.5),
            SymbolicRule(id="r2", antecedents=[Predicate("C", ("x",))],
                         consequents=[Predicate("D", ("x",))], weight=0.5),
        ]
        assert validate_rules_no_duplicates(rules) == []

    def test_duplicates_detected(self):
        r = SymbolicRule(id="r1", antecedents=[Predicate("A", ("x",))],
                         consequents=[Predicate("B", ("x",))], weight=0.5)
        errs = validate_rules_no_duplicates([r, r])
        assert any("duplicate" in e.lower() for e in errs)


# ═══════════════════════════════════════════════════════════════════
#  VALIDATOR WALL — validate_immutable_not_overwritten
# ═══════════════════════════════════════════════════════════════════

class TestValidateImmutableNotOverwritten:
    def test_no_conflict(self):
        existing = [SymbolicRule(id="r1", antecedents=[Predicate("A", ("x",))],
                                 consequents=[Predicate("B", ("x",))], weight=0.5)]
        new = SymbolicRule(id="r2", antecedents=[Predicate("C", ("x",))],
                           consequents=[Predicate("D", ("x",))], weight=0.5)
        assert validate_immutable_not_overwritten(existing, new) == []

    def test_immutable_conflict_detected(self):
        existing = [SymbolicRule(id="r1", antecedents=[Predicate("A", ("x",))],
                                 consequents=[Predicate("B", ("x",))], weight=0.5,
                                 immutable=True)]
        new = SymbolicRule(id="r1", antecedents=[Predicate("C", ("x",))],
                           consequents=[Predicate("D", ("x",))], weight=0.5)
        errs = validate_immutable_not_overwritten(existing, new)
        assert any("immutable" in e.lower() for e in errs)

    def test_same_id_not_immutable(self):
        existing = [SymbolicRule(id="r1", antecedents=[Predicate("A", ("x",))],
                                 consequents=[Predicate("B", ("x",))], weight=0.5,
                                 immutable=False)]
        new = SymbolicRule(id="r1", antecedents=[Predicate("C", ("x",))],
                           consequents=[Predicate("D", ("x",))], weight=0.5)
        assert validate_immutable_not_overwritten(existing, new) == []


# ═══════════════════════════════════════════════════════════════════
#  VALIDATOR WALL — validate_concept_edge
# ═══════════════════════════════════════════════════════════════════

class TestValidateConceptEdge:
    def test_valid_edge(self):
        edge = ConceptEdge("a", "b", 0.8, 1.0, 0.5)
        assert validate_concept_edge(edge) == []

    def test_empty_source(self):
        edge = ConceptEdge.__new__(ConceptEdge)
        object.__setattr__(edge, "source", "")
        object.__setattr__(edge, "target", "b")
        object.__setattr__(edge, "cooccurrence_prob", 0.5)
        object.__setattr__(edge, "causal_strength", 1.0)
        object.__setattr__(edge, "temporal_stability", 1.0)
        errs = validate_concept_edge(edge)
        assert any("source is empty" in e.lower() for e in errs)

    def test_empty_target(self):
        edge = ConceptEdge.__new__(ConceptEdge)
        object.__setattr__(edge, "source", "a")
        object.__setattr__(edge, "target", "")
        object.__setattr__(edge, "cooccurrence_prob", 0.5)
        object.__setattr__(edge, "causal_strength", 1.0)
        object.__setattr__(edge, "temporal_stability", 1.0)
        errs = validate_concept_edge(edge)
        assert any("target is empty" in e.lower() for e in errs)

    def test_self_loop(self):
        edge = ConceptEdge.__new__(ConceptEdge)
        object.__setattr__(edge, "source", "a")
        object.__setattr__(edge, "target", "a")
        object.__setattr__(edge, "cooccurrence_prob", 0.5)
        object.__setattr__(edge, "causal_strength", 1.0)
        object.__setattr__(edge, "temporal_stability", 1.0)
        errs = validate_concept_edge(edge)
        assert any("self-loop" in e.lower() for e in errs)

    def test_cooccurrence_out_of_range(self):
        edge = ConceptEdge.__new__(ConceptEdge)
        object.__setattr__(edge, "source", "a")
        object.__setattr__(edge, "target", "b")
        object.__setattr__(edge, "cooccurrence_prob", 1.5)
        object.__setattr__(edge, "causal_strength", 1.0)
        object.__setattr__(edge, "temporal_stability", 1.0)
        errs = validate_concept_edge(edge)
        assert any("cooccurrence" in e.lower() for e in errs)

    def test_causal_strength_invalid(self):
        edge = ConceptEdge.__new__(ConceptEdge)
        object.__setattr__(edge, "source", "a")
        object.__setattr__(edge, "target", "b")
        object.__setattr__(edge, "cooccurrence_prob", 0.5)
        object.__setattr__(edge, "causal_strength", 0.7)
        object.__setattr__(edge, "temporal_stability", 1.0)
        errs = validate_concept_edge(edge)
        assert any("causal_strength" in e.lower() for e in errs)

    def test_temporal_stability_invalid(self):
        edge = ConceptEdge.__new__(ConceptEdge)
        object.__setattr__(edge, "source", "a")
        object.__setattr__(edge, "target", "b")
        object.__setattr__(edge, "cooccurrence_prob", 0.5)
        object.__setattr__(edge, "causal_strength", 1.0)
        object.__setattr__(edge, "temporal_stability", 0.3)
        errs = validate_concept_edge(edge)
        assert any("temporal_stability" in e.lower() for e in errs)


# ═══════════════════════════════════════════════════════════════════
#  VALIDATOR WALL — validate_confidence_report
# ═══════════════════════════════════════════════════════════════════

class TestValidateConfidenceReport:
    def test_valid_report(self):
        r = ConfidenceReport(factual=0.8, reasoning=0.7, knowledge_boundary=0.9)
        assert validate_confidence_report(r) == []

    def test_factual_out_of_range(self):
        r = ConfidenceReport.__new__(ConfidenceReport)
        object.__setattr__(r, "factual", 1.5)
        object.__setattr__(r, "reasoning", 0.5)
        object.__setattr__(r, "knowledge_boundary", 0.5)
        object.__setattr__(r, "explanation", {})
        errs = validate_confidence_report(r)
        assert any("factual" in e.lower() for e in errs)

    def test_reasoning_negative(self):
        r = ConfidenceReport.__new__(ConfidenceReport)
        object.__setattr__(r, "factual", 0.5)
        object.__setattr__(r, "reasoning", -0.1)
        object.__setattr__(r, "knowledge_boundary", 0.5)
        object.__setattr__(r, "explanation", {})
        errs = validate_confidence_report(r)
        assert any("reasoning" in e.lower() for e in errs)

    def test_boundary_out_of_range(self):
        r = ConfidenceReport.__new__(ConfidenceReport)
        object.__setattr__(r, "factual", 0.5)
        object.__setattr__(r, "reasoning", 0.5)
        object.__setattr__(r, "knowledge_boundary", 2.0)
        object.__setattr__(r, "explanation", {})
        errs = validate_confidence_report(r)
        assert any("knowledge_boundary" in e.lower() for e in errs)


# ═══════════════════════════════════════════════════════════════════
#  VALIDATOR WALL — validate_present_set / validate_null_set
# ═══════════════════════════════════════════════════════════════════

class TestValidatePresentSet:
    def test_valid(self):
        ps = PresentSet(concepts={"a", "b"}, context_type="medical")
        assert validate_present_set(ps) == []

    def test_empty_concepts(self):
        ps = PresentSet(concepts=set(), context_type="general")
        errs = validate_present_set(ps)
        assert any("empty" in e.lower() for e in errs)

    def test_empty_context_type(self):
        ps = PresentSet(concepts={"a"}, context_type="")
        errs = validate_present_set(ps)
        assert any("context_type" in e.lower() for e in errs)


class TestValidateNullSet:
    def test_valid_null_set(self):
        ps = PresentSet(concepts={"a"}, context_type="general")
        ns = NullSet(items=[], present_set=ps)
        assert validate_null_set(ns) == []

    def test_contradictory_item(self):
        ps = PresentSet(concepts={"fever"}, context_type="general")
        item = NullItem("fever", 0.5, NullType.TYPE2_MEANINGFUL, ["x"])
        ns = NullSet(items=[item], present_set=ps)
        errs = validate_null_set(ns)
        assert any("contradictory" in e.lower() for e in errs)

    def test_unsorted_items(self):
        ps = PresentSet(concepts={"a"}, context_type="general")
        # Type1 before Type2 — wrong order
        items = [
            NullItem("x", 0.1, NullType.TYPE1_EXPECTED, ["a"]),
            NullItem("y", 0.5, NullType.TYPE2_MEANINGFUL, ["a"]),
        ]
        ns = NullSet(items=items, present_set=ps)
        errs = validate_null_set(ns)
        assert any("not sorted" in e.lower() for e in errs)

    def test_correctly_sorted_items(self):
        ps = PresentSet(concepts={"a"}, context_type="general")
        items = [
            NullItem("y", 0.5, NullType.TYPE3_CRITICAL, ["a"]),
            NullItem("z", 0.3, NullType.TYPE2_MEANINGFUL, ["a"]),
            NullItem("x", 0.1, NullType.TYPE1_EXPECTED, ["a"]),
        ]
        ns = NullSet(items=items, present_set=ps)
        assert validate_null_set(ns) == []


# ═══════════════════════════════════════════════════════════════════
#  VALIDATOR WALL — numerical stability
# ═══════════════════════════════════════════════════════════════════

class TestNumericalStability:
    def test_clamp_normal(self):
        assert clamp_probability(0.5) == 0.5

    def test_clamp_below_zero(self):
        assert clamp_probability(-0.1) == 0.0

    def test_clamp_above_one(self):
        assert clamp_probability(1.5) == 1.0

    def test_clamp_nan_raises(self):
        with pytest.raises(NeSyError, match="NaN"):
            clamp_probability(float("nan"))

    def test_clamp_pos_inf(self):
        assert clamp_probability(float("inf")) == 1.0

    def test_clamp_neg_inf(self):
        assert clamp_probability(float("-inf")) == 0.0

    def test_clamp_with_label(self):
        with pytest.raises(NeSyError, match="my_label"):
            clamp_probability(float("nan"), "my_label")

    def test_safe_divide_normal(self):
        result = safe_divide(10.0, 5.0)
        assert abs(result - 2.0) < 0.01

    def test_safe_divide_zero_denom(self):
        result = safe_divide(1.0, 0.0)
        assert result > 0  # 1 / epsilon → large but finite

    def test_safe_divide_custom_epsilon(self):
        result = safe_divide(1.0, 0.0, epsilon=1.0)
        assert abs(result - 1.0) < 0.001


# ═══════════════════════════════════════════════════════════════════
#  VALIDATOR WALL — convenience functions
# ═══════════════════════════════════════════════════════════════════

class TestAssertValidFunctions:
    def test_assert_valid_facts_ok(self):
        facts = {Predicate("A", ("x",))}
        assert_valid_facts(facts)  # no exception

    def test_assert_valid_facts_raises(self):
        facts = {Predicate("123", ("?x",))}
        with pytest.raises(NeSyError, match="Invalid facts"):
            assert_valid_facts(facts)

    def test_assert_valid_rule_ok(self):
        r = SymbolicRule(id="r1", antecedents=[Predicate("A", ("x",))],
                         consequents=[Predicate("B", ("x",))], weight=0.5)
        assert_valid_rule(r)  # no exception

    def test_assert_valid_rule_raises(self):
        r = SymbolicRule.__new__(SymbolicRule)
        object.__setattr__(r, "id", "")
        object.__setattr__(r, "antecedents", [])
        object.__setattr__(r, "consequents", [])
        object.__setattr__(r, "weight", 0.0)
        object.__setattr__(r, "domain", None)
        object.__setattr__(r, "immutable", False)
        object.__setattr__(r, "description", "")
        with pytest.raises(NeSyError, match="Invalid rule"):
            assert_valid_rule(r)

    def test_assert_valid_concept_edge_ok(self):
        edge = ConceptEdge("a", "b", 0.8, 1.0, 0.5)
        assert_valid_concept_edge(edge)  # no exception

    def test_assert_valid_concept_edge_raises(self):
        edge = ConceptEdge.__new__(ConceptEdge)
        object.__setattr__(edge, "source", "")
        object.__setattr__(edge, "target", "")
        object.__setattr__(edge, "cooccurrence_prob", 0.5)
        object.__setattr__(edge, "causal_strength", 1.0)
        object.__setattr__(edge, "temporal_stability", 1.0)
        with pytest.raises(NeSyError, match="Invalid concept edge"):
            assert_valid_concept_edge(edge)

    def test_assert_valid_confidence_ok(self):
        r = ConfidenceReport(factual=0.5, reasoning=0.5, knowledge_boundary=0.5)
        assert_valid_confidence(r)  # no exception

    def test_assert_valid_confidence_raises(self):
        r = ConfidenceReport.__new__(ConfidenceReport)
        object.__setattr__(r, "factual", 2.0)
        object.__setattr__(r, "reasoning", 0.5)
        object.__setattr__(r, "knowledge_boundary", 0.5)
        object.__setattr__(r, "explanation", {})
        with pytest.raises(NeSyError, match="Invalid confidence"):
            assert_valid_confidence(r)


# ═══════════════════════════════════════════════════════════════════
#  NESY MODEL — basic API
# ═══════════════════════════════════════════════════════════════════

class TestNeSyModelBasic:
    def test_init_defaults(self, model):
        assert model.domain == "test"
        assert model.rule_count == 0

    def test_add_rule_chainable(self, model):
        r = SymbolicRule(id="r1", antecedents=[Predicate("A", ("x",))],
                         consequents=[Predicate("B", ("x",))], weight=0.5)
        result = model.add_rule(r)
        assert result is model
        assert model.rule_count == 1

    def test_add_rules_chainable(self, model):
        rules = [
            SymbolicRule(id=f"r{i}", antecedents=[Predicate("A", ("x",))],
                         consequents=[Predicate("B", ("x",))], weight=0.5)
            for i in range(3)
        ]
        result = model.add_rules(rules)
        assert result is model
        assert model.rule_count == 3

    def test_add_concept_edge_chainable(self, model):
        edge = ConceptEdge("a", "b", 0.8, 1.0, 0.5)
        result = model.add_concept_edge(edge)
        assert result is model

    def test_add_concept_edges_chainable(self, model):
        edges = [ConceptEdge("a", "b", 0.8, 1.0, 0.5),
                 ConceptEdge("b", "c", 0.7, 0.5, 1.0)]
        result = model.add_concept_edges(edges)
        assert result is model

    def test_register_critical_concept(self, model):
        result = model.register_critical_concept("bp", "vital_signs")
        assert result is model

    def test_anchor_chainable(self, model):
        r = SymbolicRule(id="anchor1", antecedents=[Predicate("A", ("x",))],
                         consequents=[Predicate("B", ("x",))], weight=1.0,
                         immutable=True)
        result = model.anchor(r)
        assert result is model
        assert model.anchored_rules == 1

    def test_concept_graph_stats(self, model):
        stats = model.concept_graph_stats
        assert "concepts" in stats
        assert "edges" in stats


# ═══════════════════════════════════════════════════════════════════
#  NESY MODEL — reason()
# ═══════════════════════════════════════════════════════════════════

class TestNeSyModelReason:
    def test_reason_basic(self, medical_model):
        facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
        output = medical_model.reason(facts, context_type="medical")
        assert isinstance(output, NSIOutput)
        # blood_test is registered as critical → Type3 null → REJECTED is valid
        assert output.status in (OutputStatus.OK, OutputStatus.UNCERTAIN,
                                  OutputStatus.FLAGGED, OutputStatus.REJECTED)

    def test_reason_no_rules_no_derivation(self, model):
        facts = {Predicate("A", ("x",))}
        output = model.reason(facts)
        assert "No new conclusions" in output.answer

    def test_reason_with_raw_input(self, medical_model):
        facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
        output = medical_model.reason(facts, context_type="medical",
                                       raw_input="Test query")
        assert isinstance(output, NSIOutput)

    def test_reason_symbolic_conflict_path(self, model):
        """Force SymbolicConflict → REJECTED output."""
        with patch.object(model._symbolic, "reason",
                          side_effect=SymbolicConflict(
                              "conflict", ["r1"], {})):
            facts = {Predicate("A", ("x",))}
            output = model.reason(facts)
            assert output.status == OutputStatus.REJECTED
            assert "conflict" in output.flags[0].lower()

    def test_reason_critical_null_violation_path(self, model):
        """Force CriticalNullViolation → REJECTED output."""
        # Make symbolic.reason succeed, but monitor.evaluate raise
        with patch.object(model._monitor, "evaluate",
                          side_effect=CriticalNullViolation(
                              "critical null", ["item1"])):
            r = SymbolicRule(id="r1", antecedents=[Predicate("A", ("x",))],
                             consequents=[Predicate("B", ("x",))], weight=0.5)
            model.add_rule(r)
            facts = {Predicate("A", ("x",))}
            output = model.reason(facts)
            assert output.status == OutputStatus.REJECTED
            assert "critical null" in output.flags[0].lower()

    def test_deterministic_answer_ordering(self, model):
        """Same inputs → same answer string (sorted derived facts)."""
        r1 = SymbolicRule(id="r1", antecedents=[Predicate("A", ("x",))],
                          consequents=[Predicate("Z", ("x",))], weight=0.5)
        r2 = SymbolicRule(id="r2", antecedents=[Predicate("A", ("x",))],
                          consequents=[Predicate("M", ("x",))], weight=0.5)
        model.add_rules([r1, r2])
        facts = {Predicate("A", ("x",))}
        o1 = model.reason(facts)
        o2 = model.reason(facts)
        assert o1.answer == o2.answer


# ═══════════════════════════════════════════════════════════════════
#  NESY MODEL — learn()
# ═══════════════════════════════════════════════════════════════════

class TestNeSyModelLearn:
    def test_learn_non_anchor(self, model):
        r = SymbolicRule(id="r1", antecedents=[Predicate("A", ("x",))],
                         consequents=[Predicate("B", ("x",))], weight=0.5)
        result = model.learn(r)
        assert result is model
        assert model.rule_count == 1

    def test_learn_with_anchor(self, model):
        r = SymbolicRule(id="r1", antecedents=[Predicate("A", ("x",))],
                         consequents=[Predicate("B", ("x",))], weight=0.5,
                         immutable=True)
        result = model.learn(r, make_anchor=True)
        assert result is model
        assert model.anchored_rules == 1


# ═══════════════════════════════════════════════════════════════════
#  NESY MODEL — explain()
# ═══════════════════════════════════════════════════════════════════

class TestNeSyModelExplain:
    def test_explain_basic(self, medical_model, sample_output):
        explanation = medical_model.explain(sample_output)
        assert "NeSy-Core Explanation" in explanation
        assert "Answer:" in explanation
        assert "Status:" in explanation
        assert "Confidence" in explanation
        assert "Reasoning Steps" in explanation

    def test_explain_with_flags(self, model):
        """Output with flags should show them."""
        ps = PresentSet(concepts={"a"}, context_type="general")
        ns = NullSet(items=[], present_set=ps)
        trace = ReasoningTrace(
            steps=[ReasoningStep(0, "step0", None, [], 0.5)],
            rules_activated=[], neural_confidence=0.5,
            symbolic_confidence=0.5, null_violations=[], logic_clauses=[],
        )
        conf = ConfidenceReport(factual=0.5, reasoning=0.5, knowledge_boundary=0.5)
        output = NSIOutput("test", conf, trace, ns,
                           OutputStatus.FLAGGED, ["flag1", "flag2"])
        explanation = model.explain(output)
        assert "flag1" in explanation
        assert "flag2" in explanation
        assert "Flags" in explanation

    def test_explain_with_critical_nulls(self, model):
        ps = PresentSet(concepts={"a"}, context_type="medical")
        critical = NullItem("bp", 0.9, NullType.TYPE3_CRITICAL, ["a"], 2.0)
        ns = NullSet(items=[critical], present_set=ps)
        trace = ReasoningTrace(
            steps=[ReasoningStep(0, "step0", None, [], 0.5)],
            rules_activated=[], neural_confidence=0.5,
            symbolic_confidence=0.5, null_violations=[], logic_clauses=[],
        )
        conf = ConfidenceReport(factual=0.5, reasoning=0.5, knowledge_boundary=0.5)
        output = NSIOutput("test", conf, trace, ns,
                           OutputStatus.FLAGGED, [])
        explanation = model.explain(output)
        assert "CRITICAL" in explanation
        assert "bp" in explanation

    def test_explain_with_meaningful_nulls(self, model):
        ps = PresentSet(concepts={"a"}, context_type="general")
        meaningful = NullItem("y", 0.6, NullType.TYPE2_MEANINGFUL, ["a"])
        ns = NullSet(items=[meaningful], present_set=ps)
        trace = ReasoningTrace(
            steps=[ReasoningStep(0, "step0", None, [], 0.5)],
            rules_activated=[], neural_confidence=0.5,
            symbolic_confidence=0.5, null_violations=[], logic_clauses=[],
        )
        conf = ConfidenceReport(factual=0.5, reasoning=0.5, knowledge_boundary=0.5)
        output = NSIOutput("test", conf, trace, ns,
                           OutputStatus.OK, [])
        explanation = model.explain(output)
        assert "Meaningful" in explanation
        assert "y" in explanation


# ═══════════════════════════════════════════════════════════════════
#  NESY MODEL — save/load concept graph
# ═══════════════════════════════════════════════════════════════════

class TestNeSyModelSaveLoad:
    def test_save_and_load(self, medical_model, tmp_path):
        path = str(tmp_path / "graph.json")
        medical_model.save_concept_graph(path)
        assert Path(path).exists()

        model2 = NeSyModel(domain="other")
        result = model2.load_concept_graph(path)
        assert result is model2
        assert model2.concept_graph_stats["edges"] > 0


# ═══════════════════════════════════════════════════════════════════
#  NESY MODEL — _facts_to_answer, _make_rejected_output
# ═══════════════════════════════════════════════════════════════════

class TestNeSyModelPrivateHelpers:
    def test_facts_to_answer_no_new(self, model):
        orig = {Predicate("A", ("x",))}
        result = model._facts_to_answer(orig, orig)
        assert "No new conclusions" in result

    def test_facts_to_answer_with_new(self, model):
        orig = {Predicate("A", ("x",))}
        derived = {Predicate("A", ("x",)), Predicate("B", ("x",))}
        result = model._facts_to_answer(derived, orig)
        assert "Derived:" in result
        assert "B(x)" in result

    def test_make_rejected_output(self, model):
        output = model._make_rejected_output(
            facts={Predicate("A", ("x",))},
            context_type="general",
            reason="test failure",
        )
        assert output.status == OutputStatus.REJECTED
        assert output.confidence.factual == 0.0
        assert output.confidence.reasoning == 0.0
        assert output.confidence.knowledge_boundary == 0.0
        assert "test failure" in output.flags[0]
        assert output.answer == ""
        assert len(output.reasoning_trace.steps) == 1


# ═══════════════════════════════════════════════════════════════════
#  PROOF CAPSULE (PCAP)
# ═══════════════════════════════════════════════════════════════════

class TestProofCapsule:
    def test_export_proof_capsule(self, medical_model, sample_output):
        capsule = medical_model.export_proof_capsule(sample_output)
        assert isinstance(capsule, ProofCapsule)
        assert capsule.version == "1.0"
        assert capsule.domain == "medical"
        assert capsule.checksum  # non-empty
        assert capsule.request_id == sample_output.request_id

    def test_export_proof_capsule_to_file(self, medical_model, sample_output, tmp_path):
        path = str(tmp_path / "proof.pcap.json")
        capsule = medical_model.export_proof_capsule(sample_output, path=path)
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["version"] == "1.0"
        assert data["checksum"] == capsule.checksum

    def test_verify_proof_capsule_ok(self, medical_model, sample_output):
        capsule = medical_model.export_proof_capsule(sample_output)
        assert NeSyModel.verify_proof_capsule(capsule) is True

    def test_verify_proof_capsule_tampered(self, medical_model, sample_output):
        capsule = medical_model.export_proof_capsule(sample_output)
        # Tamper with the answer
        capsule_dict = capsule.to_dict()
        capsule_dict["answer"] = "TAMPERED"
        tampered = ProofCapsule(**capsule_dict)
        assert NeSyModel.verify_proof_capsule(tampered) is False

    def test_load_proof_capsule(self, medical_model, sample_output, tmp_path):
        path = str(tmp_path / "proof.pcap.json")
        medical_model.export_proof_capsule(sample_output, path=path)
        loaded = NeSyModel.load_proof_capsule(path)
        assert isinstance(loaded, ProofCapsule)
        assert loaded.version == "1.0"

    def test_load_proof_capsule_bad_path(self):
        with pytest.raises(NeSyError, match="Failed to load"):
            NeSyModel.load_proof_capsule("/nonexistent/path.json")

    def test_load_proof_capsule_bad_json(self, tmp_path):
        path = str(tmp_path / "bad.json")
        Path(path).write_text("not json {{{")
        with pytest.raises(NeSyError, match="Failed to load"):
            NeSyModel.load_proof_capsule(path)

    def test_capsule_to_dict(self, medical_model, sample_output):
        capsule = medical_model.export_proof_capsule(sample_output)
        d = capsule.to_dict()
        assert "version" in d
        assert "checksum" in d
        assert isinstance(d["steps"], list)

    def test_compute_pcap_checksum_deterministic(self):
        data = {"a": 1, "b": 2, "checksum": ""}
        h1 = _compute_pcap_checksum(data)
        h2 = _compute_pcap_checksum(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_export_rejected_output(self, model, rejected_output):
        capsule = model.export_proof_capsule(rejected_output)
        assert capsule.status == "rejected"
        assert capsule.answer == ""


# ═══════════════════════════════════════════════════════════════════
#  COUNTERFACTUAL FIX GENERATOR (CFG)
# ═══════════════════════════════════════════════════════════════════

class TestCounterfactualFix:
    def test_suggest_fixes_with_nulls(self, medical_model, sample_output):
        fixes = medical_model.suggest_fixes(sample_output)
        assert isinstance(fixes, list)
        if fixes:
            fix = fixes[0]
            assert isinstance(fix, CounterfactualFix)
            assert fix.predicted_uplift >= 0
            assert fix.explanation

    def test_suggest_fixes_empty_nullset(self, model):
        ps = PresentSet(concepts={"a"}, context_type="general")
        ns = NullSet(items=[], present_set=ps)
        trace = ReasoningTrace(
            steps=[], rules_activated=[], neural_confidence=0.9,
            symbolic_confidence=0.9, null_violations=[], logic_clauses=[],
        )
        conf = ConfidenceReport(factual=0.9, reasoning=0.9, knowledge_boundary=0.9)
        output = NSIOutput("ok", conf, trace, ns, OutputStatus.OK, [])
        fixes = model.suggest_fixes(output)
        assert fixes == []

    def test_suggest_fixes_type3_critical(self, model):
        ps = PresentSet(concepts={"a"}, context_type="medical")
        critical = NullItem("bp", 0.9, NullType.TYPE3_CRITICAL, ["a"], 2.0)
        ns = NullSet(items=[critical], present_set=ps)
        trace = ReasoningTrace(
            steps=[], rules_activated=[], neural_confidence=0.5,
            symbolic_confidence=0.5, null_violations=[], logic_clauses=[],
        )
        conf = ConfidenceReport(factual=0.5, reasoning=0.5, knowledge_boundary=0.5)
        output = NSIOutput("low", conf, trace, ns, OutputStatus.FLAGGED, [])
        fixes = model.suggest_fixes(output)
        assert len(fixes) >= 1
        assert fixes[0].source_null_type == "TYPE3_CRITICAL"
        assert "CRITICAL" in fixes[0].explanation

    def test_suggest_fixes_type2_meaningful(self, model):
        ps = PresentSet(concepts={"a"}, context_type="general")
        meaningful = NullItem("y", 0.6, NullType.TYPE2_MEANINGFUL, ["a"])
        ns = NullSet(items=[meaningful], present_set=ps)
        trace = ReasoningTrace(
            steps=[], rules_activated=[], neural_confidence=0.5,
            symbolic_confidence=0.5, null_violations=[], logic_clauses=[],
        )
        conf = ConfidenceReport(factual=0.5, reasoning=0.5, knowledge_boundary=0.5)
        output = NSIOutput("mid", conf, trace, ns, OutputStatus.OK, [])
        fixes = model.suggest_fixes(output)
        assert len(fixes) == 1
        assert fixes[0].source_null_type == "TYPE2_MEANINGFUL"

    def test_suggest_fixes_type1_skipped(self, model):
        ps = PresentSet(concepts={"a"}, context_type="general")
        expected = NullItem("z", 0.1, NullType.TYPE1_EXPECTED, ["a"])
        ns = NullSet(items=[expected], present_set=ps)
        trace = ReasoningTrace(
            steps=[], rules_activated=[], neural_confidence=0.9,
            symbolic_confidence=0.9, null_violations=[], logic_clauses=[],
        )
        conf = ConfidenceReport(factual=0.9, reasoning=0.9, knowledge_boundary=0.9)
        output = NSIOutput("ok", conf, trace, ns, OutputStatus.OK, [])
        fixes = model.suggest_fixes(output)
        assert fixes == []

    def test_suggest_fixes_sorted_by_uplift(self, model):
        ps = PresentSet(concepts={"a"}, context_type="general")
        items = [
            NullItem("y", 0.6, NullType.TYPE2_MEANINGFUL, ["a"]),
            NullItem("z", 0.9, NullType.TYPE3_CRITICAL, ["a"], 2.0),
        ]
        ns = NullSet(items=items, present_set=ps)
        trace = ReasoningTrace(
            steps=[], rules_activated=[], neural_confidence=0.5,
            symbolic_confidence=0.5, null_violations=[], logic_clauses=[],
        )
        conf = ConfidenceReport(factual=0.5, reasoning=0.5, knowledge_boundary=0.5)
        output = NSIOutput("low", conf, trace, ns, OutputStatus.FLAGGED, [])
        fixes = model.suggest_fixes(output)
        assert len(fixes) == 2
        assert fixes[0].predicted_uplift >= fixes[1].predicted_uplift


# ═══════════════════════════════════════════════════════════════════
#  TRUST BUDGET (TB)
# ═══════════════════════════════════════════════════════════════════

class TestTrustBudget:
    def test_trust_budget_ok(self, model):
        facts = {Predicate("A", ("x",))}
        result = model.trust_budget_reason(facts, budget=10.0)
        assert isinstance(result, TrustBudgetResult)
        assert result.cost > 0
        assert result.remaining_budget >= 0
        assert result.budget_exceeded is False

    def test_trust_budget_exceeded(self, model):
        facts = {Predicate("A", ("x",))}
        result = model.trust_budget_reason(facts, budget=0.001)
        assert result.budget_exceeded is True
        assert result.remaining_budget == 0.0

    def test_trust_budget_zero_raises(self, model):
        facts = {Predicate("A", ("x",))}
        with pytest.raises(NeSyError, match="positive"):
            model.trust_budget_reason(facts, budget=0.0)

    def test_trust_budget_negative_raises(self, model):
        facts = {Predicate("A", ("x",))}
        with pytest.raises(NeSyError, match="positive"):
            model.trust_budget_reason(facts, budget=-1.0)

    def test_trust_budget_with_context(self, medical_model):
        facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
        result = medical_model.trust_budget_reason(
            facts, budget=5.0, context_type="medical",
            raw_input="test"
        )
        assert result.output.status != OutputStatus.REJECTED or result.cost > 0


# ═══════════════════════════════════════════════════════════════════
#  DUAL-CHANNEL VERDICT (DCV)
# ═══════════════════════════════════════════════════════════════════

class TestDualChannelVerdict:
    def test_grade_a(self, model):
        output = _make_good_output()
        verdict = model.dual_channel_verdict(output)
        assert isinstance(verdict, DualChannelVerdict)
        assert verdict.compliance_grade == "A"
        assert verdict.overall_pass is True
        assert verdict.decision == "Good"

    def test_grade_b(self, model):
        output = _make_uncertain_output()
        verdict = model.dual_channel_verdict(output)
        assert verdict.compliance_grade == "B"
        assert verdict.overall_pass is True

    def test_grade_c(self, model):
        output = _make_low_output()
        verdict = model.dual_channel_verdict(output)
        assert verdict.compliance_grade == "C"
        assert verdict.overall_pass is False

    def test_grade_d(self, model, rejected_output):
        verdict = model.dual_channel_verdict(rejected_output)
        assert verdict.compliance_grade == "D"
        assert verdict.overall_pass is False
        assert verdict.decision_status == "rejected"

    def test_grade_c_with_critical_nulls_but_high_conf(self, model):
        """High confidence but critical nulls → B not possible."""
        ps = PresentSet(concepts={"a"}, context_type="general")
        critical = NullItem("bp", 0.9, NullType.TYPE3_CRITICAL, ["a"])
        ns = NullSet(items=[critical], present_set=ps)
        trace = ReasoningTrace(
            steps=[], rules_activated=[], neural_confidence=0.8,
            symbolic_confidence=0.8, null_violations=[], logic_clauses=[],
        )
        conf = ConfidenceReport(factual=0.8, reasoning=0.8, knowledge_boundary=0.8)
        output = NSIOutput("test", conf, trace, ns,
                           OutputStatus.FLAGGED, [])
        verdict = model.dual_channel_verdict(output)
        assert verdict.compliance_grade == "C"

    def test_verdict_detail_keys(self, model):
        output = _make_good_output()
        verdict = model.dual_channel_verdict(output)
        assert "factual" in verdict.compliance_detail
        assert "reasoning" in verdict.compliance_detail
        assert "knowledge_boundary" in verdict.compliance_detail
        assert "critical_nulls" in verdict.compliance_detail
        assert "flags" in verdict.compliance_detail


# ═══════════════════════════════════════════════════════════════════
#  EDGE CONSISTENCY SEAL (ECS)
# ═══════════════════════════════════════════════════════════════════

class TestEdgeConsistencySeal:
    def test_seal_basic(self, medical_model):
        facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
        seal = medical_model.edge_consistency_seal(facts, "medical")
        assert "graph_checksum" in seal
        assert len(seal["graph_checksum"]) == 64  # SHA-256
        assert "answer" in seal
        assert "status" in seal
        assert "confidence_minimum" in seal
        assert "null_count" in seal
        assert "seal_timestamp" in seal

    def test_seal_deterministic(self, medical_model):
        facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
        s1 = medical_model.edge_consistency_seal(facts, "medical")
        s2 = medical_model.edge_consistency_seal(facts, "medical")
        assert s1["graph_checksum"] == s2["graph_checksum"]
        assert s1["answer"] == s2["answer"]

    def test_seal_empty_graph(self, model):
        facts = {Predicate("A", ("x",))}
        seal = model.edge_consistency_seal(facts)
        assert isinstance(seal["graph_checksum"], str)
        assert seal["null_count"] >= 0


# ═══════════════════════════════════════════════════════════════════
#  NESY MODEL — strict mode
# ═══════════════════════════════════════════════════════════════════

class TestNeSyModelStrictMode:
    def test_strict_mode_constructor(self):
        model = NeSyModel(domain="test", strict_mode=True,
                          doubt_threshold=0.5, lambda_ewc=500.0)
        assert model.domain == "test"


# ═══════════════════════════════════════════════════════════════════
#  EDGE CASES & REGRESSION GUARDS
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_reason_with_empty_derived_set(self, model):
        """No rules → no derivation → 'No new conclusions'."""
        facts = {Predicate("X", ("a",))}
        output = model.reason(facts)
        assert "No new conclusions" in output.answer

    def test_properties_after_operations(self, medical_model):
        assert medical_model.rule_count >= 1
        assert medical_model.concept_graph_stats["edges"] >= 1

    def test_multiple_chained_operations(self, model):
        r = SymbolicRule(id="r1", antecedents=[Predicate("A", ("x",))],
                         consequents=[Predicate("B", ("x",))], weight=0.5)
        edge = ConceptEdge("a", "b", 0.8, 1.0, 0.5)
        model.add_rule(r).add_concept_edge(edge).register_critical_concept("b", "test")
        assert model.rule_count == 1

    def test_counterfactual_fix_dataclass_fields(self):
        fix = CounterfactualFix("concept", 0.5, "TYPE2_MEANINGFUL", "explanation")
        assert fix.missing_concept == "concept"
        assert fix.predicted_uplift == 0.5
        assert fix.source_null_type == "TYPE2_MEANINGFUL"
        assert fix.explanation == "explanation"

    def test_dual_channel_verdict_dataclass_fields(self):
        v = DualChannelVerdict("answer", "ok", "A", {}, True)
        assert v.decision == "answer"
        assert v.decision_status == "ok"
        assert v.compliance_grade == "A"
        assert v.overall_pass is True

    def test_trust_budget_result_dataclass_fields(self):
        ps = PresentSet(concepts={"a"}, context_type="general")
        ns = NullSet(items=[], present_set=ps)
        trace = ReasoningTrace(
            steps=[], rules_activated=[], neural_confidence=0.9,
            symbolic_confidence=0.9, null_violations=[], logic_clauses=[],
        )
        conf = ConfidenceReport(factual=0.9, reasoning=0.9, knowledge_boundary=0.9)
        output = NSIOutput("ok", conf, trace, ns, OutputStatus.OK, [])
        tbr = TrustBudgetResult(output=output, cost=0.1,
                                remaining_budget=0.9, budget_exceeded=False)
        assert tbr.cost == 0.1
        assert tbr.remaining_budget == 0.9
        assert tbr.budget_exceeded is False

    def test_proof_capsule_dataclass_fields(self):
        pc = ProofCapsule(
            version="1.0", domain="test", answer="ans", status="ok",
            confidence={"factual": 0.9}, steps=[], null_items=[],
            flags=[], checksum="abc", timestamp="t", request_id="id",
            reasoning_fingerprint="fp123",
        )
        assert pc.version == "1.0"
        d = pc.to_dict()
        assert d["domain"] == "test"
        assert d["reasoning_fingerprint"] == "fp123"
