"""
tests/unit/test_nsil.py
========================
Comprehensive tests for NSIL (Neural–Symbolic Integrity Link) — Revolution #7.

Coverage targets:
    - nesy/neural/nsil.py → 100%
    - nesy/neural/bridge.py  (assess_integrity path)
    - nesy/metacognition/monitor.py (integrity coupling path)

Test categories (behavioural, not coverage-only):
    1. Symbolic-only mode → neutral integrity, no penalty
    2. Grounded evidence strong → no penalty
    3. Mismatch (symbolic derived, no evidence) → penalty + flag
    4. Clamping behaviour (NaN/Inf sim, out-of-range)
    5. Determinism (run twice, identical report)
    6. Bridge.assess_integrity forwarding
    7. Monitor integrity coupling to boundary confidence
"""

import pytest

from nesy.core.types import (
    GroundedSymbol,
    NullItem,
    NullSet,
    NullType,
    OutputStatus,
    Predicate,
    PresentSet,
    ReasoningStep,
    SymbolicRule,
)
from nesy.metacognition.monitor import MetaCognitionMonitor
from nesy.neural.bridge import NeuralSymbolicBridge
from nesy.neural.grounding import PredicatePrototype, SymbolGrounder
from nesy.neural.nsil import (
    IntegrityItem,
    IntegrityReport,
    _best_proto_sim,
    _clamp01,
    _predicate_schema,
    _predicate_schemas_present,
    _predicates_required_by_rules,
    compute_integrity_report,
)


# ═══════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def symptom_pred():
    return Predicate("HasSymptom", ("?p", "fever"))


@pytest.fixture
def diagnosis_pred():
    return Predicate("PossiblyHas", ("?p", "infection"))


@pytest.fixture
def simple_rule(symptom_pred, diagnosis_pred):
    return SymbolicRule(
        id="fever_infection",
        antecedents=[symptom_pred],
        consequents=[diagnosis_pred],
        weight=0.85,
        description="Fever → infection",
    )


@pytest.fixture
def grounded_symptom(symptom_pred):
    return GroundedSymbol(
        predicate=symptom_pred,
        embedding=[1.0, 0.0, 0.0],
        grounding_confidence=0.92,
    )


@pytest.fixture
def grounded_diagnosis(diagnosis_pred):
    return GroundedSymbol(
        predicate=diagnosis_pred,
        embedding=[0.0, 1.0, 0.0],
        grounding_confidence=0.88,
    )


@pytest.fixture
def empty_null_set():
    return NullSet(
        items=[],
        present_set=PresentSet(concepts=set(), context_type="general"),
    )


@pytest.fixture
def null_set_with_critical():
    return NullSet(
        items=[
            NullItem(
                concept="blood_test",
                weight=0.9,
                null_type=NullType.TYPE3_CRITICAL,
                expected_because_of=["fever"],
                criticality=1.0,
            ),
        ],
        present_set=PresentSet(concepts={"fever"}, context_type="medical"),
    )


# ═══════════════════════════════════════════════════════════════════
#  HELPER FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestClamp01:
    def test_normal_value(self):
        assert _clamp01(0.5) == 0.5

    def test_zero(self):
        assert _clamp01(0.0) == 0.0

    def test_one(self):
        assert _clamp01(1.0) == 1.0

    def test_negative(self):
        assert _clamp01(-0.5) == 0.0

    def test_above_one(self):
        assert _clamp01(1.5) == 1.0

    def test_nan(self):
        assert _clamp01(float("nan")) == 0.0

    def test_positive_inf(self):
        assert _clamp01(float("inf")) == 0.0

    def test_negative_inf(self):
        assert _clamp01(float("-inf")) == 0.0


class TestPredicateSchema:
    def test_extracts_name(self, symptom_pred):
        assert _predicate_schema(symptom_pred) == "HasSymptom"

    def test_no_args(self):
        p = Predicate("Alive", ())
        assert _predicate_schema(p) == "Alive"


class TestPredicatesRequiredByRules:
    def test_collects_antecedents_and_consequents(self, simple_rule):
        schemas = _predicates_required_by_rules([simple_rule])
        assert "HasSymptom" in schemas
        assert "PossiblyHas" in schemas

    def test_empty_rules(self):
        assert _predicates_required_by_rules([]) == set()

    def test_multiple_rules(self, simple_rule):
        rule2 = SymbolicRule(
            id="r2",
            antecedents=[Predicate("A", ())],
            consequents=[Predicate("B", ())],
            weight=0.5,
        )
        schemas = _predicates_required_by_rules([simple_rule, rule2])
        assert schemas == {"HasSymptom", "PossiblyHas", "A", "B"}


class TestPredicateSchemasPresent:
    def test_from_predicate_set(self, symptom_pred, diagnosis_pred):
        schemas = _predicate_schemas_present([symptom_pred, diagnosis_pred])
        assert schemas == {"HasSymptom", "PossiblyHas"}

    def test_empty(self):
        assert _predicate_schemas_present([]) == set()


class TestBestProtoSim:
    def test_finds_best_match(self, grounded_symptom, grounded_diagnosis):
        sim = _best_proto_sim("HasSymptom", [grounded_symptom, grounded_diagnosis])
        assert sim == pytest.approx(0.92)

    def test_no_match(self, grounded_diagnosis):
        sim = _best_proto_sim("HasSymptom", [grounded_diagnosis])
        assert sim == 0.0

    def test_empty_grounded(self):
        assert _best_proto_sim("X", []) == 0.0

    def test_clamps_high_confidence(self):
        gs = GroundedSymbol(
            predicate=Predicate("X", ()),
            embedding=[1.0],
            grounding_confidence=1.5,
        )
        assert _best_proto_sim("X", [gs]) == 1.0

    def test_clamps_nan_confidence(self):
        gs = GroundedSymbol(
            predicate=Predicate("X", ()),
            embedding=[1.0],
            grounding_confidence=float("nan"),
        )
        assert _best_proto_sim("X", [gs]) == 0.0


# ═══════════════════════════════════════════════════════════════════
#  INTEGRITY ITEM / REPORT DATACLASS TESTS
# ═══════════════════════════════════════════════════════════════════


class TestIntegrityItem:
    def test_fields(self):
        item = IntegrityItem(
            schema="HasSymptom",
            evidence=0.9,
            membership=1.0,
            need=1.0,
            residual=0.1,
        )
        assert item.schema == "HasSymptom"
        assert item.residual == pytest.approx(0.1)


class TestIntegrityReport:
    def test_default_neutral(self):
        r = IntegrityReport()
        assert r.integrity_score == 1.0
        assert r.is_neutral is False
        assert r.items == []
        assert r.flags == []
        assert r.suggestions == []

    def test_neutral_report(self):
        r = IntegrityReport(integrity_score=1.0, is_neutral=True)
        assert r.is_neutral is True


# ═══════════════════════════════════════════════════════════════════
#  CORE compute_integrity_report TESTS
# ═══════════════════════════════════════════════════════════════════


class TestPassthroughMode:
    """Symbolic-only mode → neutral integrity, no penalty."""

    def test_passthrough_returns_neutral(self, simple_rule, symptom_pred):
        report = compute_integrity_report(
            grounded=[],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred},
            is_passthrough=True,
        )
        assert report.is_neutral is True
        assert report.integrity_score == 1.0
        assert report.items == []
        assert report.flags == []

    def test_passthrough_with_grounded_still_neutral(
        self, simple_rule, symptom_pred, grounded_symptom
    ):
        """Even with grounded evidence, passthrough → neutral."""
        report = compute_integrity_report(
            grounded=[grounded_symptom],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred},
            is_passthrough=True,
        )
        assert report.is_neutral is True
        assert report.integrity_score == 1.0


class TestStrongEvidence:
    """Grounded evidence strong → no penalty."""

    def test_perfect_alignment(
        self,
        simple_rule,
        symptom_pred,
        diagnosis_pred,
        grounded_symptom,
        grounded_diagnosis,
    ):
        """All required predicates grounded with high confidence → score ~1.0."""
        report = compute_integrity_report(
            grounded=[grounded_symptom, grounded_diagnosis],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred, diagnosis_pred},
        )
        assert report.integrity_score > 0.85
        assert report.is_neutral is False
        assert "NSIL_LOW_INTEGRITY" not in report.flags

    def test_derived_with_matching_evidence(self, symptom_pred, grounded_symptom):
        """Derived fact with grounding evidence → low residual."""
        report = compute_integrity_report(
            grounded=[grounded_symptom],
            activated_rules=[],
            derived_facts={symptom_pred},
        )
        # derived + evidence → |1.0 - 0.92| × 0.5 = 0.04
        assert report.integrity_score > 0.9


class TestMismatch:
    """Mismatch (symbolic derives, no evidence) → penalty + flag."""

    def test_derived_no_evidence_penalty(self, simple_rule, symptom_pred, diagnosis_pred):
        """Derived facts with zero grounding evidence → ISR high → penalty."""
        report = compute_integrity_report(
            grounded=[],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred, diagnosis_pred},
        )
        # m=1.0, e=0.0 → ISR = 1.0 * |1.0 - 0.0| = 1.0
        assert report.integrity_score < 0.5
        assert "NSIL_LOW_INTEGRITY" in report.flags

    def test_weak_evidence_flag(self, symptom_pred, diagnosis_pred):
        """Derived fact with very weak evidence → NSIL_WEAK_EVIDENCE flag."""
        weak_gs = GroundedSymbol(
            predicate=diagnosis_pred,
            embedding=[0.0, 1.0, 0.0],
            grounding_confidence=0.1,
        )
        rule = SymbolicRule(
            id="r1",
            antecedents=[symptom_pred],
            consequents=[diagnosis_pred],
            weight=0.9,
        )
        report = compute_integrity_report(
            grounded=[weak_gs],
            activated_rules=[rule],
            derived_facts={diagnosis_pred},
        )
        weak_flags = [f for f in report.flags if "NSIL_WEAK_EVIDENCE" in f]
        assert len(weak_flags) >= 1

    def test_unused_evidence_flag(self, symptom_pred, grounded_symptom):
        """Strong grounding for a predicate that was NOT derived → flag."""
        rule = SymbolicRule(
            id="r1",
            antecedents=[symptom_pred],
            consequents=[Predicate("Other", ())],
            weight=0.9,
        )
        report = compute_integrity_report(
            grounded=[grounded_symptom],
            activated_rules=[rule],
            derived_facts=set(),
        )
        unused = [f for f in report.flags if "NSIL_UNUSED_EVIDENCE" in f]
        assert len(unused) >= 1

    def test_penalty_affects_score(self, simple_rule, symptom_pred, diagnosis_pred):
        """Verify the actual score formula with a known computation."""
        report = compute_integrity_report(
            grounded=[],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred, diagnosis_pred},
        )
        # Both schemas: ISR = 1.0 each → avg=1.0, max=1.0
        # score = clamp01(1.0 - (0.7*1.0 + 0.3*1.0)) = clamp01(0.0) = 0.0
        assert report.integrity_score == pytest.approx(0.0)


class TestClamping:
    """Clamping behaviour (NaN/Inf sim, out-of-range)."""

    def test_nan_grounding_confidence(self, symptom_pred):
        """NaN grounding_confidence → clamped to 0.0."""
        gs = GroundedSymbol(
            predicate=symptom_pred,
            embedding=[1.0],
            grounding_confidence=float("nan"),
        )
        report = compute_integrity_report(
            grounded=[gs],
            activated_rules=[],
            derived_facts={symptom_pred},
        )
        item = report.items[0]
        assert item.evidence == 0.0

    def test_inf_grounding_confidence(self, symptom_pred):
        """Inf grounding_confidence → clamped to 0.0 (not 1.0)."""
        gs = GroundedSymbol(
            predicate=symptom_pred,
            embedding=[1.0],
            grounding_confidence=float("inf"),
        )
        report = compute_integrity_report(
            grounded=[gs],
            activated_rules=[],
            derived_facts={symptom_pred},
        )
        item = report.items[0]
        assert item.evidence == 0.0

    def test_negative_confidence_clamped(self, symptom_pred):
        gs = GroundedSymbol(
            predicate=symptom_pred,
            embedding=[1.0],
            grounding_confidence=-0.5,
        )
        report = compute_integrity_report(
            grounded=[gs],
            activated_rules=[],
            derived_facts={symptom_pred},
        )
        item = report.items[0]
        assert item.evidence == 0.0

    def test_above_one_confidence_clamped(self, symptom_pred):
        gs = GroundedSymbol(
            predicate=symptom_pred,
            embedding=[1.0],
            grounding_confidence=1.5,
        )
        report = compute_integrity_report(
            grounded=[gs],
            activated_rules=[],
            derived_facts={symptom_pred},
        )
        item = report.items[0]
        assert item.evidence == 1.0

    def test_score_never_below_zero(self, simple_rule, symptom_pred, diagnosis_pred):
        """Score must be in [0, 1] even with worst-case residuals."""
        report = compute_integrity_report(
            grounded=[],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred, diagnosis_pred},
        )
        assert 0.0 <= report.integrity_score <= 1.0


class TestDeterminism:
    """Determinism (run twice, identical report)."""

    def test_identical_reports(
        self,
        simple_rule,
        symptom_pred,
        grounded_symptom,
        empty_null_set,
    ):
        kwargs = dict(
            grounded=[grounded_symptom],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred},
            null_set=empty_null_set,
        )
        r1 = compute_integrity_report(**kwargs)
        r2 = compute_integrity_report(**kwargs)

        assert r1.integrity_score == r2.integrity_score
        assert len(r1.items) == len(r2.items)
        for a, b in zip(r1.items, r2.items):
            assert a.schema == b.schema
            assert a.residual == b.residual
            assert a.evidence == b.evidence
            assert a.membership == b.membership
            assert a.need == b.need
        assert r1.flags == r2.flags

    def test_determinism_with_different_grounding_order(
        self,
        simple_rule,
        symptom_pred,
        diagnosis_pred,
        grounded_symptom,
        grounded_diagnosis,
    ):
        """Reports must be identical regardless of grounded symbol order."""
        r1 = compute_integrity_report(
            grounded=[grounded_symptom, grounded_diagnosis],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred, diagnosis_pred},
        )
        r2 = compute_integrity_report(
            grounded=[grounded_diagnosis, grounded_symptom],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred, diagnosis_pred},
        )
        assert r1.integrity_score == r2.integrity_score
        assert [i.schema for i in r1.items] == [i.schema for i in r2.items]


class TestNullSetEnrichment:
    """NSIL enrichment with critical null items."""

    def test_critical_null_adds_suggestion(
        self,
        simple_rule,
        symptom_pred,
        null_set_with_critical,
    ):
        report = compute_integrity_report(
            grounded=[],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred},
            null_set=null_set_with_critical,
        )
        assert any("blood_test" in s for s in report.suggestions)

    def test_empty_null_set_no_suggestion(
        self,
        simple_rule,
        symptom_pred,
        empty_null_set,
    ):
        report = compute_integrity_report(
            grounded=[],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred},
            null_set=empty_null_set,
        )
        assert all("critical" not in s.lower() for s in report.suggestions)


class TestEdgeCases:
    """Edge cases for compute_integrity_report."""

    def test_empty_everything_neutral(self):
        """No rules, no grounded, no derived → neutral."""
        report = compute_integrity_report(
            grounded=[],
            activated_rules=[],
            derived_facts=set(),
        )
        assert report.is_neutral is True
        assert report.integrity_score == 1.0

    def test_only_derived_no_rules(self, symptom_pred):
        """Derived without rules → need = 0.5 (lower requirement)."""
        report = compute_integrity_report(
            grounded=[],
            activated_rules=[],
            derived_facts={symptom_pred},
        )
        assert report.items[0].need == 0.5

    def test_custom_doubt_threshold(self, simple_rule, symptom_pred):
        """Custom doubt threshold affects flag generation."""
        report = compute_integrity_report(
            grounded=[],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred},
            doubt_threshold=0.99,
        )
        assert "NSIL_LOW_INTEGRITY" in report.flags

    def test_need_is_one_for_required(self, simple_rule, symptom_pred):
        """Schemas required by rules should have need=1.0."""
        report = compute_integrity_report(
            grounded=[],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred},
        )
        for item in report.items:
            if item.schema in ("HasSymptom", "PossiblyHas"):
                assert item.need == 1.0


# ═══════════════════════════════════════════════════════════════════
#  BRIDGE INTEGRATION
# ═══════════════════════════════════════════════════════════════════


class TestBridgeAssessIntegrity:
    """NeuralSymbolicBridge.assess_integrity delegates to compute_integrity_report."""

    @pytest.fixture
    def bridge(self):
        grounder = SymbolGrounder(threshold=0.7)
        grounder.register(
            PredicatePrototype(
                predicate=Predicate("HasSymptom", ("?p", "fever")),
                prototype=[1.0, 0.0, 0.0],
                domain="medical",
            )
        )
        return NeuralSymbolicBridge(grounder)

    def test_returns_report(self, bridge, simple_rule, symptom_pred, grounded_symptom):
        report = bridge.assess_integrity(
            grounded=[grounded_symptom],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred},
        )
        assert isinstance(report, IntegrityReport)
        assert report.is_neutral is False

    def test_passthrough_mode(self, bridge, simple_rule, symptom_pred):
        report = bridge.assess_integrity(
            grounded=[],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred},
            is_passthrough=True,
        )
        assert report.is_neutral is True
        assert report.integrity_score == 1.0

    def test_null_set_forwarded(self, bridge, simple_rule, symptom_pred, null_set_with_critical):
        report = bridge.assess_integrity(
            grounded=[],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred},
            null_set=null_set_with_critical,
        )
        assert any("blood_test" in s for s in report.suggestions)

    def test_custom_doubt_threshold(self, bridge, simple_rule, symptom_pred):
        report = bridge.assess_integrity(
            grounded=[],
            activated_rules=[simple_rule],
            derived_facts={symptom_pred},
            doubt_threshold=0.99,
        )
        assert "NSIL_LOW_INTEGRITY" in report.flags


# ═══════════════════════════════════════════════════════════════════
#  MONITOR INTEGRATION
# ═══════════════════════════════════════════════════════════════════


class TestMonitorIntegrityCoupling:
    """MetaCognitionMonitor applies NSIL integrity coupling to boundary."""

    @pytest.fixture
    def monitor(self):
        return MetaCognitionMonitor(doubt_threshold=0.5, strict_mode=False)

    @pytest.fixture
    def minimal_args(self):
        return dict(
            answer="test",
            neural_confidence=0.9,
            symbolic_confidence=0.9,
            reasoning_steps=[
                ReasoningStep(
                    step_number=0,
                    description="Initial",
                    rule_applied=None,
                    predicates=[],
                    confidence=1.0,
                )
            ],
            logic_clauses=[],
            null_set=NullSet(
                items=[],
                present_set=PresentSet(concepts=set(), context_type="general"),
            ),
        )

    def test_no_integrity_report_no_change(self, monitor, minimal_args):
        """Without integrity_report, boundary is unchanged."""
        conf, _, _, flags = monitor.evaluate(**minimal_args)
        assert conf.knowledge_boundary > 0.0
        assert "NSIL_LOW_INTEGRITY" not in flags

    def test_neutral_report_no_change(self, monitor, minimal_args):
        """Neutral report (passthrough) → no boundary penalty."""
        neutral = IntegrityReport(integrity_score=1.0, is_neutral=True)
        conf_without, _, _, _ = monitor.evaluate(**minimal_args)
        conf_with, _, _, flags = monitor.evaluate(**minimal_args, integrity_report=neutral)
        assert conf_with.knowledge_boundary == conf_without.knowledge_boundary
        assert "NSIL_LOW_INTEGRITY" not in flags

    def test_low_integrity_reduces_boundary(self, monitor, minimal_args):
        """Low integrity score → boundary reduced proportionally."""
        low = IntegrityReport(
            integrity_score=0.3,
            items=[],
            flags=["NSIL_LOW_INTEGRITY"],
            is_neutral=False,
        )
        conf_without, _, _, _ = monitor.evaluate(**minimal_args)
        conf_with, _, _, flags = monitor.evaluate(**minimal_args, integrity_report=low)
        assert conf_with.knowledge_boundary < conf_without.knowledge_boundary
        assert conf_with.knowledge_boundary == pytest.approx(
            conf_without.knowledge_boundary * 0.3, abs=0.01
        )
        assert "NSIL_LOW_INTEGRITY" in flags

    def test_perfect_integrity_no_penalty(self, monitor, minimal_args):
        """Perfect integrity score → boundary unchanged."""
        perfect = IntegrityReport(
            integrity_score=1.0,
            items=[],
            flags=[],
            is_neutral=False,
        )
        conf_without, _, _, _ = monitor.evaluate(**minimal_args)
        conf_with, _, _, _ = monitor.evaluate(**minimal_args, integrity_report=perfect)
        assert conf_with.knowledge_boundary == pytest.approx(
            conf_without.knowledge_boundary, abs=0.01
        )

    def test_integrity_flags_propagated(self, monitor, minimal_args):
        """NSIL flags are propagated to the output flags list."""
        report = IntegrityReport(
            integrity_score=0.5,
            items=[],
            flags=[
                "NSIL_LOW_INTEGRITY",
                "NSIL_WEAK_EVIDENCE: 'X' derived but grounding evidence is 0.05",
            ],
            is_neutral=False,
        )
        _, _, _, flags = monitor.evaluate(**minimal_args, integrity_report=report)
        assert "NSIL_LOW_INTEGRITY" in flags
        assert any("NSIL_WEAK_EVIDENCE" in f for f in flags)

    def test_integrity_can_trigger_flagged_status(self, monitor, minimal_args):
        """Very low integrity → boundary drops below doubt → FLAGGED."""
        very_low = IntegrityReport(
            integrity_score=0.1,
            items=[],
            flags=["NSIL_LOW_INTEGRITY"],
            is_neutral=False,
        )
        _, _, status, _ = monitor.evaluate(**minimal_args, integrity_report=very_low)
        # confidence.minimum will be gated by boundary * 0.1
        assert status in (OutputStatus.FLAGGED, OutputStatus.UNCERTAIN)


# ═══════════════════════════════════════════════════════════════════
#  BRIDGE FULL COVERAGE TESTS
# ═══════════════════════════════════════════════════════════════════


class TestBridgeNeuralToSymbolic:
    """Cover neural_to_symbolic success and empty paths."""

    @pytest.fixture
    def bridge_with_protos(self):
        grounder = SymbolGrounder(threshold=0.7)
        grounder.register(
            PredicatePrototype(
                predicate=Predicate("HasSymptom", ("?p", "fever")),
                prototype=[1.0, 0.0, 0.0],
                domain="medical",
            )
        )
        return NeuralSymbolicBridge(grounder)

    def test_success_path(self, bridge_with_protos):
        """Identical embedding → grounded predicate returned."""
        preds, conf = bridge_with_protos.neural_to_symbolic([1.0, 0.0, 0.0])
        assert len(preds) == 1
        assert conf == pytest.approx(1.0, abs=0.01)
        assert Predicate("HasSymptom", ("?p", "fever")) in preds

    def test_empty_path(self, bridge_with_protos):
        """Orthogonal embedding → empty set, 0.0."""
        preds, conf = bridge_with_protos.neural_to_symbolic([0.0, 0.0, 1.0])
        assert preds == set()
        assert conf == 0.0

    def test_domain_filter(self, bridge_with_protos):
        preds, conf = bridge_with_protos.neural_to_symbolic([1.0, 0.0, 0.0], domain="medical")
        assert len(preds) >= 1


class TestBridgeConstraintGradient:
    """Cover symbolic_constraint_gradient with and without prototype match."""

    @pytest.fixture
    def bridge_with_proto(self):
        grounder = SymbolGrounder(threshold=0.7)
        grounder.register(
            PredicatePrototype(
                predicate=Predicate("Target", ()),
                prototype=[1.0, 0.0, 0.0],
            )
        )
        return NeuralSymbolicBridge(grounder)

    def test_no_violated_rules(self, bridge_with_proto):
        grad = bridge_with_proto.symbolic_constraint_gradient([0.0, 0.0, 0.0], [])
        assert grad == [0.0, 0.0, 0.0]

    def test_with_violated_rules_and_proto_match(self, bridge_with_proto):
        """Rule consequent matches a registered prototype → gradient ≠ 0."""
        rule = SymbolicRule(
            id="r1",
            antecedents=[Predicate("A", ())],
            consequents=[Predicate("Target", ())],
            weight=1.0,
        )
        grad = bridge_with_proto.symbolic_constraint_gradient(
            [0.0, 0.0, 0.0],
            [rule],
            step_size=0.1,
        )
        # Gradient pushes toward [1.0, 0.0, 0.0]
        assert grad[0] > 0.0  # should push toward 1.0

    def test_no_proto_match_zero_gradient(self):
        """Rule consequent has no registered prototype → gradient = 0."""
        grounder = SymbolGrounder(threshold=0.7)
        bridge = NeuralSymbolicBridge(grounder)
        rule = SymbolicRule(
            id="r1",
            antecedents=[Predicate("A", ())],
            consequents=[Predicate("NoProto", ())],
            weight=1.0,
        )
        grad = bridge.symbolic_constraint_gradient(
            [0.5, 0.5, 0.5],
            [rule],
            step_size=0.1,
        )
        assert grad == [0.0, 0.0, 0.0]


class TestBridgeSymbolicToLoss:
    """Cover symbolic_to_loss fully."""

    @pytest.fixture
    def bridge(self):
        grounder = SymbolGrounder(threshold=0.7)
        return NeuralSymbolicBridge(grounder)

    def test_no_violations(self, bridge):
        loss = bridge.symbolic_to_loss([0.0, 0.0], [])
        assert loss == 0.0

    def test_with_violations(self, bridge):
        rule = SymbolicRule(
            id="r1",
            antecedents=[Predicate("A", ())],
            consequents=[Predicate("B", ())],
            weight=0.8,
        )
        loss = bridge.symbolic_to_loss([0.0, 0.0], [rule])
        # satisfaction=0 → hinge=1 → loss = 0.5 * 0.8 * 1 = 0.4
        assert loss == pytest.approx(0.4)
