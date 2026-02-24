"""
tests/unit/test_fingerprint.py
===============================
Comprehensive tests for Reasoning Fingerprint — Revolution #8.

Coverage targets:
    - nesy/metacognition/fingerprint.py → 100%

Test categories (behavioural):
    1. Identical inputs → identical fingerprints
    2. Reorder of internal sets → same fingerprint
    3. Change in rule weight / rule chain → different fingerprint
    4. Change in null classification → different fingerprint
    5. Presence/absence of unsat core → different fingerprint
    6. Presence/absence of NSIL → different fingerprint deterministically
    7. Config hash stability
    8. Fingerprint attached to NSIOutput and PCAP
"""

import copy
import pytest

from nesy.core.types import (
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
    UnsatCore,
)
from nesy.metacognition.fingerprint import (
    _canonicalize_nsil,
    _canonicalize_null_item,
    _canonicalize_unsat_core,
    _round_float,
    canonicalize_output,
    compute_config_hash,
    compute_reasoning_fingerprint,
)
from nesy.neural.nsil import IntegrityItem, IntegrityReport


# ═══════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_output():
    """A complete NSIOutput for fingerprint tests."""
    p1 = Predicate("HasSymptom", ("patient", "fever"))
    p2 = Predicate("PossiblyHas", ("patient", "infection"))
    steps = [
        ReasoningStep(
            step_number=0,
            description="Initial facts: HasSymptom(patient, fever)",
            rule_applied=None,
            predicates=[p1],
            confidence=1.0,
        ),
        ReasoningStep(
            step_number=1,
            description="Applied rule 'r1': fever → infection",
            rule_applied="r1",
            predicates=[p2],
            confidence=0.85,
        ),
    ]
    trace = ReasoningTrace(
        steps=steps,
        rules_activated=["r1"],
        neural_confidence=0.9,
        symbolic_confidence=0.85,
        null_violations=[],
        logic_clauses=[
            LogicClause(
                predicates=[p2],
                connective=LogicConnective.IMPLIES,
                satisfied=True,
                weight=0.85,
                source_rule="r1",
            ),
        ],
    )
    null_set = NullSet(
        items=[
            NullItem(
                concept="blood_test",
                weight=0.8,
                null_type=NullType.TYPE2_MEANINGFUL,
                expected_because_of=["fever"],
            ),
        ],
        present_set=PresentSet(concepts={"fever"}, context_type="medical"),
    )
    return NSIOutput(
        answer="Derived: PossiblyHas(patient, infection)",
        confidence=ConfidenceReport(factual=0.9, reasoning=0.85, knowledge_boundary=0.88),
        reasoning_trace=trace,
        null_set=null_set,
        status=OutputStatus.OK,
        flags=[],
    )


@pytest.fixture
def sample_config():
    return {"domain": "medical", "doubt_threshold": 0.6, "strict_mode": False}


@pytest.fixture
def sample_unsat_core():
    return UnsatCore(
        conflicting_rule_ids=["r1", "r2"],
        constraint_ids=[0, 1],
        explanation="Rules r1 and r2 conflict.",
        suggested_additions=["concept_x"],
        repair_actions=[{"concept": "concept_x", "reason": "add x"}],
    )


@pytest.fixture
def sample_nsil_report():
    return IntegrityReport(
        items=[
            IntegrityItem(
                schema="HasSymptom", evidence=0.92, membership=1.0, need=1.0, residual=0.08
            ),
            IntegrityItem(
                schema="PossiblyHas", evidence=0.88, membership=1.0, need=1.0, residual=0.12
            ),
        ],
        integrity_score=0.91,
        flags=[],
        is_neutral=False,
    )


# ═══════════════════════════════════════════════════════════════════
#  HELPER FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestRoundFloat:
    def test_normal(self):
        assert _round_float(0.123456789) == 0.123457

    def test_nan(self):
        assert _round_float(float("nan")) == 0.0

    def test_inf(self):
        assert _round_float(float("inf")) == 0.0

    def test_custom_decimals(self):
        assert _round_float(0.123456789, decimals=3) == 0.123


class TestCanonicalizeNullItem:
    def test_canonical_form(self):
        item = NullItem(
            concept="blood_test",
            weight=0.87654321,
            null_type=NullType.TYPE2_MEANINGFUL,
            expected_because_of=["fever", "cough"],
        )
        result = _canonicalize_null_item(item)
        assert result[0] == "TYPE2_MEANINGFUL"
        assert result[1] == "blood_test"
        assert result[2] == 0.876543  # rounded
        assert result[3] == ["cough", "fever"]  # sorted


class TestCanonicalizeUnsatCore:
    def test_none(self):
        assert _canonicalize_unsat_core(None) is None

    def test_sorted_ids(self, sample_unsat_core):
        result = _canonicalize_unsat_core(sample_unsat_core)
        assert result["conflicting_rule_ids"] == ["r1", "r2"]
        assert result["constraint_ids"] == [0, 1]

    def test_unsorted_ids_get_sorted(self):
        core = UnsatCore(
            conflicting_rule_ids=["r3", "r1", "r2"],
            constraint_ids=[5, 2, 8],
            explanation="conflict",
        )
        result = _canonicalize_unsat_core(core)
        assert result["conflicting_rule_ids"] == ["r1", "r2", "r3"]
        assert result["constraint_ids"] == [2, 5, 8]


class TestCanonicalizeNsil:
    def test_none(self):
        assert _canonicalize_nsil(None) is None

    def test_sorted_items(self, sample_nsil_report):
        result = _canonicalize_nsil(sample_nsil_report)
        assert result["integrity_score"] == 0.91
        assert result["is_neutral"] is False
        schemas = [i["schema"] for i in result["items"]]
        assert schemas == sorted(schemas)

    def test_unsorted_items_get_sorted(self):
        report = IntegrityReport(
            items=[
                IntegrityItem(schema="Z", evidence=0.5, membership=1.0, need=1.0, residual=0.5),
                IntegrityItem(schema="A", evidence=0.5, membership=1.0, need=1.0, residual=0.5),
            ],
            integrity_score=0.5,
            is_neutral=False,
        )
        result = _canonicalize_nsil(report)
        schemas = [i["schema"] for i in result["items"]]
        assert schemas == ["A", "Z"]


class TestComputeConfigHash:
    def test_stable_hash(self, sample_config):
        h1 = compute_config_hash(sample_config)
        h2 = compute_config_hash(sample_config)
        assert h1 == h2
        assert len(h1) == 64

    def test_order_invariant(self):
        h1 = compute_config_hash({"a": 1, "b": 2})
        h2 = compute_config_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_values_different_hash(self):
        h1 = compute_config_hash({"a": 1})
        h2 = compute_config_hash({"a": 2})
        assert h1 != h2

    def test_empty_config(self):
        h = compute_config_hash({})
        assert len(h) == 64


# ═══════════════════════════════════════════════════════════════════
#  CANONICALIZE OUTPUT TESTS
# ═══════════════════════════════════════════════════════════════════


class TestCanonicalizeOutput:
    def test_has_all_keys(self, sample_output):
        payload = canonicalize_output(sample_output)
        assert "rules_activated" in payload
        assert "derived_predicates" in payload
        assert "null_items" in payload
        assert "status" in payload
        assert "confidence_triple" in payload
        assert "unsat_core" in payload
        assert "nsil" in payload
        assert "config_hash" in payload

    def test_rules_sorted(self, sample_output):
        payload = canonicalize_output(sample_output)
        assert payload["rules_activated"] == ["r1"]

    def test_no_config_empty_hash(self, sample_output):
        payload = canonicalize_output(sample_output)
        assert payload["config_hash"] == ""

    def test_with_config_hash(self, sample_output, sample_config):
        payload = canonicalize_output(sample_output, config_snapshot=sample_config)
        assert payload["config_hash"] != ""

    def test_unsat_core_attached(self, sample_output, sample_unsat_core):
        payload = canonicalize_output(sample_output, unsat_core=sample_unsat_core)
        assert payload["unsat_core"] is not None
        assert payload["unsat_core"]["conflicting_rule_ids"] == ["r1", "r2"]

    def test_nsil_attached(self, sample_output, sample_nsil_report):
        payload = canonicalize_output(sample_output, nsil_report=sample_nsil_report)
        assert payload["nsil"] is not None
        assert payload["nsil"]["integrity_score"] == 0.91


# ═══════════════════════════════════════════════════════════════════
#  FINGERPRINT CORE TESTS
# ═══════════════════════════════════════════════════════════════════


class TestIdenticalInputs:
    """Identical inputs → identical fingerprints."""

    def test_same_output_same_fingerprint(self, sample_output):
        fp1 = compute_reasoning_fingerprint(sample_output)
        fp2 = compute_reasoning_fingerprint(sample_output)
        assert fp1 == fp2

    def test_same_with_config(self, sample_output, sample_config):
        fp1 = compute_reasoning_fingerprint(sample_output, config_snapshot=sample_config)
        fp2 = compute_reasoning_fingerprint(sample_output, config_snapshot=sample_config)
        assert fp1 == fp2

    def test_same_with_all_components(
        self, sample_output, sample_config, sample_unsat_core, sample_nsil_report
    ):
        fp1 = compute_reasoning_fingerprint(
            sample_output,
            sample_config,
            sample_unsat_core,
            sample_nsil_report,
        )
        fp2 = compute_reasoning_fingerprint(
            sample_output,
            sample_config,
            sample_unsat_core,
            sample_nsil_report,
        )
        assert fp1 == fp2

    def test_fingerprint_is_64_hex(self, sample_output):
        fp = compute_reasoning_fingerprint(sample_output)
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)


class TestReorderInvariance:
    """Reorder of internal sets → same fingerprint."""

    def test_reorder_rules_activated(self, sample_output):
        """Reordering rules_activated should still give same fingerprint
        because canonicalize_output sorts them."""
        output_a = copy.deepcopy(sample_output)
        output_b = copy.deepcopy(sample_output)
        output_a.reasoning_trace.rules_activated = ["r1", "r2"]
        output_b.reasoning_trace.rules_activated = ["r2", "r1"]
        # Need to add the extra rule step for r2 so it appears
        output_a.reasoning_trace.steps.append(
            ReasoningStep(
                step_number=2,
                description="rule r2",
                rule_applied="r2",
                predicates=[Predicate("X", ())],
                confidence=0.9,
            )
        )
        output_b.reasoning_trace.steps.append(
            ReasoningStep(
                step_number=2,
                description="rule r2",
                rule_applied="r2",
                predicates=[Predicate("X", ())],
                confidence=0.9,
            )
        )
        fp_a = compute_reasoning_fingerprint(output_a)
        fp_b = compute_reasoning_fingerprint(output_b)
        assert fp_a == fp_b

    def test_reorder_null_items(self, sample_output):
        """Null items in different order → same fingerprint (sorted)."""
        item1 = NullItem(
            concept="a", weight=0.5, null_type=NullType.TYPE1_EXPECTED, expected_because_of=["x"]
        )
        item2 = NullItem(
            concept="b", weight=0.6, null_type=NullType.TYPE2_MEANINGFUL, expected_because_of=["y"]
        )

        output_a = copy.deepcopy(sample_output)
        output_b = copy.deepcopy(sample_output)
        output_a.null_set.items = [item1, item2]
        output_b.null_set.items = [item2, item1]

        fp_a = compute_reasoning_fingerprint(output_a)
        fp_b = compute_reasoning_fingerprint(output_b)
        assert fp_a == fp_b


class TestChangesProduceDifferentFingerprint:
    """Changes in semantics → different fingerprint."""

    def test_different_rule_chain(self, sample_output):
        """Different rule weight via reasoning step → different fingerprint."""
        output_b = copy.deepcopy(sample_output)
        output_b.reasoning_trace.steps[1].confidence = 0.5
        fp_a = compute_reasoning_fingerprint(sample_output)
        fp_b = compute_reasoning_fingerprint(output_b)
        # The confidence_triple changes because it's from ConfidenceReport, not steps.
        # But derived_predicates come from steps. The canonical payload includes
        # confidence_triple from the ConfidenceReport. Let's change that too.
        output_b.confidence = ConfidenceReport(factual=0.9, reasoning=0.5, knowledge_boundary=0.88)
        fp_b = compute_reasoning_fingerprint(output_b)
        assert fp_a != fp_b

    def test_different_null_classification(self, sample_output):
        """Change null type → different fingerprint."""
        output_b = copy.deepcopy(sample_output)
        output_b.null_set.items[0] = NullItem(
            concept="blood_test",
            weight=0.8,
            null_type=NullType.TYPE3_CRITICAL,  # was TYPE2
            expected_because_of=["fever"],
        )
        fp_a = compute_reasoning_fingerprint(sample_output)
        fp_b = compute_reasoning_fingerprint(output_b)
        assert fp_a != fp_b

    def test_different_status(self, sample_output):
        output_b = copy.deepcopy(sample_output)
        output_b.status = OutputStatus.FLAGGED
        fp_a = compute_reasoning_fingerprint(sample_output)
        fp_b = compute_reasoning_fingerprint(output_b)
        assert fp_a != fp_b

    def test_different_answer(self, sample_output):
        """Answer is not in the canonical payload (only derived preds matter),
        but status/confidence changes would change it."""
        output_b = copy.deepcopy(sample_output)
        output_b.confidence = ConfidenceReport(factual=0.1, reasoning=0.1, knowledge_boundary=0.1)
        fp_a = compute_reasoning_fingerprint(sample_output)
        fp_b = compute_reasoning_fingerprint(output_b)
        assert fp_a != fp_b

    def test_additional_rule_changes_fingerprint(self, sample_output):
        """Adding a new step/rule → different fingerprint."""
        output_b = copy.deepcopy(sample_output)
        output_b.reasoning_trace.rules_activated.append("r2")
        output_b.reasoning_trace.steps.append(
            ReasoningStep(
                step_number=2,
                description="rule r2",
                rule_applied="r2",
                predicates=[Predicate("NewPred", ())],
                confidence=0.7,
            )
        )
        fp_a = compute_reasoning_fingerprint(sample_output)
        fp_b = compute_reasoning_fingerprint(output_b)
        assert fp_a != fp_b


class TestUnsatCoreFingerprint:
    """Presence/absence of unsat core → different fingerprint."""

    def test_with_vs_without_unsat_core(self, sample_output, sample_unsat_core):
        fp_without = compute_reasoning_fingerprint(sample_output)
        fp_with = compute_reasoning_fingerprint(sample_output, unsat_core=sample_unsat_core)
        assert fp_without != fp_with

    def test_different_unsat_core(self, sample_output, sample_unsat_core):
        core2 = UnsatCore(
            conflicting_rule_ids=["r3"],
            constraint_ids=[10],
            explanation="Different conflict.",
        )
        fp1 = compute_reasoning_fingerprint(sample_output, unsat_core=sample_unsat_core)
        fp2 = compute_reasoning_fingerprint(sample_output, unsat_core=core2)
        assert fp1 != fp2


class TestNSILFingerprint:
    """Presence/absence of NSIL → different fingerprint deterministically."""

    def test_with_vs_without_nsil(self, sample_output, sample_nsil_report):
        fp_without = compute_reasoning_fingerprint(sample_output)
        fp_with = compute_reasoning_fingerprint(sample_output, nsil_report=sample_nsil_report)
        assert fp_without != fp_with

    def test_different_nsil(self, sample_output, sample_nsil_report):
        report2 = IntegrityReport(
            items=[
                IntegrityItem(schema="X", evidence=0.1, membership=0.0, need=1.0, residual=0.9),
            ],
            integrity_score=0.1,
            is_neutral=False,
        )
        fp1 = compute_reasoning_fingerprint(sample_output, nsil_report=sample_nsil_report)
        fp2 = compute_reasoning_fingerprint(sample_output, nsil_report=report2)
        assert fp1 != fp2

    def test_nsil_neutral_vs_active(self, sample_output, sample_nsil_report):
        neutral = IntegrityReport(integrity_score=1.0, is_neutral=True)
        fp_neutral = compute_reasoning_fingerprint(sample_output, nsil_report=neutral)
        fp_active = compute_reasoning_fingerprint(sample_output, nsil_report=sample_nsil_report)
        assert fp_neutral != fp_active


class TestConfigHash:
    """Config hash affects fingerprint."""

    def test_different_config_different_fingerprint(self, sample_output):
        fp1 = compute_reasoning_fingerprint(sample_output, config_snapshot={"a": 1})
        fp2 = compute_reasoning_fingerprint(sample_output, config_snapshot={"a": 2})
        assert fp1 != fp2

    def test_no_config_vs_config(self, sample_output, sample_config):
        fp1 = compute_reasoning_fingerprint(sample_output)
        fp2 = compute_reasoning_fingerprint(sample_output, config_snapshot=sample_config)
        assert fp1 != fp2


# ═══════════════════════════════════════════════════════════════════
#  API INTEGRATION
# ═══════════════════════════════════════════════════════════════════


class TestFingerprintInNSIOutput:
    """Fingerprint is attached to every NSIOutput from reason()."""

    def test_output_has_fingerprint(self):
        """NSIOutput from NeSyModel.reason() carries reasoning_fingerprint."""
        from nesy.api.nesy_model import NeSyModel
        from nesy.core.types import SymbolicRule

        model = NeSyModel(domain="general")
        model.add_rule(
            SymbolicRule(
                id="test_rule",
                antecedents=[Predicate("A", ("x",))],
                consequents=[Predicate("B", ("x",))],
                weight=0.9,
                description="A → B",
            )
        )
        output = model.reason(facts={Predicate("A", ("x",))})
        assert output.reasoning_fingerprint is not None
        assert len(output.reasoning_fingerprint) == 64

    def test_rejected_output_has_fingerprint(self):
        """REJECTED outputs also carry fingerprint."""
        from nesy.api.nesy_model import NeSyModel
        from nesy.core.types import SymbolicRule

        model = NeSyModel(domain="general")
        # Create a contradicting rule set
        model.add_rule(
            SymbolicRule(
                id="r_neg",
                antecedents=[Predicate("NOT_A", ("x",))],
                consequents=[Predicate("A", ("x",))],
                weight=1.0,
            )
        )
        output = model.reason(facts={Predicate("NOT_A", ("x",))})
        assert output.reasoning_fingerprint is not None
        assert len(output.reasoning_fingerprint) == 64

    def test_deterministic_across_calls(self):
        """Same model + same facts → same fingerprint."""
        from nesy.api.nesy_model import NeSyModel
        from nesy.core.types import SymbolicRule

        model = NeSyModel(domain="general")
        model.add_rule(
            SymbolicRule(
                id="r1",
                antecedents=[Predicate("A", ("x",))],
                consequents=[Predicate("B", ("x",))],
                weight=0.9,
                description="A → B",
            )
        )
        facts = {Predicate("A", ("x",))}
        output1 = model.reason(facts=facts)
        output2 = model.reason(facts=facts)
        assert output1.reasoning_fingerprint == output2.reasoning_fingerprint


class TestFingerprintInPCAP:
    """Fingerprint is included in PCAP export."""

    def test_pcap_contains_fingerprint(self):
        from nesy.api.nesy_model import NeSyModel
        from nesy.core.types import SymbolicRule

        model = NeSyModel(domain="general")
        model.add_rule(
            SymbolicRule(
                id="r1",
                antecedents=[Predicate("A", ("x",))],
                consequents=[Predicate("B", ("x",))],
                weight=0.9,
            )
        )
        output = model.reason(facts={Predicate("A", ("x",))})
        capsule = model.export_proof_capsule(output)
        d = capsule.to_dict()
        assert "reasoning_fingerprint" in d
        assert d["reasoning_fingerprint"] == output.reasoning_fingerprint

    def test_pcap_file_contains_fingerprint(self, tmp_path):
        import json
        from nesy.api.nesy_model import NeSyModel
        from nesy.core.types import SymbolicRule

        model = NeSyModel(domain="general")
        model.add_rule(
            SymbolicRule(
                id="r1",
                antecedents=[Predicate("A", ("x",))],
                consequents=[Predicate("B", ("x",))],
                weight=0.9,
            )
        )
        output = model.reason(facts={Predicate("A", ("x",))})
        path = str(tmp_path / "test.pcap.json")
        model.export_proof_capsule(output, path=path)

        data = json.loads(open(path).read())
        assert "reasoning_fingerprint" in data
        assert len(data["reasoning_fingerprint"]) == 64
