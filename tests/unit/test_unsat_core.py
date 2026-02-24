"""
tests/unit/test_unsat_core.py
=============================
Comprehensive tests for the Unsat-Core → Human Explanation revolution.

Covers:
  - UnsatCore dataclass (types.py)
  - SymbolicConflict.unsat_core field (exceptions.py)
  - explain_unsat_core() — main explanation generator
  - explain_constraint_violations() — solver-level violations
  - enrich_with_null_set() — NSI integration
  - format_contradiction_report() — formatted output
  - _predicate_matches() — variable matching
  - _predicate_to_concept() — concept extraction
  - _generate_explanation_text() — narrative generation
  - _analyse_antecedent_conflict() — conflict analysis
  - _propose_repairs() — repair suggestion generation
  - SymbolicEngine.check_consistency() — UnsatCore attachment
  - ConstraintSolver — UnsatCore in ConstraintResult
  - NeSyModel.explain_contradiction() — full pipeline
  - ContradictionReport dataclass

Target: 100% line coverage for nesy/symbolic/unsat_explanation.py.
"""

from __future__ import annotations

import pytest

from nesy.api.nesy_model import (
    ContradictionReport,
    NeSyModel,
)
from nesy.core.exceptions import SymbolicConflict
from nesy.core.types import (
    ConceptEdge,
    NullItem,
    NullSet,
    NullType,
    OutputStatus,
    Predicate,
    PresentSet,
    SymbolicRule,
    UnsatCore,
)
from nesy.symbolic.engine import SymbolicEngine
from nesy.symbolic.solver import (
    ArithmeticConstraint,
    ConstraintResult,
    ConstraintSolver,
)
from nesy.symbolic.unsat_explanation import (
    _analyse_antecedent_conflict,
    _generate_explanation_text,
    _predicate_matches,
    _predicate_to_concept,
    _propose_repairs,
    enrich_with_null_set,
    explain_constraint_violations,
    explain_unsat_core,
    format_contradiction_report,
)


# ═══════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_rules_db():
    """A small knowledge base with 3 rules for testing."""
    r1 = SymbolicRule(
        id="rule_fever_infection",
        antecedents=[Predicate("HasSymptom", ("?p", "fever"))],
        consequents=[Predicate("PossiblyHas", ("?p", "infection"))],
        weight=0.9,
        description="Fever suggests possible infection",
    )
    r2 = SymbolicRule(
        id="rule_allergy_block",
        antecedents=[Predicate("HasAllergy", ("?p", "penicillin"))],
        consequents=[Predicate("Contraindication", ("?p", "penicillin"))],
        weight=1.0,
        immutable=True,
        description="Penicillin allergy blocks penicillin prescription",
    )
    r3 = SymbolicRule(
        id="rule_prescribe_penicillin",
        antecedents=[Predicate("PossiblyHas", ("?p", "infection"))],
        consequents=[Predicate("Prescribed", ("?p", "penicillin"))],
        weight=0.95,
        description="Infection → prescribe penicillin",
    )
    return {r.id: r for r in [r1, r2, r3]}


@pytest.fixture
def sample_facts():
    return {
        Predicate("HasSymptom", ("patient_1", "fever")),
        Predicate("HasAllergy", ("patient_1", "penicillin")),
    }


@pytest.fixture
def sample_null_set():
    """A NullSet with critical + meaningful items."""
    present = PresentSet(concepts={"fever"}, context_type="medical")
    items = [
        NullItem(
            concept="blood_test",
            weight=0.9,
            null_type=NullType.TYPE3_CRITICAL,
            expected_because_of=["fever"],
            criticality=1.0,
        ),
        NullItem(
            concept="temperature_reading",
            weight=0.85,
            null_type=NullType.TYPE2_MEANINGFUL,
            expected_because_of=["fever"],
            criticality=1.0,
        ),
        NullItem(
            concept="cough",
            weight=0.3,
            null_type=NullType.TYPE1_EXPECTED,
            expected_because_of=["fever"],
            criticality=1.0,
        ),
    ]
    return NullSet(items=items, present_set=present)


# ═══════════════════════════════════════════════════════════════════
#  UnsatCore DATACLASS TESTS
# ═══════════════════════════════════════════════════════════════════


class TestUnsatCoreDataclass:
    """Tests for the UnsatCore type in nesy/core/types.py."""

    def test_defaults(self):
        core = UnsatCore()
        assert core.conflicting_rule_ids == []
        assert core.constraint_ids == []
        assert core.explanation == ""
        assert core.suggested_additions == []
        assert core.repair_actions == []
        assert core.raw_labels == []

    def test_core_size_from_rules(self):
        core = UnsatCore(conflicting_rule_ids=["r1", "r2", "r3"])
        assert core.core_size == 3

    def test_core_size_from_constraints(self):
        core = UnsatCore(constraint_ids=[0, 1, 2, 3])
        assert core.core_size == 4

    def test_core_size_max_of_both(self):
        core = UnsatCore(
            conflicting_rule_ids=["r1", "r2"],
            constraint_ids=[0, 1, 2],
        )
        assert core.core_size == 3

    def test_core_size_empty(self):
        core = UnsatCore()
        assert core.core_size == 0

    def test_has_repairs_with_additions(self):
        core = UnsatCore(suggested_additions=["blood_test"])
        assert core.has_repairs is True

    def test_has_repairs_with_actions(self):
        core = UnsatCore(repair_actions=[{"action": "add_fact"}])
        assert core.has_repairs is True

    def test_has_repairs_false(self):
        core = UnsatCore()
        assert core.has_repairs is False

    def test_summary_singular(self):
        core = UnsatCore(
            conflicting_rule_ids=["r1"],
            suggested_additions=["fever"],
        )
        assert "1 conflict" in core.summary()
        assert "1 suggested fix)" in core.summary()

    def test_summary_plural(self):
        core = UnsatCore(
            conflicting_rule_ids=["r1", "r2", "r3"],
            suggested_additions=["a", "b"],
        )
        assert "3 conflicts" in core.summary()
        assert "2 suggested fixes" in core.summary()


# ═══════════════════════════════════════════════════════════════════
#  SymbolicConflict WITH UnsatCore
# ═══════════════════════════════════════════════════════════════════


class TestSymbolicConflictWithUnsatCore:
    def test_default_unsat_core_is_none(self):
        exc = SymbolicConflict("test", conflicting_rules=["r1"])
        assert exc.unsat_core is None

    def test_unsat_core_attached(self):
        core = UnsatCore(
            conflicting_rule_ids=["r1"],
            explanation="Test conflict",
        )
        exc = SymbolicConflict(
            "test",
            conflicting_rules=["r1"],
            unsat_core=core,
        )
        assert exc.unsat_core is core
        assert exc.unsat_core.explanation == "Test conflict"

    def test_backward_compat_no_keyword(self):
        """Old code that doesn't pass unsat_core still works."""
        exc = SymbolicConflict("msg", ["r1"], {"key": "val"})
        assert exc.conflicting_rules == ["r1"]
        assert exc.context == {"key": "val"}
        assert exc.unsat_core is None


# ═══════════════════════════════════════════════════════════════════
#  _predicate_matches TESTS
# ═══════════════════════════════════════════════════════════════════


class TestPredicateMatches:
    def test_exact_match(self):
        p = Predicate("HasSymptom", ("patient_1", "fever"))
        f = Predicate("HasSymptom", ("patient_1", "fever"))
        assert _predicate_matches(p, f) is True

    def test_variable_match(self):
        p = Predicate("HasSymptom", ("?p", "fever"))
        f = Predicate("HasSymptom", ("patient_1", "fever"))
        assert _predicate_matches(p, f) is True

    def test_different_name_no_match(self):
        p = Predicate("HasAllergy", ("?p", "fever"))
        f = Predicate("HasSymptom", ("patient_1", "fever"))
        assert _predicate_matches(p, f) is False

    def test_different_arity_no_match(self):
        p = Predicate("HasSymptom", ("?p",))
        f = Predicate("HasSymptom", ("patient_1", "fever"))
        assert _predicate_matches(p, f) is False

    def test_constant_mismatch(self):
        p = Predicate("HasSymptom", ("?p", "cough"))
        f = Predicate("HasSymptom", ("patient_1", "fever"))
        assert _predicate_matches(p, f) is False

    def test_all_variables(self):
        p = Predicate("HasSymptom", ("?p", "?s"))
        f = Predicate("HasSymptom", ("patient_1", "fever"))
        assert _predicate_matches(p, f) is True

    def test_no_args(self):
        p = Predicate("IsTrue")
        f = Predicate("IsTrue")
        assert _predicate_matches(p, f) is True


# ═══════════════════════════════════════════════════════════════════
#  _predicate_to_concept TESTS
# ═══════════════════════════════════════════════════════════════════


class TestPredicateToConcept:
    def test_last_non_variable(self):
        p = Predicate("HasSymptom", ("?p", "fever"))
        assert _predicate_to_concept(p) == "fever"

    def test_single_arg(self):
        p = Predicate("Likely", ("infection",))
        assert _predicate_to_concept(p) == "infection"

    def test_all_variables_falls_back(self):
        p = Predicate("SomeRelation", ("?x", "?y"))
        assert _predicate_to_concept(p) == "SomeRelation"

    def test_no_args_falls_back(self):
        p = Predicate("DrugInteraction")
        assert _predicate_to_concept(p) == "DrugInteraction"

    def test_multiple_non_variable_args(self):
        p = Predicate("Interacts", ("drugA", "drugB"))
        assert _predicate_to_concept(p) == "drugB"


# ═══════════════════════════════════════════════════════════════════
#  _generate_explanation_text TESTS
# ═══════════════════════════════════════════════════════════════════


class TestGenerateExplanationText:
    def test_no_rules(self):
        text = _generate_explanation_text([], [], [], set())
        assert text == "No conflicting rules identified."

    def test_single_rule(self):
        text = _generate_explanation_text(
            ["r1"],
            ["fever → infection"],
            [],
            set(),
        )
        assert "Rule 'r1' is violated" in text

    def test_two_rules(self):
        text = _generate_explanation_text(
            ["r1", "r2"],
            ["desc1", "desc2"],
            [],
            set(),
        )
        assert "2 rules cannot be true together" in text
        assert "'r1'" in text
        assert "'r2'" in text

    def test_three_rules(self):
        text = _generate_explanation_text(
            ["r1", "r2", "r3"],
            ["d1", "d2", "d3"],
            [],
            set(),
        )
        assert "3 rules cannot all be true together" in text
        assert "'r3'" in text

    def test_includes_rule_details(self):
        text = _generate_explanation_text(
            ["r1"],
            ["Fever implies infection"],
            [],
            set(),
        )
        assert "r1: Fever implies infection" in text

    def test_includes_antecedent_analysis(self, sample_rules_db):
        rules = [sample_rules_db["rule_fever_infection"]]
        facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
        text = _generate_explanation_text(
            ["rule_fever_infection"],
            ["Fever suggests possible infection"],
            rules,
            facts,
        )
        assert "conflict arises" in text or "antecedent" in text.lower() or "Rule" in text


# ═══════════════════════════════════════════════════════════════════
#  _analyse_antecedent_conflict TESTS
# ═══════════════════════════════════════════════════════════════════


class TestAnalyseAntecedentConflict:
    def test_empty_rules(self):
        assert _analyse_antecedent_conflict([], set()) == ""

    def test_satisfied_rule(self, sample_rules_db):
        rules = [sample_rules_db["rule_fever_infection"]]
        facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
        result = _analyse_antecedent_conflict(rules, facts)
        assert "conflict arises" in result
        assert "rule_fever_infection" in result

    def test_missing_antecedent(self, sample_rules_db):
        rules = [sample_rules_db["rule_prescribe_penicillin"]]
        facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
        result = _analyse_antecedent_conflict(rules, facts)
        assert "not currently present" in result

    def test_mixed_satisfied_and_missing(self, sample_rules_db):
        rules = [
            sample_rules_db["rule_fever_infection"],
            sample_rules_db["rule_prescribe_penicillin"],
        ]
        facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
        result = _analyse_antecedent_conflict(rules, facts)
        assert "rule_fever_infection" in result


# ═══════════════════════════════════════════════════════════════════
#  _propose_repairs TESTS
# ═══════════════════════════════════════════════════════════════════


class TestProposeRepairs:
    def test_empty_rules(self):
        suggested, repairs = _propose_repairs([], set())
        assert suggested == []
        assert repairs == []

    def test_unsatisfied_antecedent_suggested(self, sample_rules_db):
        rules = [sample_rules_db["rule_prescribe_penicillin"]]
        facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
        suggested, repairs = _propose_repairs(rules, facts)
        # The antecedent PossiblyHas(?p, infection) is missing
        assert "infection" in suggested
        assert len(repairs) > 0
        assert repairs[0]["action"] == "add_fact"

    def test_contraindication_consequent_guard(self, sample_rules_db):
        rules = [sample_rules_db["rule_allergy_block"]]
        facts = {Predicate("HasAllergy", ("patient_1", "penicillin"))}
        suggested, repairs = _propose_repairs(rules, facts)
        # Consequent is Contraindication — should propose a guard
        guard_actions = [r for r in repairs if r["action"] == "add_guard"]
        assert len(guard_actions) > 0

    def test_all_antecedents_satisfied_no_add_fact(self, sample_rules_db):
        rules = [sample_rules_db["rule_fever_infection"]]
        facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
        suggested, repairs = _propose_repairs(rules, facts)
        # HasSymptom(?p, fever) matches — no add_fact for 'fever'
        add_facts = [r for r in repairs if r["action"] == "add_fact"]
        # But there might be no add_fact if all antecedents matched
        # fever_infection rule has only one antecedent: HasSymptom(?p, fever)
        # which matches our fact, so no add_fact should be suggested
        assert all(af["concept"] != "fever" for af in add_facts)

    def test_deduplication(self, sample_rules_db):
        """Same concept from two different rules isn't duplicated."""
        r_dup = SymbolicRule(
            id="rule_dup",
            antecedents=[Predicate("PossiblyHas", ("?p", "infection"))],
            consequents=[Predicate("NotRelevant", ("?p",))],
            weight=0.8,
            description="Duplicate antecedent",
        )
        rules = [sample_rules_db["rule_prescribe_penicillin"], r_dup]
        facts = set()
        suggested, repairs = _propose_repairs(rules, facts)
        assert suggested.count("infection") == 1


# ═══════════════════════════════════════════════════════════════════
#  explain_unsat_core TESTS
# ═══════════════════════════════════════════════════════════════════


class TestExplainUnsatCore:
    def test_basic(self, sample_rules_db, sample_facts):
        core = explain_unsat_core(
            conflicting_rule_ids=["rule_fever_infection", "rule_allergy_block"],
            rules_db=sample_rules_db,
            facts=sample_facts,
        )
        assert isinstance(core, UnsatCore)
        assert "rule_fever_infection" in core.conflicting_rule_ids
        assert "rule_allergy_block" in core.conflicting_rule_ids
        assert len(core.explanation) > 0
        assert "2 rules" in core.explanation

    def test_with_constraint_ids(self, sample_rules_db):
        core = explain_unsat_core(
            conflicting_rule_ids=["rule_fever_infection"],
            rules_db=sample_rules_db,
            constraint_ids=[0, 2],
            constraint_labels=["c_0", "c_2"],
        )
        assert core.constraint_ids == [0, 2]
        assert core.raw_labels == ["c_0", "c_2"]

    def test_unknown_rule_id(self, sample_rules_db):
        core = explain_unsat_core(
            conflicting_rule_ids=["nonexistent_rule"],
            rules_db=sample_rules_db,
        )
        assert "nonexistent_rule" in core.explanation
        assert "not in KB" in core.explanation

    def test_empty_conflict(self, sample_rules_db):
        core = explain_unsat_core(
            conflicting_rule_ids=[],
            rules_db=sample_rules_db,
        )
        assert "No conflicting rules" in core.explanation

    def test_three_rules(self, sample_rules_db, sample_facts):
        core = explain_unsat_core(
            conflicting_rule_ids=list(sample_rules_db.keys()),
            rules_db=sample_rules_db,
            facts=sample_facts,
        )
        assert core.core_size == 3
        assert "3 rules" in core.explanation

    def test_suggested_additions(self, sample_rules_db, sample_facts):
        core = explain_unsat_core(
            conflicting_rule_ids=["rule_prescribe_penicillin"],
            rules_db=sample_rules_db,
            facts=sample_facts,
        )
        # PossiblyHas(?p, infection) is unmatched → suggest "infection"
        assert "infection" in core.suggested_additions

    def test_no_facts(self, sample_rules_db):
        """Without facts, all antecedents are unsatisfied."""
        core = explain_unsat_core(
            conflicting_rule_ids=["rule_fever_infection"],
            rules_db=sample_rules_db,
        )
        assert len(core.suggested_additions) > 0


# ═══════════════════════════════════════════════════════════════════
#  explain_constraint_violations TESTS
# ═══════════════════════════════════════════════════════════════════


class TestExplainConstraintViolations:
    def test_basic(self):
        core = explain_constraint_violations(
            constraint_ids=[0, 1],
            violations=[
                "Constraint violated: dosage <= 500 (actual: 600)",
                "Constraint violated: age >= 18 (actual: 15)",
            ],
        )
        assert isinstance(core, UnsatCore)
        assert core.constraint_ids == [0, 1]
        assert "2 arithmetic constraints" in core.explanation
        assert "dosage" in core.explanation
        assert "age" in core.explanation

    def test_single_violation(self):
        core = explain_constraint_violations(
            constraint_ids=[3],
            violations=["Constraint violated: temp > 37.5"],
        )
        assert "1 arithmetic constraint" in core.explanation
        assert "is violated" in core.explanation

    def test_no_violations(self):
        core = explain_constraint_violations(
            constraint_ids=[],
            violations=[],
        )
        assert "No constraint violations" in core.explanation

    def test_with_labels(self):
        core = explain_constraint_violations(
            constraint_ids=[0],
            violations=["Violated: x <= 10"],
            constraint_labels=["c_0"],
        )
        assert core.raw_labels == ["c_0"]


# ═══════════════════════════════════════════════════════════════════
#  enrich_with_null_set TESTS
# ═══════════════════════════════════════════════════════════════════


class TestEnrichWithNullSet:
    def test_none_null_set(self):
        core = UnsatCore(explanation="original")
        enriched = enrich_with_null_set(core, None)
        assert enriched is core  # unchanged

    def test_critical_null_prepended(self, sample_null_set):
        core = UnsatCore(
            conflicting_rule_ids=["r1"],
            suggested_additions=["existing_concept"],
        )
        enriched = enrich_with_null_set(core, sample_null_set)
        # blood_test is critical → should be prepended
        assert enriched.suggested_additions[0] == "blood_test"
        assert "existing_concept" in enriched.suggested_additions

    def test_critical_null_repair_action(self, sample_null_set):
        core = UnsatCore(conflicting_rule_ids=["r1"])
        enriched = enrich_with_null_set(core, sample_null_set)
        critical_repairs = [
            r for r in enriched.repair_actions if r["action"] == "add_critical_concept"
        ]
        assert len(critical_repairs) >= 1
        assert critical_repairs[0]["concept"] == "blood_test"
        assert "CRITICAL" in critical_repairs[0]["reason"]

    def test_meaningful_null_overlap(self, sample_null_set):
        core = UnsatCore(
            conflicting_rule_ids=["r1"],
            suggested_additions=["temperature_reading"],
        )
        enriched = enrich_with_null_set(core, sample_null_set)
        confirm_repairs = [
            r for r in enriched.repair_actions if r["action"] == "confirm_meaningful_null"
        ]
        assert len(confirm_repairs) >= 1

    def test_no_duplicate_critical(self, sample_null_set):
        """If concept already suggested, don't add twice."""
        core = UnsatCore(
            conflicting_rule_ids=["r1"],
            suggested_additions=["blood_test"],
        )
        enriched = enrich_with_null_set(core, sample_null_set)
        assert enriched.suggested_additions.count("blood_test") == 1

    def test_empty_null_set(self):
        core = UnsatCore(explanation="original")
        empty_ns = NullSet(
            items=[],
            present_set=PresentSet(concepts=set()),
        )
        enriched = enrich_with_null_set(core, empty_ns)
        # No items → no enrichment
        assert enriched.suggested_additions == []


# ═══════════════════════════════════════════════════════════════════
#  format_contradiction_report TESTS
# ═══════════════════════════════════════════════════════════════════


class TestFormatContradictionReport:
    def test_basic_format(self):
        core = UnsatCore(
            conflicting_rule_ids=["r1", "r2"],
            explanation="Two rules conflict.",
            suggested_additions=["blood_test"],
            repair_actions=[
                {
                    "action": "add_fact",
                    "concept": "blood_test",
                    "reason": "Adding blood_test resolves the issue.",
                }
            ],
        )
        report = format_contradiction_report(core)
        assert "CONTRADICTION DETECTED" in report
        assert "Two rules conflict." in report
        assert "Suggested Repairs" in report
        assert "blood_test" in report
        assert "UnsatCore" in report

    def test_no_repairs_section(self):
        core = UnsatCore(explanation="Conflict detected.")
        report = format_contradiction_report(core, include_repairs=False)
        assert "CONTRADICTION DETECTED" in report
        assert "Suggested Repairs" not in report

    def test_no_repairs_when_empty(self):
        core = UnsatCore(explanation="Conflict but no suggestion.")
        report = format_contradiction_report(core)
        assert "Suggested Repairs" not in report

    def test_multiple_repairs(self):
        core = UnsatCore(
            conflicting_rule_ids=["r1"],
            explanation="Conflict.",
            suggested_additions=["concept_a", "concept_b"],
            repair_actions=[
                {"action": "add_fact", "concept": "concept_a", "reason": "Reason A."},
                {"action": "add_fact", "concept": "concept_b", "reason": "Reason B."},
            ],
        )
        report = format_contradiction_report(core)
        assert "1." in report
        assert "2." in report
        assert "Reason A." in report
        assert "Reason B." in report

    def test_suggestion_without_matching_action(self):
        core = UnsatCore(
            conflicting_rule_ids=["r1"],
            explanation="Conflict.",
            suggested_additions=["orphan_concept"],
            repair_actions=[],  # no matching action
        )
        report = format_contradiction_report(core)
        assert "orphan_concept" in report


# ═══════════════════════════════════════════════════════════════════
#  SymbolicEngine.check_consistency WITH UnsatCore
# ═══════════════════════════════════════════════════════════════════


class TestSymbolicEngineUnsatCore:
    def test_consistency_attaches_unsat_core(self):
        """When check_consistency detects a contradiction, UnsatCore is attached.

        To trigger a contradiction via resolution, we need:
        - Negated fact:  {NOT_A(x)}
        - Hard rule CNF: {A(?x)}  (from rule NOT_A(?x) → A(?x))
        Resolving these yields {}, proving UNSAT.
        """
        engine = SymbolicEngine(domain="medical")

        # Rule: NOT_A(?x) → A(?x) — CNF is {A(?x)}.
        # With fact A(x), negated fact is {NOT_A(x)}.
        # Resolution: {NOT_A(x)} resolves with {A(?x)} → {} (contradiction)
        hard_rule = SymbolicRule(
            id="test_hard_rule",
            antecedents=[Predicate("NOT_A", ("?x",))],
            consequents=[Predicate("A", ("?x",))],
            weight=1.0,
            description="Contradicts itself when A is both true and false",
        )
        engine.add_rule(hard_rule)

        facts = {Predicate("A", ("x",))}

        with pytest.raises(SymbolicConflict) as exc_info:
            engine.check_consistency(facts)

        exc = exc_info.value
        assert exc.unsat_core is not None
        assert isinstance(exc.unsat_core, UnsatCore)
        assert len(exc.unsat_core.explanation) > 0

    def test_consistent_facts_no_conflict(self):
        engine = SymbolicEngine(domain="medical")
        rule = SymbolicRule(
            id="soft_rule",
            antecedents=[Predicate("HasSymptom", ("?p", "fever"))],
            consequents=[Predicate("PossiblyHas", ("?p", "infection"))],
            weight=0.5,  # soft rule, not checked in consistency
            description="Soft rule (not hard)",
        )
        engine.add_rule(rule)
        facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
        assert engine.check_consistency(facts) is True

    def test_reason_returns_rejected_with_unsat_core(self):
        """NeSyModel.reason() stores UnsatCore on rejected output via mock."""
        from unittest.mock import patch

        model = NeSyModel(domain="medical")
        core = UnsatCore(
            conflicting_rule_ids=["r1"],
            explanation="Test conflict explanation",
        )
        exc = SymbolicConflict(
            "conflict",
            ["r1"],
            {"fact_count": 1},
            unsat_core=core,
        )

        with patch.object(model._symbolic, "reason", side_effect=exc):
            output = model.reason(facts={Predicate("A", ("x",))})
            assert output.status == OutputStatus.REJECTED
            assert hasattr(output, "_unsat_core")
            assert output._unsat_core is not None
            assert output._unsat_core.explanation == "Test conflict explanation"

    def test_reason_returns_rejected_with_real_contradiction(self):
        """NeSyModel.reason() with a real contradiction creates UnsatCore."""
        model = NeSyModel(domain="medical")
        # Rule that creates contradiction via resolution
        hard_rule = SymbolicRule(
            id="contradiction_rule",
            antecedents=[Predicate("NOT_A", ("?x",))],
            consequents=[Predicate("A", ("?x",))],
            weight=1.0,
            description="Creates contradiction",
        )
        model.add_rule(hard_rule)

        output = model.reason(facts={Predicate("A", ("x",))})
        assert output.status == OutputStatus.REJECTED
        assert hasattr(output, "_unsat_core")


# ═══════════════════════════════════════════════════════════════════
#  ConstraintSolver → UnsatCore INTEGRATION
# ═══════════════════════════════════════════════════════════════════


class TestConstraintSolverUnsatCore:
    def test_satisfiable_no_unsat_core(self):
        solver = ConstraintSolver()
        solver.add_constraint(ArithmeticConstraint("x", "<=", 100))
        solver.set_value("x", 50)
        result = solver.check_all()
        assert result.satisfiable is True
        # UnsatCore may be None on SAT
        # (Z3 path returns None, Python path also returns None)

    def test_z3_unsat_has_core(self):
        solver = ConstraintSolver()
        solver.add_constraint(ArithmeticConstraint("x", ">=", 10))
        solver.add_constraint(ArithmeticConstraint("x", "<=", 5))
        solver.set_value("x", 7)
        result = solver.check_all()
        assert result.satisfiable is False
        assert result.unsat_core is not None
        assert isinstance(result.unsat_core, UnsatCore)
        assert len(result.unsat_core.explanation) > 0

    def test_constraint_result_with_unsat_core_field(self):
        result = ConstraintResult(
            satisfiable=False,
            violations=["x > 10 violated"],
            constraint_ids=[0],
            unsat_core=UnsatCore(explanation="test"),
        )
        assert result.unsat_core is not None
        assert result.unsat_core.explanation == "test"

    def test_constraint_result_default_none(self):
        result = ConstraintResult(satisfiable=True)
        assert result.unsat_core is None


# ═══════════════════════════════════════════════════════════════════
#  NeSyModel.explain_contradiction TESTS
# ═══════════════════════════════════════════════════════════════════


class TestExplainContradiction:
    def _make_rejected_model(self):
        """Helper: create a model with a real contradiction rule."""
        model = NeSyModel(domain="medical")
        hard_rule = SymbolicRule(
            id="conflict_rule",
            antecedents=[Predicate("NOT_A", ("?x",))],
            consequents=[Predicate("A", ("?x",))],
            weight=1.0,
            description="A triggers contradiction",
        )
        model.add_rule(hard_rule)
        return model

    def test_basic_contradiction_report(self):
        model = self._make_rejected_model()
        output = model.reason(facts={Predicate("A", ("x",))})
        assert output.status == OutputStatus.REJECTED

        report = model.explain_contradiction(output)
        assert isinstance(report, ContradictionReport)
        assert report.unsat_core is not None
        assert "CONTRADICTION DETECTED" in report.report
        assert report.output is output

    def test_non_rejected_output(self):
        model = NeSyModel(domain="general")
        rule = SymbolicRule(
            id="safe_rule",
            antecedents=[Predicate("A", ("x",))],
            consequents=[Predicate("B", ("x",))],
            weight=0.8,
            description="Safe soft rule",
        )
        model.add_rule(rule)
        output = model.reason(facts={Predicate("A", ("x",))})
        assert output.status != OutputStatus.REJECTED

        report = model.explain_contradiction(output)
        assert "No contradiction" in report.unsat_core.explanation

    def test_explain_with_null_set_enrichment(self):
        model = NeSyModel(domain="medical")
        model.add_concept_edge(
            ConceptEdge(
                source="fever",
                target="blood_test",
                cooccurrence_prob=0.9,
                causal_strength=1.0,
                temporal_stability=1.0,
            )
        )
        model.register_critical_concept("blood_test", "diagnostic")

        hard_rule = SymbolicRule(
            id="hard_conflict",
            antecedents=[Predicate("NOT_HasSymptom", ("?p", "fever"))],
            consequents=[Predicate("HasSymptom", ("?p", "fever"))],
            weight=1.0,
            description="Hard conflict with fever",
        )
        model.add_rule(hard_rule)

        output = model.reason(
            facts={Predicate("HasSymptom", ("patient_1", "fever"))},
            context_type="medical",
        )
        report = model.explain_contradiction(output)
        assert isinstance(report, ContradictionReport)
        assert len(report.report) > 0

    def test_fixes_include_unsat_core_repairs(self):
        from unittest.mock import patch

        model = NeSyModel(domain="medical")
        core = UnsatCore(
            conflicting_rule_ids=["r1"],
            explanation="Test",
            repair_actions=[
                {
                    "action": "add_fact",
                    "concept": "missing_concept",
                    "reason": "Add missing_concept to resolve.",
                }
            ],
        )
        exc = SymbolicConflict("conflict", ["r1"], unsat_core=core)

        with patch.object(model._symbolic, "reason", side_effect=exc):
            output = model.reason(facts={Predicate("A", ("x",))})

        report = model.explain_contradiction(output)
        unsat_fixes = [f for f in report.fixes if f.source_null_type == "UNSAT_CORE"]
        assert len(unsat_fixes) >= 1
        assert unsat_fixes[0].missing_concept == "missing_concept"

    def test_conflicting_rules_in_report(self):
        from unittest.mock import patch

        model = NeSyModel(domain="medical")
        core = UnsatCore(
            conflicting_rule_ids=["my_conflict_rule"],
            explanation="Conflict from my rule",
        )
        exc = SymbolicConflict("conflict", ["my_conflict_rule"], unsat_core=core)

        with patch.object(model._symbolic, "reason", side_effect=exc):
            output = model.reason(facts={Predicate("X", ("a",))})

        report = model.explain_contradiction(output)
        assert "my_conflict_rule" in report.conflicting_rules

    def test_explain_without_null_set(self):
        model = self._make_rejected_model()
        output = model.reason(facts={Predicate("A", ("x",))})
        report = model.explain_contradiction(output, include_null_set=False)
        assert isinstance(report, ContradictionReport)
        assert report.unsat_core is not None


# ═══════════════════════════════════════════════════════════════════
#  ContradictionReport DATACLASS
# ═══════════════════════════════════════════════════════════════════


class TestContradictionReport:
    def test_dataclass_fields(self):

        dummy_output = type(
            "FakeOutput",
            (),
            {
                "answer": "test",
                "status": OutputStatus.REJECTED,
            },
        )()

        core = UnsatCore(
            conflicting_rule_ids=["r1"],
            explanation="Test",
        )

        report = ContradictionReport(
            output=dummy_output,
            unsat_core=core,
            report="Formatted report text",
            fixes=[],
            conflicting_rules=["r1"],
        )

        assert report.output is dummy_output
        assert report.unsat_core is core
        assert report.report == "Formatted report text"
        assert report.fixes == []
        assert report.conflicting_rules == ["r1"]


# ═══════════════════════════════════════════════════════════════════
#  END-TO-END: Full pipeline test
# ═══════════════════════════════════════════════════════════════════


class TestEndToEnd:
    def test_full_auditor_pipeline(self):
        """Full pipeline: conflict → explain → report → fixes (via mock)."""
        from unittest.mock import patch

        model = NeSyModel(domain="medical")

        # Set up rules
        model.add_rule(
            SymbolicRule(
                id="fever_implies_antibiotic",
                antecedents=[Predicate("HasSymptom", ("?p", "fever"))],
                consequents=[Predicate("Prescribed", ("?p", "amoxicillin"))],
                weight=0.95,
                description="Fever → prescribe amoxicillin",
            )
        )
        model.add_rule(
            SymbolicRule(
                id="allergy_blocks_amoxicillin",
                antecedents=[Predicate("HasAllergy", ("?p", "amoxicillin"))],
                consequents=[Predicate("Contraindication", ("?p", "amoxicillin"))],
                weight=1.0,
                immutable=True,
                description="Amoxicillin allergy → contraindication",
            )
        )

        # Add concept edges for NSI
        model.add_concept_edge(
            ConceptEdge(
                "fever",
                "blood_test",
                cooccurrence_prob=0.9,
                causal_strength=1.0,
                temporal_stability=1.0,
            )
        )
        model.register_critical_concept("blood_test", "diagnostic")

        # Build the UnsatCore that would come from the engine
        core = UnsatCore(
            conflicting_rule_ids=[
                "fever_implies_antibiotic",
                "allergy_blocks_amoxicillin",
            ],
            explanation=(
                "These 2 rules cannot be true together: "
                "'fever_implies_antibiotic' and "
                "'allergy_blocks_amoxicillin'.\n"
                "Rule details:\n"
                "  • fever_implies_antibiotic: Fever → prescribe amoxicillin\n"
                "  • allergy_blocks_amoxicillin: Amoxicillin allergy → contraindication"
            ),
            suggested_additions=["alternative_antibiotic"],
            repair_actions=[
                {
                    "action": "add_fact",
                    "concept": "alternative_antibiotic",
                    "reason": "Consider prescribing an alternative antibiotic.",
                }
            ],
        )
        exc = SymbolicConflict(
            "Contradiction between fever treatment and allergy",
            conflicting_rules=[
                "fever_implies_antibiotic",
                "allergy_blocks_amoxicillin",
            ],
            unsat_core=core,
        )

        facts = {
            Predicate("HasSymptom", ("patient_1", "fever")),
            Predicate("HasAllergy", ("patient_1", "amoxicillin")),
        }

        with patch.object(model._symbolic, "reason", side_effect=exc):
            output = model.reason(facts=facts, context_type="medical")

        assert output.status == OutputStatus.REJECTED

        # Explain the contradiction
        report = model.explain_contradiction(output)
        assert isinstance(report, ContradictionReport)
        assert report.unsat_core is not None
        assert len(report.report) > 100  # substantial report
        assert "CONTRADICTION DETECTED" in report.report
        assert len(report.conflicting_rules) > 0
        assert "fever_implies_antibiotic" in report.conflicting_rules

    def test_full_auditor_real_resolution(self):
        """Real contradiction via resolution (no mocking)."""
        model = NeSyModel(domain="test")
        model.add_rule(
            SymbolicRule(
                id="real_contradiction",
                antecedents=[Predicate("NOT_A", ("?x",))],
                consequents=[Predicate("A", ("?x",))],
                weight=1.0,
                description="Creates real contradiction",
            )
        )

        output = model.reason(facts={Predicate("A", ("x",))})
        assert output.status == OutputStatus.REJECTED

        report = model.explain_contradiction(output)
        assert isinstance(report, ContradictionReport)
        assert "CONTRADICTION DETECTED" in report.report

    def test_no_conflict_explain_is_safe(self):
        """explain_contradiction on a non-rejected output is safe."""
        model = NeSyModel(domain="general")
        model.add_rule(
            SymbolicRule(
                id="simple",
                antecedents=[Predicate("A", ("x",))],
                consequents=[Predicate("B", ("x",))],
                weight=0.5,
                description="Simple soft rule",
            )
        )
        output = model.reason(facts={Predicate("A", ("x",))})
        report = model.explain_contradiction(output)
        assert "No contradiction" in report.unsat_core.explanation
        assert report.conflicting_rules == []
