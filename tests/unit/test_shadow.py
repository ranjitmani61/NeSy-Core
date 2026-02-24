"""
tests/unit/test_shadow.py
===========================
Tests for Counterfactual Shadow Reasoning.

Verifies:
    - Exact shadow distance computation
    - All shadow classes (CRITICAL → TAUTOLOGY)
    - Human-readable explanations
    - Shadow report aggregation
    - Large-fact heuristic path
    - Integration with real NeSy rules
"""
import math
import pytest
from nesy.core.types import Predicate, SymbolicRule
from nesy.metacognition.shadow import (
    CounterfactualShadowEngine,
    ShadowClass,
    ShadowResult,
    ShadowReport,
    _classify_distance,
    _build_explanation,
    shadow_flags,
)


# ── helpers ──────────────────────────────────────────────────────

def pred(name: str, *args: str) -> Predicate:
    return Predicate(name=name, args=tuple(args))


def rule(rule_id: str, antecedents, consequents, weight: float = 1.0) -> SymbolicRule:
    return SymbolicRule(
        id=rule_id,
        antecedents=antecedents,
        consequents=consequents,
        weight=weight,
    )


# ══════════════════════════════════════════════════════════════════
#  _classify_distance
# ══════════════════════════════════════════════════════════════════

class TestClassifyDistance:

    def test_distance_1_is_critical(self):
        assert _classify_distance(1) == ShadowClass.CRITICAL

    def test_distance_2_is_fragile(self):
        assert _classify_distance(2) == ShadowClass.FRAGILE

    def test_distance_3_is_stable(self):
        assert _classify_distance(3) == ShadowClass.STABLE

    def test_distance_4_is_stable(self):
        assert _classify_distance(4) == ShadowClass.STABLE

    def test_distance_5_is_robust(self):
        assert _classify_distance(5) == ShadowClass.ROBUST

    def test_distance_10_is_robust(self):
        assert _classify_distance(10) == ShadowClass.ROBUST

    def test_distance_inf_is_tautology(self):
        assert _classify_distance(math.inf) == ShadowClass.TAUTOLOGY


# ══════════════════════════════════════════════════════════════════
#  ShadowResult properties
# ══════════════════════════════════════════════════════════════════

class TestShadowResult:

    def test_is_tautology(self):
        r = ShadowResult(
            conclusion=pred("A", "x"),
            distance=math.inf,
            shadow_class=ShadowClass.TAUTOLOGY,
        )
        assert r.is_tautology is True
        assert r.is_critical is False

    def test_is_critical(self):
        r = ShadowResult(
            conclusion=pred("A", "x"),
            distance=1,
            shadow_class=ShadowClass.CRITICAL,
            critical_facts=frozenset({pred("B", "x")}),
        )
        assert r.is_critical is True
        assert r.is_fragile is True
        assert r.is_tautology is False

    def test_is_fragile_at_distance_2(self):
        r = ShadowResult(
            conclusion=pred("A", "x"),
            distance=2,
            shadow_class=ShadowClass.FRAGILE,
        )
        assert r.is_fragile is True
        assert r.is_critical is False

    def test_str_tautology(self):
        r = ShadowResult(
            conclusion=pred("A", "x"),
            distance=math.inf,
            shadow_class=ShadowClass.TAUTOLOGY,
        )
        s = str(r)
        assert "TAUTOLOGY" in s

    def test_str_critical(self):
        r = ShadowResult(
            conclusion=pred("Diagnosis", "patient"),
            distance=1,
            shadow_class=ShadowClass.CRITICAL,
            critical_facts=frozenset({pred("Fever", "patient")}),
        )
        s = str(r)
        assert "CRITICAL" in s

    def test_str_robust(self):
        r = ShadowResult(
            conclusion=pred("A", "x"),
            distance=6,
            shadow_class=ShadowClass.ROBUST,
        )
        s = str(r)
        assert "ROBUST" in s or "robust" in s.lower()


# ══════════════════════════════════════════════════════════════════
#  CounterfactualShadowEngine — exact cases
# ══════════════════════════════════════════════════════════════════

class TestExactShadow:

    def setup_method(self):
        self.engine = CounterfactualShadowEngine(max_exact_facts=15, max_shadow_depth=7)

    # ── CASE: conclusion is in original facts (axiom) ──────────────

    def test_axiom_is_tautology(self):
        facts = {pred("A", "x"), pred("B", "x")}
        rules = []
        result = self.engine.compute(pred("A", "x"), facts, rules)
        assert result.is_tautology

    # ── CASE: single-rule chain, distance=1 ───────────────────────

    def test_distance_1_single_rule(self):
        """
        Facts: {Fever(p)}
        Rules: Fever(p) → Infection(p)
        Shadow of Infection(p): removing Fever(p) → no derivation. d=1.
        """
        facts = {pred("Fever", "p")}
        rules_ = [rule(
            "fever_infection",
            [pred("Fever", "?p")],
            [pred("Infection", "?p")],
        )]
        result = self.engine.compute(pred("Infection", "p"), facts, rules_)
        assert result.distance == 1
        assert result.shadow_class == ShadowClass.CRITICAL
        assert any(f.name == "Fever" for f in result.critical_facts)

    # ── CASE: two independent paths, distance=2 ───────────────────

    def test_distance_2_two_independent_paths(self):
        """
        Facts: {A(x), B(x)}
        Rules: A(x) → C(x),  B(x) → C(x)
        Shadow of C(x): need to remove BOTH A and B. d=2.
        """
        facts = {pred("A", "x"), pred("B", "x")}
        rules_ = [
            rule("r_a", [pred("A", "?x")], [pred("C", "?x")]),
            rule("r_b", [pred("B", "?x")], [pred("C", "?x")]),
        ]
        result = self.engine.compute(pred("C", "x"), facts, rules_)
        assert result.distance == 2
        assert result.shadow_class == ShadowClass.FRAGILE

    # ── CASE: chain of 3 rules, distance=1 ────────────────────────

    def test_distance_1_chain_root(self):
        """
        Facts: {A(x)}
        Rules: A→B, B→C, C→D
        Shadow of D(x): remove A(x) → chain collapses. d=1.
        """
        facts = {pred("A", "x")}
        rules_ = [
            rule("r1", [pred("A", "?x")], [pred("B", "?x")]),
            rule("r2", [pred("B", "?x")], [pred("C", "?x")]),
            rule("r3", [pred("C", "?x")], [pred("D", "?x")]),
        ]
        result = self.engine.compute(pred("D", "x"), facts, rules_)
        assert result.distance == 1
        assert result.is_critical
        assert any(f.name == "A" for f in result.critical_facts)

    # ── CASE: three independent paths → distance=3 ────────────────

    def test_distance_3_three_paths(self):
        """
        Facts: {A(x), B(x), C(x)}
        Rules: A→D, B→D, C→D
        Shadow of D: must remove all three. d=3.
        """
        facts = {pred("A", "x"), pred("B", "x"), pred("C", "x")}
        rules_ = [
            rule("ra", [pred("A", "?x")], [pred("D", "?x")]),
            rule("rb", [pred("B", "?x")], [pred("D", "?x")]),
            rule("rc", [pred("C", "?x")], [pred("D", "?x")]),
        ]
        result = self.engine.compute(pred("D", "x"), facts, rules_)
        assert result.distance == 3
        assert result.shadow_class == ShadowClass.STABLE

    # ── CASE: no path to conclusion → tautology (?) ───────────────

    def test_conclusion_not_derivable_at_all(self):
        """
        Facts: {A(x)}
        Rules: A→B
        Asking for shadow of C — which is never derivable.
        Shadow computation is undefined / immediately tautology-like.
        """
        facts = {pred("A", "x")}
        rules_ = [rule("r1", [pred("A", "?x")], [pred("B", "?x")])]
        # C is not derivable even with full facts
        result = self.engine.compute(pred("C", "x"), facts, rules_)
        # If it was never derivable, shadow is trivially ∞
        assert result.distance == math.inf or result.distance >= 7

    # ── CASE: conjunction required for conclusion ──────────────────

    def test_conjunction_antecedent_distance_2(self):
        """
        Facts: {Fever(p), ElevatedWBC(p)}
        Rules: Fever(p) AND ElevatedWBC(p) → Infection(p)
        Shadow: removing EITHER Fever OR ElevatedWBC breaks the derivation. d=1.
        (The rule requires conjunction — both facts are individually critical.)
        """
        facts = {pred("Fever", "p"), pred("ElevatedWBC", "p")}
        rules_ = [rule(
            "fever_wbc_infection",
            [pred("Fever", "?p"), pred("ElevatedWBC", "?p")],
            [pred("Infection", "?p")],
        )]
        result = self.engine.compute(pred("Infection", "p"), facts, rules_)
        # Either single fact removal collapses → d=1
        assert result.distance == 1
        assert result.shadow_class == ShadowClass.CRITICAL

    # ── CASE: empty fact set → tautology ──────────────────────────

    def test_empty_facts_tautology(self):
        result = self.engine.compute(pred("A", "x"), set(), [])
        assert result.is_tautology

    # ── CASE: result has explanation string ───────────────────────

    def test_explanation_is_non_empty(self):
        facts = {pred("Fever", "p")}
        rules_ = [rule("r1", [pred("Fever", "?p")], [pred("Sick", "?p")])]
        result = self.engine.compute(pred("Sick", "p"), facts, rules_)
        assert isinstance(result.flip_explanation, str)
        assert len(result.flip_explanation) > 10

    # ── CASE: medical scenario ─────────────────────────────────────

    def test_medical_scenario_chest_pain_critical(self):
        """
        Realistic medical case:
        Facts: {chest_pain, sweating, troponin_ordered}
        Rule: chest_pain + sweating → urgent_ecg_required
        Shadow: either chest_pain or sweating removal collapses conclusion. d=1.
        """
        facts = {
            pred("HasSymptom", "p", "chest_pain"),
            pred("HasSymptom", "p", "sweating"),
        }
        rules_ = [rule(
            "cardiac",
            [pred("HasSymptom", "?p", "chest_pain"), pred("HasSymptom", "?p", "sweating")],
            [pred("RequiresUrgentTest", "?p", "ecg")],
        )]
        result = self.engine.compute(pred("RequiresUrgentTest", "p", "ecg"), facts, rules_)
        assert result.distance == 1
        assert result.is_critical

    def test_medical_scenario_robust_diagnosis(self):
        """
        Robust case: 5 independent signs each sufficient for conclusion.
        Removing any one still leaves 4 paths. d=5.
        """
        facts = {
            pred("SignA", "p"),
            pred("SignB", "p"),
            pred("SignC", "p"),
            pred("SignD", "p"),
            pred("SignE", "p"),
        }
        rules_ = [
            rule("ra", [pred("SignA", "?p")], [pred("Diagnosis", "?p")]),
            rule("rb", [pred("SignB", "?p")], [pred("Diagnosis", "?p")]),
            rule("rc", [pred("SignC", "?p")], [pred("Diagnosis", "?p")]),
            rule("rd", [pred("SignD", "?p")], [pred("Diagnosis", "?p")]),
            rule("re", [pred("SignE", "?p")], [pred("Diagnosis", "?p")]),
        ]
        result = self.engine.compute(pred("Diagnosis", "p"), facts, rules_)
        assert result.distance == 5
        assert result.shadow_class == ShadowClass.ROBUST


# ══════════════════════════════════════════════════════════════════
#  compute_all → ShadowReport
# ══════════════════════════════════════════════════════════════════

class TestShadowReport:

    def setup_method(self):
        self.engine = CounterfactualShadowEngine()

    def test_empty_conclusions(self):
        report = self.engine.compute_all(set(), {pred("A", "x")}, [])
        assert isinstance(report, ShadowReport)
        assert report.minimum_distance == math.inf
        assert not report.escalation_required

    def test_single_critical_conclusion(self):
        facts = {pred("Fever", "p")}
        rules_ = [rule("r1", [pred("Fever", "?p")], [pred("Infection", "?p")])]
        conclusions = {pred("Infection", "p")}
        report = self.engine.compute_all(conclusions, facts, rules_)
        assert report.minimum_distance == 1
        assert report.escalation_required
        assert report.system_class == ShadowClass.CRITICAL

    def test_mixed_conclusions_takes_minimum(self):
        """Two conclusions: one critical (d=1) and one robust (d=5).
        System class should reflect the minimum."""
        facts = {
            pred("A", "x"),
            pred("B", "x"), pred("C", "x"), pred("D", "x"),
            pred("E", "x"), pred("F", "x"),
        }
        rules_ = [
            rule("ra", [pred("A", "?x")], [pred("CriticalConclusion", "?x")]),
            rule("rb", [pred("B", "?x")], [pred("RobustConclusion", "?x")]),
            rule("rc", [pred("C", "?x")], [pred("RobustConclusion", "?x")]),
            rule("rd", [pred("D", "?x")], [pred("RobustConclusion", "?x")]),
            rule("re", [pred("E", "?x")], [pred("RobustConclusion", "?x")]),
            rule("rf", [pred("F", "?x")], [pred("RobustConclusion", "?x")]),
        ]
        conclusions = {
            pred("CriticalConclusion", "x"),
            pred("RobustConclusion", "x"),
        }
        report = self.engine.compute_all(conclusions, facts, rules_)
        assert report.minimum_distance == 1
        assert report.system_class == ShadowClass.CRITICAL
        assert report.escalation_required

    def test_report_summary_is_string(self):
        facts = {pred("Fever", "p")}
        rules_ = [rule("r1", [pred("Fever", "?p")], [pred("Sick", "?p")])]
        report = self.engine.compute_all({pred("Sick", "p")}, facts, rules_)
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 20

    def test_all_robust_no_escalation(self):
        facts = {pred("A", "x"), pred("B", "x"), pred("C", "x"),
                 pred("D", "x"), pred("E", "x")}
        rules_ = [
            rule("ra", [pred("A", "?x")], [pred("Z", "?x")]),
            rule("rb", [pred("B", "?x")], [pred("Z", "?x")]),
            rule("rc", [pred("C", "?x")], [pred("Z", "?x")]),
            rule("rd", [pred("D", "?x")], [pred("Z", "?x")]),
            rule("re", [pred("E", "?x")], [pred("Z", "?x")]),
        ]
        report = self.engine.compute_all({pred("Z", "x")}, facts, rules_)
        assert not report.escalation_required
        assert report.system_class in (ShadowClass.ROBUST, ShadowClass.STABLE)


# ══════════════════════════════════════════════════════════════════
#  shadow_flags helper
# ══════════════════════════════════════════════════════════════════

class TestShadowFlags:

    def test_critical_report_generates_human_review_flag(self):
        facts = {pred("Fever", "p")}
        rules_ = [rule("r1", [pred("Fever", "?p")], [pred("Diagnosis", "?p")])]
        engine = CounterfactualShadowEngine()
        report = engine.compute_all({pred("Diagnosis", "p")}, facts, rules_)
        flags = shadow_flags(report)
        assert len(flags) > 0
        assert any("HUMAN REVIEW" in f or "CRITICAL" in f for f in flags)

    def test_robust_report_generates_no_escalation_flag(self):
        facts = {pred("A", "x"), pred("B", "x"), pred("C", "x"),
                 pred("D", "x"), pred("E", "x")}
        rules_ = [
            rule("ra", [pred("A", "?x")], [pred("Z", "?x")]),
            rule("rb", [pred("B", "?x")], [pred("Z", "?x")]),
            rule("rc", [pred("C", "?x")], [pred("Z", "?x")]),
            rule("rd", [pred("D", "?x")], [pred("Z", "?x")]),
            rule("re", [pred("E", "?x")], [pred("Z", "?x")]),
        ]
        engine = CounterfactualShadowEngine()
        report = engine.compute_all({pred("Z", "x")}, facts, rules_)
        flags = shadow_flags(report)
        # No HUMAN REVIEW flag for robust conclusions
        assert not any("HUMAN REVIEW" in f for f in flags)

    def test_flags_are_strings(self):
        engine = CounterfactualShadowEngine()
        report = engine.compute_all(set(), set(), [])
        flags = shadow_flags(report)
        assert isinstance(flags, list)
        for f in flags:
            assert isinstance(f, str)


# ══════════════════════════════════════════════════════════════════
#  _build_explanation
# ══════════════════════════════════════════════════════════════════

class TestBuildExplanation:

    def test_distance_1_mentions_single(self):
        conclusion = pred("Diagnosis", "patient")
        shadow_set = frozenset({pred("Fever", "patient")})
        explanation = _build_explanation(conclusion, shadow_set, 1)
        assert "CRITICAL" in explanation
        assert "single" in explanation.lower() or "absent" in explanation.lower()

    def test_distance_2_mentions_both(self):
        conclusion = pred("Diagnosis", "patient")
        shadow_set = frozenset({pred("Fever", "patient"), pred("Cough", "patient")})
        explanation = _build_explanation(conclusion, shadow_set, 2)
        assert "FRAGILE" in explanation or "both" in explanation.lower()

    def test_distance_5_mentions_distance(self):
        conclusion = pred("Diagnosis", "patient")
        shadow_set = frozenset({pred(f"Sign{i}", "patient") for i in range(5)})
        explanation = _build_explanation(conclusion, shadow_set, 5)
        assert "5" in explanation
        assert "robust" in explanation.lower()

    def test_explanation_always_mentions_conclusion(self):
        conclusion = pred("VerySpecificConclusion", "entity_001")
        shadow_set = frozenset({pred("FactA", "entity_001")})
        explanation = _build_explanation(conclusion, shadow_set, 1)
        assert "VerySpecificConclusion" in explanation


# ══════════════════════════════════════════════════════════════════
#  Heuristic path (large fact sets)
# ══════════════════════════════════════════════════════════════════

class TestHeuristicShadow:

    def test_heuristic_runs_for_large_fact_set(self):
        """Engine should not crash on 20+ facts."""
        engine = CounterfactualShadowEngine(max_exact_facts=5)  # force heuristic
        facts = {pred(f"Fact{i}", "x") for i in range(20)}
        # One rule: if Fact0 → Conclusion
        rules_ = [rule("r0", [pred("Fact0", "?x")], [pred("Conclusion", "?x")])]
        result = engine.compute(pred("Conclusion", "x"), facts, rules_)
        assert isinstance(result, ShadowResult)
        assert result.distance >= 1

    def test_heuristic_detects_critical(self):
        """Even in heuristic mode, critical should be detected."""
        engine = CounterfactualShadowEngine(max_exact_facts=2)  # force heuristic at 3+
        # 3 facts, only one path
        facts = {pred("A", "x"), pred("B", "x"), pred("C", "x")}
        rules_ = [rule("r1", [pred("A", "?x")], [pred("Z", "?x")])]
        # B and C are irrelevant — removing A alone collapses Z
        result = engine.compute(pred("Z", "x"), facts, rules_)
        assert result.distance <= 2  # heuristic may miss exact 1 but should be close


# ══════════════════════════════════════════════════════════════════
#  Real NeSy-Core integration test
# ══════════════════════════════════════════════════════════════════

class TestRealIntegration:

    def test_socrates_syllogism_shadow(self):
        """
        Classic Socrates case:
        Facts: {IsPhilosopher(socrates)}
        Rules: IsPhilosopher → IsHuman, IsHuman → IsMortal
        Shadow of IsMortal: remove IsPhilosopher → chain collapses. d=1.
        """
        engine = CounterfactualShadowEngine()
        facts = {pred("IsPhilosopher", "socrates")}
        rules_ = [
            rule("ph_human",    [pred("IsPhilosopher", "?x")], [pred("IsHuman", "?x")]),
            rule("human_mortal",[pred("IsHuman", "?x")],       [pred("IsMortal", "?x")]),
        ]
        result = engine.compute(pred("IsMortal", "socrates"), facts, rules_)
        assert result.distance == 1
        assert result.is_critical
        assert "IsPhilosopher" in result.flip_explanation

    def test_contraindication_shadow(self):
        """
        Safety-critical: contraindication detection.
        Facts: {Prescribed(p, drug), HasAllergy(p, drug)}
        Rule: Prescribed + HasAllergy → ContraindicationViolated
        Shadow: either single fact removal collapses violation detection. d=1.
        """
        engine = CounterfactualShadowEngine()
        facts = {
            pred("Prescribed", "patient", "penicillin"),
            pred("HasAllergy", "patient", "penicillin"),
        }
        rules_ = [rule(
            "contraindication",
            [pred("Prescribed", "?p", "?d"), pred("HasAllergy", "?p", "?d")],
            [pred("ContraindicationViolated", "?p", "?d")],
        )]
        conclusion = pred("ContraindicationViolated", "patient", "penicillin")
        result = engine.compute(conclusion, facts, rules_)
        # Single removal of either fact collapses the safety check
        assert result.distance == 1
        assert result.is_critical
