"""
tests/unit/test_shadow_integration.py
======================================
Tests for Shadow Reasoning integration into MetaCognitionMonitor.

Covers:
    1. Medical strict + shadow_policy="flag" → FLAGGED on CRITICAL shadow
    2. Medical strict + shadow_policy="reject" → REJECTED on CRITICAL shadow
    3. Non-strict / policy="none" → flags present, status unchanged
    4. Non-derivable conclusion → distance=∞, no downgrade
    5. Robust conclusion (distance≥5) → no downgrade
    6. Domain not in shadow_apply_domains → no downgrade
    7. Shadow disabled → no flags, no downgrade
    8. NeSyConfig.for_domain presets
    9. End-to-end via NeSyModel.reason() in medical domain
    10. No regression on existing metacognition confidence
"""

from __future__ import annotations

import pytest

from nesy.core.config import MetaCognitionConfig, NeSyConfig
from nesy.core.types import (
    NullItem,
    NullSet,
    NullType,
    OutputStatus,
    Predicate,
    PresentSet,
    SymbolicRule,
)
from nesy.metacognition.monitor import MetaCognitionMonitor


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════


def _pred(name: str, *args: str) -> Predicate:
    return Predicate(name=name, args=tuple(args))


def _rule(rid: str, ants: list, cons: list, weight: float = 1.0) -> SymbolicRule:
    return SymbolicRule(id=rid, antecedents=ants, consequents=cons, weight=weight)


def _empty_null_set() -> NullSet:
    return NullSet(
        items=[],
        present_set=PresentSet(concepts=set(), context_type="general"),
    )


def _make_monitor(**kwargs) -> MetaCognitionMonitor:
    """Build a monitor with sensible defaults for shadow tests."""
    defaults = dict(
        doubt_threshold=0.50,
        strict_mode=False,
        shadow_enabled=True,
        shadow_policy="none",
        shadow_critical_distance=1,
        domain="general",
    )
    defaults.update(kwargs)
    return MetaCognitionMonitor(**defaults)


def _evaluate_with_shadow(
    monitor: MetaCognitionMonitor,
    input_facts: set,
    derived_facts: set,
    rules: list,
    neural_confidence: float = 0.90,
    symbolic_confidence: float = 0.90,
):
    """Shorthand: evaluate with shadow inputs, returns (status, flags)."""
    _, _, status, flags = monitor.evaluate(
        answer="test",
        neural_confidence=neural_confidence,
        symbolic_confidence=symbolic_confidence,
        reasoning_steps=[],
        logic_clauses=[],
        null_set=_empty_null_set(),
        input_facts=input_facts,
        derived_facts=derived_facts,
        rules=rules,
    )
    return status, flags


# ═══════════════════════════════════════════════════════════════════
#  1. MEDICAL STRICT + FLAG POLICY
# ═══════════════════════════════════════════════════════════════════


class TestShadowFlagPolicy:
    """Shadow distance 1 + policy='flag' → FLAGGED."""

    def test_critical_shadow_flags_status(self):
        monitor = _make_monitor(
            shadow_policy="flag",
            domain="medical",
        )
        facts = {_pred("HasSymptom", "p", "fever")}
        rules = [
            _rule(
                "r1", [_pred("HasSymptom", "?p", "fever")], [_pred("Diagnosis", "?p", "infection")]
            )
        ]
        derived = facts | {_pred("Diagnosis", "p", "infection")}

        status, flags = _evaluate_with_shadow(monitor, facts, derived, rules)
        assert status == OutputStatus.FLAGGED
        assert any("SHADOW" in f for f in flags)

    def test_critical_shadow_does_not_upgrade_rejected(self):
        """If status is already REJECTED, shadow flag policy keeps REJECTED."""
        monitor = _make_monitor(
            shadow_policy="flag",
            domain="medical",
        )
        # Use critical nulls to force REJECTED from _determine_status
        null_set = NullSet(
            items=[
                NullItem(
                    concept="blood_test",
                    weight=0.9,
                    null_type=NullType.TYPE3_CRITICAL,
                    expected_because_of=["fever"],
                    criticality=2.0,
                )
            ],
            present_set=PresentSet(concepts={"fever"}, context_type="medical"),
        )
        facts = {_pred("HasSymptom", "p", "fever")}
        rules = [
            _rule(
                "r1", [_pred("HasSymptom", "?p", "fever")], [_pred("Diagnosis", "?p", "infection")]
            )
        ]
        derived = facts | {_pred("Diagnosis", "p", "infection")}

        _, _, status, flags = monitor.evaluate(
            answer="test",
            neural_confidence=0.90,
            symbolic_confidence=0.90,
            reasoning_steps=[],
            logic_clauses=[],
            null_set=null_set,
            input_facts=facts,
            derived_facts=derived,
            rules=rules,
        )
        assert status == OutputStatus.REJECTED


# ═══════════════════════════════════════════════════════════════════
#  2. REJECT POLICY
# ═══════════════════════════════════════════════════════════════════


class TestShadowRejectPolicy:
    """Shadow distance 1 + policy='reject' → REJECTED."""

    def test_critical_shadow_rejects(self):
        monitor = _make_monitor(
            shadow_policy="reject",
            domain="medical",
        )
        facts = {_pred("HasSymptom", "p", "fever")}
        rules = [
            _rule(
                "r1", [_pred("HasSymptom", "?p", "fever")], [_pred("Diagnosis", "?p", "infection")]
            )
        ]
        derived = facts | {_pred("Diagnosis", "p", "infection")}

        status, flags = _evaluate_with_shadow(monitor, facts, derived, rules)
        assert status == OutputStatus.REJECTED


# ═══════════════════════════════════════════════════════════════════
#  3. NONE POLICY — FLAGS WITHOUT DOWNGRADE
# ═══════════════════════════════════════════════════════════════════


class TestShadowNonePolicy:
    """policy='none' → shadow flags present but status unchanged."""

    def test_flags_present_status_unchanged(self):
        monitor = _make_monitor(shadow_policy="none", domain="medical")
        facts = {_pred("HasSymptom", "p", "fever")}
        rules = [
            _rule(
                "r1", [_pred("HasSymptom", "?p", "fever")], [_pred("Diagnosis", "?p", "infection")]
            )
        ]
        derived = facts | {_pred("Diagnosis", "p", "infection")}

        status, flags = _evaluate_with_shadow(monitor, facts, derived, rules)
        # With high confidence and no critical nulls, should be OK
        assert status == OutputStatus.OK
        assert any("SHADOW" in f for f in flags)


# ═══════════════════════════════════════════════════════════════════
#  4. NON-DERIVABLE CONCLUSION → distance=∞, NO DOWNGRADE
# ═══════════════════════════════════════════════════════════════════


class TestNonDerivableConclusion:
    """If conclusion is not derivable from full facts, distance=∞."""

    def test_non_derivable_no_downgrade(self):
        monitor = _make_monitor(shadow_policy="flag", domain="medical")
        facts = {_pred("A", "x")}
        rules = [_rule("r1", [_pred("B", "?x")], [_pred("C", "?x")])]
        # C(x) is not derivable from A(x), but we pretend derived_facts includes it
        # The shadow engine will see C(x) is not derivable and return distance=∞
        derived = facts | {_pred("C", "x")}

        status, flags = _evaluate_with_shadow(monitor, facts, derived, rules)
        # distance=∞ → no downgrade
        assert status == OutputStatus.OK
        # No SHADOW-CRITICAL flag for non-derivable conclusions
        assert not any("SHADOW-CRITICAL" in f for f in flags)


# ═══════════════════════════════════════════════════════════════════
#  5. ROBUST CONCLUSION (d>=5) → NO DOWNGRADE
# ═══════════════════════════════════════════════════════════════════


class TestRobustConclusion:
    """Distance >= 5 → no downgrade regardless of policy."""

    def test_robust_no_downgrade(self):
        monitor = _make_monitor(shadow_policy="flag", domain="medical")
        facts = {
            _pred("HasSymptom", "p", "fever"),
            _pred("HasLabResult", "p", "elevated_wbc"),
            _pred("HasLabResult", "p", "positive_culture"),
            _pred("HasLabResult", "p", "elevated_crp"),
            _pred("HasVitalSign", "p", "tachycardia"),
        }
        rules = [
            _rule(
                "r1", [_pred("HasSymptom", "?p", "fever")], [_pred("Diagnosis", "?p", "infection")]
            ),
            _rule(
                "r2",
                [_pred("HasLabResult", "?p", "elevated_wbc")],
                [_pred("Diagnosis", "?p", "infection")],
            ),
            _rule(
                "r3",
                [_pred("HasLabResult", "?p", "positive_culture")],
                [_pred("Diagnosis", "?p", "infection")],
            ),
            _rule(
                "r4",
                [_pred("HasLabResult", "?p", "elevated_crp")],
                [_pred("Diagnosis", "?p", "infection")],
            ),
            _rule(
                "r5",
                [_pred("HasVitalSign", "?p", "tachycardia")],
                [_pred("Diagnosis", "?p", "infection")],
            ),
        ]
        derived = facts | {_pred("Diagnosis", "p", "infection")}

        status, flags = _evaluate_with_shadow(monitor, facts, derived, rules)
        # 5 independent paths → d=5 → ROBUST → no downgrade
        assert status == OutputStatus.OK


# ═══════════════════════════════════════════════════════════════════
#  6. DOMAIN NOT IN APPLY LIST → NO DOWNGRADE
# ═══════════════════════════════════════════════════════════════════


class TestDomainNotInApplyList:
    """Shadow enforcement is domain-scoped."""

    def test_code_domain_no_downgrade(self):
        monitor = _make_monitor(
            shadow_policy="flag",
            domain="code",
            shadow_apply_domains=["medical", "legal"],
        )
        facts = {_pred("HasBug", "module_x")}
        rules = [_rule("r1", [_pred("HasBug", "?m")], [_pred("NeedsFix", "?m")])]
        derived = facts | {_pred("NeedsFix", "module_x")}

        status, flags = _evaluate_with_shadow(monitor, facts, derived, rules)
        # d=1 but domain is 'code' → not in apply list → no downgrade
        assert status == OutputStatus.OK

    def test_general_domain_no_downgrade(self):
        monitor = _make_monitor(shadow_policy="flag", domain="general")
        facts = {_pred("A", "x")}
        rules = [_rule("r1", [_pred("A", "?x")], [_pred("B", "?x")])]
        derived = facts | {_pred("B", "x")}

        status, flags = _evaluate_with_shadow(monitor, facts, derived, rules)
        assert status == OutputStatus.OK


# ═══════════════════════════════════════════════════════════════════
#  7. SHADOW DISABLED → NO FLAGS, NO DOWNGRADE
# ═══════════════════════════════════════════════════════════════════


class TestShadowDisabled:
    """shadow_enabled=False → shadow is completely skipped."""

    def test_disabled_no_shadow_flags(self):
        monitor = _make_monitor(shadow_enabled=False, shadow_policy="flag", domain="medical")
        facts = {_pred("A", "x")}
        rules = [_rule("r1", [_pred("A", "?x")], [_pred("B", "?x")])]
        derived = facts | {_pred("B", "x")}

        status, flags = _evaluate_with_shadow(monitor, facts, derived, rules)
        assert not any("SHADOW" in f for f in flags)
        assert status == OutputStatus.OK


# ═══════════════════════════════════════════════════════════════════
#  8. CONFIG PRESETS
# ═══════════════════════════════════════════════════════════════════


class TestNeSyConfigPresets:
    """NeSyConfig.for_domain should set shadow defaults."""

    def test_medical_preset(self):
        cfg = NeSyConfig.for_domain("medical")
        assert cfg.metacognition.shadow_enabled is True
        assert cfg.metacognition.shadow_policy == "flag"
        assert cfg.metacognition.shadow_critical_distance == 1

    def test_legal_preset(self):
        cfg = NeSyConfig.for_domain("legal")
        assert cfg.metacognition.shadow_enabled is True
        assert cfg.metacognition.shadow_policy == "flag"

    def test_general_preset(self):
        cfg = NeSyConfig.for_domain("general")
        assert cfg.metacognition.shadow_enabled is True
        assert cfg.metacognition.shadow_policy == "none"

    def test_config_field_defaults(self):
        mc = MetaCognitionConfig()
        assert mc.shadow_enabled is True
        assert mc.shadow_policy == "none"
        assert mc.shadow_critical_distance == 1
        assert mc.shadow_apply_domains == ["medical", "legal"]
        assert mc.shadow_escalation_flag_prefix == "SHADOW"


# ═══════════════════════════════════════════════════════════════════
#  9. END-TO-END VIA NeSyModel.reason()
# ═══════════════════════════════════════════════════════════════════


class TestNeSyModelShadowE2E:
    """End-to-end test through the full API."""

    def test_medical_model_shadow_flag(self):
        """Medical model with single fact → shadow flags in output."""
        from nesy.api.nesy_model import NeSyModel

        model = NeSyModel(domain="medical", doubt_threshold=0.50, strict_mode=False)
        model.add_rule(
            SymbolicRule(
                id="fever_diagnosis",
                antecedents=[Predicate("HasSymptom", ("?p", "fever"))],
                consequents=[Predicate("Diagnosis", ("?p", "infection"))],
                weight=0.85,
            )
        )

        facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
        output = model.reason(facts=facts, context_type="medical")

        # Shadow should have computed CRITICAL (d=1) and flagged
        assert any("SHADOW" in f for f in output.flags)
        assert output.status == OutputStatus.FLAGGED

    def test_general_model_no_downgrade(self):
        """General model → shadow flags present but no downgrade."""
        from nesy.api.nesy_model import NeSyModel

        model = NeSyModel(domain="general", doubt_threshold=0.50)
        model.add_rule(
            SymbolicRule(
                id="r1",
                antecedents=[Predicate("A", ("?x",))],
                consequents=[Predicate("B", ("?x",))],
                weight=0.85,
            )
        )

        facts = {Predicate("A", ("val",))}
        output = model.reason(facts=facts, context_type="general")

        # Shadow flags present but status not downgraded
        assert any("SHADOW" in f for f in output.flags)
        assert output.status == OutputStatus.OK

    def test_shadow_disabled_model(self):
        """Model with shadow_enabled=False → no shadow flags."""
        from nesy.api.nesy_model import NeSyModel

        model = NeSyModel(
            domain="medical",
            doubt_threshold=0.50,
            strict_mode=False,
            shadow_enabled=False,
        )
        model.add_rule(
            SymbolicRule(
                id="r1",
                antecedents=[Predicate("A", ("?x",))],
                consequents=[Predicate("B", ("?x",))],
                weight=0.85,
            )
        )

        facts = {Predicate("A", ("val",))}
        output = model.reason(facts=facts, context_type="medical")
        assert not any("SHADOW" in f for f in output.flags)


# ═══════════════════════════════════════════════════════════════════
#  10. NO REGRESSION ON EXISTING CONFIDENCE
# ═══════════════════════════════════════════════════════════════════


class TestNoRegressionExistingConfidence:
    """Shadow integration must not change confidence computation."""

    def test_confidence_unchanged_by_shadow(self):
        # Evaluate without shadow inputs → baseline
        monitor_no_shadow = MetaCognitionMonitor(
            doubt_threshold=0.50,
            shadow_enabled=False,
        )
        null_set = _empty_null_set()
        conf_base, _, _, _ = monitor_no_shadow.evaluate(
            answer="test",
            neural_confidence=0.90,
            symbolic_confidence=0.85,
            reasoning_steps=[],
            logic_clauses=[],
            null_set=null_set,
        )

        # Now with shadow enabled but no shadow inputs
        monitor_shadow = MetaCognitionMonitor(
            doubt_threshold=0.50,
            shadow_enabled=True,
            shadow_policy="flag",
            domain="medical",
        )
        conf_shadow, _, _, _ = monitor_shadow.evaluate(
            answer="test",
            neural_confidence=0.90,
            symbolic_confidence=0.85,
            reasoning_steps=[],
            logic_clauses=[],
            null_set=null_set,
        )

        # Confidence scores must be identical
        assert conf_base.factual == pytest.approx(conf_shadow.factual, abs=1e-10)
        assert conf_base.reasoning == pytest.approx(conf_shadow.reasoning, abs=1e-10)
        assert conf_base.knowledge_boundary == pytest.approx(
            conf_shadow.knowledge_boundary, abs=1e-10
        )

    def test_no_conclusions_no_shadow_flags(self):
        """If no new facts are derived, no shadow flags."""
        monitor = _make_monitor(shadow_policy="flag", domain="medical")
        facts = {_pred("A", "x")}
        # derived == input → no new conclusions
        status, flags = _evaluate_with_shadow(monitor, facts, facts, [])
        assert not any("SHADOW" in f for f in flags)
        assert status == OutputStatus.OK


# ═══════════════════════════════════════════════════════════════════
#  11. CUSTOM CRITICAL DISTANCE THRESHOLD
# ═══════════════════════════════════════════════════════════════════


class TestCustomThreshold:
    """shadow_critical_distance > 1 raises the bar."""

    def test_distance_2_threshold_no_flag_on_d1(self):
        """With threshold=2, a d=1 conclusion IS flagged (d<=threshold)."""
        monitor = _make_monitor(
            shadow_policy="flag",
            shadow_critical_distance=2,
            domain="medical",
        )
        facts = {_pred("A", "x")}
        rules = [_rule("r1", [_pred("A", "?x")], [_pred("B", "?x")])]
        derived = facts | {_pred("B", "x")}

        status, flags = _evaluate_with_shadow(monitor, facts, derived, rules)
        assert status == OutputStatus.FLAGGED

    def test_distance_2_ok_with_d3(self):
        """d=3 > threshold=2 → no downgrade."""
        monitor = _make_monitor(
            shadow_policy="flag",
            shadow_critical_distance=2,
            domain="medical",
        )
        facts = {
            _pred("Evidence", "x", "a"),
            _pred("Evidence", "x", "b"),
            _pred("Evidence", "x", "c"),
        }
        rules = [
            _rule("r1", [_pred("Evidence", "?x", "a")], [_pred("Conclusion", "?x")]),
            _rule("r2", [_pred("Evidence", "?x", "b")], [_pred("Conclusion", "?x")]),
            _rule("r3", [_pred("Evidence", "?x", "c")], [_pred("Conclusion", "?x")]),
        ]
        derived = facts | {_pred("Conclusion", "x")}

        status, flags = _evaluate_with_shadow(monitor, facts, derived, rules)
        assert status == OutputStatus.OK


# ═══════════════════════════════════════════════════════════════════
#  12. MONITOR SHADOW POLICY VALIDATION
# ═══════════════════════════════════════════════════════════════════


class TestShadowPolicyValidation:
    """Invalid policy should raise."""

    def test_invalid_policy_raises(self):
        with pytest.raises(AssertionError, match="shadow_policy"):
            MetaCognitionMonitor(
                doubt_threshold=0.50,
                shadow_policy="invalid",
            )
