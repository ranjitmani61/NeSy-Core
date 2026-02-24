"""
nesy/metacognition/monitor.py
=============================
MetaCognitionMonitor — The PRIMARY NOVELTY of NeSy-Core.

This module answers: "Does the system know what it knows?"

Mathematical basis:
    Three independent confidence dimensions (orthogonal axes):

    C_factual(q, kb)   = consistency of answer with known facts
                       = 1 - (violations / total_checks)

    C_reasoning(chain) = geometric mean of rule weights in derivation
                       = ∏(wᵢ)^(1/n)

    C_boundary(q, D)   = similarity of query to training distribution
                       = 1 / (1 + mean_distance_to_k_nearest_neighbors)

    Overall reliability gate: min(C_factual, C_reasoning, C_boundary)
    → The system is only as confident as its weakest dimension.

    Self-doubt threshold: if min < τ, output is HELD until operator review.
    This is the "SelfDoubt" mechanism — no other production AI framework
    exposes this as a configurable developer API.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Set, Tuple

from nesy.core.exceptions import CriticalNullViolation
from nesy.core.types import (
    ConfidenceReport,
    ConfidenceType,
    LogicClause,
    NullSet,
    NullType,
    OutputStatus,
    Predicate,
    ReasoningStep,
    ReasoningTrace,
    SymbolicRule,
)
from nesy.neural.nsil import IntegrityReport
from nesy.metacognition.shadow import (
    CounterfactualShadowEngine,
    ShadowReport,
    shadow_flags,
)

logger = logging.getLogger(__name__)


class MetaCognitionMonitor:
    """Monitors the system's own reasoning quality.

    Every output passes through the MetaCognitionMonitor before
    being returned to the caller. The monitor:

        1. Computes three-dimensional confidence
        2. Classifies output status (ok / uncertain / flagged / rejected)
        3. Builds the full reasoning trace (audit trail)
        4. Applies self-doubt: holds output if confidence below threshold

    Configuration:
        doubt_threshold:   min confidence to auto-approve output (default 0.6)
        strict_mode:       if True, Type3 null violations raise exception
        trace_all_steps:   if True, include every derivation step in trace
    """

    def __init__(
        self,
        doubt_threshold: float = 0.60,
        strict_mode: bool = False,
        trace_all_steps: bool = True,
        shadow_enabled: bool = False,
        shadow_critical_distance: int = 1,
        shadow_policy: str = "none",
        shadow_apply_domains: Optional[List[str]] = None,
        shadow_escalation_flag_prefix: str = "SHADOW",
        shadow_max_exact_facts: int = 15,
        shadow_max_depth: int = 7,
        domain: Optional[str] = None,
    ):
        assert 0.0 < doubt_threshold < 1.0
        self.doubt_threshold = doubt_threshold
        self.strict_mode = strict_mode
        self.trace_all_steps = trace_all_steps

        # Shadow configuration
        self.shadow_enabled = shadow_enabled
        self.shadow_critical_distance = shadow_critical_distance
        assert shadow_policy in ("none", "flag", "reject"), (
            f"shadow_policy must be 'none', 'flag', or 'reject', got '{shadow_policy}'"
        )
        self.shadow_policy = shadow_policy
        self.shadow_apply_domains = shadow_apply_domains or ["medical", "legal"]
        self.shadow_escalation_flag_prefix = shadow_escalation_flag_prefix
        self.domain = domain

        # Lazily instantiated shadow engine (only when shadow_enabled)
        self._shadow_engine: Optional[CounterfactualShadowEngine] = None
        if self.shadow_enabled:
            self._shadow_engine = CounterfactualShadowEngine(
                max_exact_facts=shadow_max_exact_facts,
                max_shadow_depth=shadow_max_depth,
            )

        # Calibration data: stores (predicted_confidence, actual_correct)
        # Used by calibration.py to tune thresholds
        self._calibration_log: List[Tuple[float, bool]] = []

    # ─── MAIN MONITOR METHOD ───────────────────────────────────────

    def evaluate(
        self,
        answer: str,
        neural_confidence: float,
        symbolic_confidence: float,
        reasoning_steps: List[ReasoningStep],
        logic_clauses: List[LogicClause],
        null_set: NullSet,
        query: Optional[str] = None,
        integrity_report: Optional[IntegrityReport] = None,
        input_facts: Optional[Set[Predicate]] = None,
        derived_facts: Optional[Set[Predicate]] = None,
        rules: Optional[List[SymbolicRule]] = None,
    ) -> Tuple[ConfidenceReport, ReasoningTrace, OutputStatus, List[str]]:
        """Evaluate the quality of a reasoning result.

        This is the core metacognitive act: the system reflecting on
        the quality of its own output before releasing it.

        Shadow reasoning (when enabled):
            If input_facts, derived_facts, and rules are provided and
            shadow_enabled is True, the monitor computes a ShadowReport
            for every *new* conclusion (derived_facts - input_facts).
            Shadow flags are appended.  If the domain matches
            shadow_apply_domains and shadow_policy is 'flag' or 'reject',
            CRITICAL conclusions trigger a status downgrade.

        Returns:
            confidence_report: three-dimensional epistemic scores
            reasoning_trace:   full auditable derivation
            status:            ok / uncertain / flagged / rejected
            flags:             human-readable warning messages
        """
        flags: List[str] = []

        # ── Step 1: Compute three confidence dimensions ───────────
        c_factual = self._compute_factual_confidence(null_set)
        c_reasoning = self._compute_reasoning_confidence(symbolic_confidence, logic_clauses)
        c_boundary = self._compute_boundary_confidence(neural_confidence, null_set)

        # ── NSIL integrity coupling (deterministic) ────────────
        if integrity_report is not None and not integrity_report.is_neutral:
            c_boundary = max(0.0, min(1.0, c_boundary * integrity_report.integrity_score))

        confidence = ConfidenceReport(
            factual=c_factual,
            reasoning=c_reasoning,
            knowledge_boundary=c_boundary,
            explanation={
                ConfidenceType.FACTUAL: self._explain_factual(c_factual, null_set),
                ConfidenceType.REASONING: self._explain_reasoning(c_reasoning, logic_clauses),
                ConfidenceType.KNOWLEDGE_BOUNDARY: self._explain_boundary(c_boundary),
            },
        )

        # ── Step 2: Build reasoning trace ─────────────────────────
        trace = ReasoningTrace(
            steps=reasoning_steps if self.trace_all_steps else reasoning_steps[-3:],
            rules_activated=[s.rule_applied for s in reasoning_steps if s.rule_applied],
            neural_confidence=neural_confidence,
            symbolic_confidence=symbolic_confidence,
            null_violations=null_set.items,
            logic_clauses=logic_clauses,
        )

        # ── Step 3: Determine output status ───────────────────────
        status, new_flags = self._determine_status(confidence, null_set)
        flags.extend(new_flags)

        # ── Step 3b: NSIL flags integration ─────────────────────
        if integrity_report is not None and not integrity_report.is_neutral:
            flags.extend(integrity_report.flags)

        # ── Step 3c: Shadow reasoning ─────────────────────────────
        shadow_report = self._compute_shadow(input_facts, derived_facts, rules)
        if shadow_report is not None:
            shadow_fl = shadow_flags(shadow_report)
            flags.extend(shadow_fl)
            status = self._apply_shadow_policy(status, shadow_report)

        # ── Step 4: Strict mode — Type3 null → exception ──────────
        if self.strict_mode and null_set.critical_items:
            raise CriticalNullViolation(
                f"Strict mode: {len(null_set.critical_items)} critical null violations.",
                critical_items=null_set.critical_items,
            )

        # ── Step 5: Log for calibration ───────────────────────────
        self._calibration_log.append((confidence.minimum, status == OutputStatus.OK))

        return confidence, trace, status, flags

    # ─── SHADOW COMPUTATION ────────────────────────────────────────

    def _compute_shadow(
        self,
        input_facts: Optional[Set[Predicate]],
        derived_facts: Optional[Set[Predicate]],
        rules: Optional[List[SymbolicRule]],
    ) -> Optional[ShadowReport]:
        """Compute counterfactual shadow for all new conclusions.

        New conclusions = derived_facts - input_facts.
        Returns None if shadow is disabled or inputs are missing.
        """
        if not self.shadow_enabled or self._shadow_engine is None:
            return None
        if input_facts is None or derived_facts is None or rules is None:
            return None

        conclusions = derived_facts - input_facts
        if not conclusions:
            logger.debug("Shadow: no new conclusions to analyse.")
            return None

        report = self._shadow_engine.compute_all(conclusions, input_facts, rules)
        logger.info(
            "Shadow analysis: %d conclusion(s), min_distance=%s, class=%s",
            len(conclusions),
            "inf" if math.isinf(report.minimum_distance) else int(report.minimum_distance),
            report.system_class.value,
        )
        return report

    def _apply_shadow_policy(
        self,
        current_status: OutputStatus,
        shadow_report: ShadowReport,
    ) -> OutputStatus:
        """Downgrade output status based on shadow policy and domain.

        Rules:
            1. If shadow_policy == "none" → no change.
            2. If domain not in shadow_apply_domains → no change.
            3. If min_shadow_distance <= shadow_critical_distance:
                 policy "flag"   → at most FLAGGED
                 policy "reject" → REJECTED
            4. Never *upgrade* status (e.g. REJECTED stays REJECTED).
        """
        if self.shadow_policy == "none":
            return current_status

        if self.domain is not None and self.domain not in self.shadow_apply_domains:
            return current_status

        min_dist = shadow_report.minimum_distance
        if math.isinf(min_dist):
            return current_status

        if min_dist <= self.shadow_critical_distance:
            if self.shadow_policy == "reject":
                logger.warning(
                    "Shadow policy REJECT: min distance %d <= threshold %d",
                    int(min_dist),
                    self.shadow_critical_distance,
                )
                return OutputStatus.REJECTED
            elif self.shadow_policy == "flag":
                # Downgrade but don't upgrade from REJECTED
                if current_status == OutputStatus.REJECTED:
                    return OutputStatus.REJECTED
                logger.warning(
                    "Shadow policy FLAG: min distance %d <= threshold %d, downgrading to FLAGGED",
                    int(min_dist),
                    self.shadow_critical_distance,
                )
                return OutputStatus.FLAGGED

        return current_status

    # ─── CONFIDENCE COMPUTATION ────────────────────────────────────

    def _compute_factual_confidence(self, null_set: NullSet) -> float:
        """C_factual: how consistent is the answer with known facts?

        Penalised by:
        - Type3 (critical) null violations: heavy penalty
        - Type2 (meaningful) null violations: moderate penalty

        Formula:
            C_factual = 1 / (1 + Σ anomaly_contribution)

        This is a monotone decreasing function of violation severity.
        Perfect: no violations → C_factual = 1.0
        """
        anomaly = null_set.total_anomaly_score
        return 1.0 / (1.0 + anomaly)

    def _compute_reasoning_confidence(
        self,
        symbolic_confidence: float,
        logic_clauses: List[LogicClause],
    ) -> float:
        """C_reasoning: how valid is the logical derivation chain?

        Combines:
        - symbolic_confidence from SymbolicEngine (geometric mean of rule weights)
        - proportion of logic clauses that were satisfied

        Formula:
            satisfied_ratio = |satisfied clauses| / |total clauses|
            C_reasoning = √(symbolic_confidence × satisfied_ratio)
                        = geometric mean of the two components
        """
        if not logic_clauses:
            return symbolic_confidence

        satisfied_ratio = sum(1 for c in logic_clauses if c.satisfied) / len(logic_clauses)
        # Geometric mean ensures both components must be high
        return math.sqrt(symbolic_confidence * satisfied_ratio)

    def _compute_boundary_confidence(
        self,
        neural_confidence: float,
        null_set: NullSet,
    ) -> float:
        """C_boundary: is this query within the system's competence?

        Proxy computation (full implementation requires embedding-space
        distance to training distribution — done in calibration.py):

        Here we use:
        - Neural confidence as a base estimate
        - Discount if many Type1 (expected) items are also missing
          (if even obvious background concepts aren't there, we're in
          unfamiliar territory)

        Formula:
            background_gap = |Type1 items| / max(|total expected|, 1)
            C_boundary = neural_confidence × (1 - 0.3 × background_gap)
        """
        total_expected = len(null_set.items) if null_set.items else 1
        type1_count = sum(1 for i in null_set.items if i.null_type == NullType.TYPE1_EXPECTED)
        background_gap = type1_count / total_expected

        return neural_confidence * (1.0 - 0.3 * background_gap)

    # ─── STATUS DETERMINATION ──────────────────────────────────────

    def _determine_status(
        self,
        confidence: ConfidenceReport,
        null_set: NullSet,
    ) -> Tuple[OutputStatus, List[str]]:
        """Determine output status based on confidence and null violations.

        Decision logic:
            REJECTED:  Type3 critical nulls present (hard failure)
            FLAGGED:   confidence.minimum < doubt_threshold (soft failure)
            UNCERTAIN: confidence.minimum ∈ [doubt_threshold, 0.75)
            OK:        confidence.minimum ≥ 0.75
        """
        flags: List[str] = []

        if null_set.critical_items:
            for item in null_set.critical_items:
                flags.append(
                    f"CRITICAL: '{item.concept}' is absent but causally required "
                    f"(expected because of: {', '.join(item.expected_because_of)})"
                )
            return OutputStatus.REJECTED, flags

        if null_set.meaningful_items:
            for item in null_set.meaningful_items[:3]:  # top 3 by weight
                flags.append(
                    f"NOTE: '{item.concept}' is absent but may be relevant "
                    f"(weight={item.weight:.2f}, expected from: {item.expected_because_of[0]})"
                )

        min_conf = confidence.minimum

        if min_conf < self.doubt_threshold:
            flags.append(
                f"SELF-DOUBT: minimum confidence {min_conf:.2f} < threshold {self.doubt_threshold:.2f}. "
                f"Weakest dimension: {self._weakest_dimension(confidence)}"
            )
            return OutputStatus.FLAGGED, flags

        if min_conf < 0.75:
            return OutputStatus.UNCERTAIN, flags

        return OutputStatus.OK, flags

    def _weakest_dimension(self, confidence: ConfidenceReport) -> str:
        dims = {
            "factual": confidence.factual,
            "reasoning": confidence.reasoning,
            "knowledge_boundary": confidence.knowledge_boundary,
        }
        return min(dims, key=lambda k: dims[k])

    # ─── EXPLANATION GENERATORS ────────────────────────────────────

    def _explain_factual(self, score: float, null_set: NullSet) -> str:
        if score >= 0.9:
            return "No significant null violations detected."
        type2 = len(null_set.meaningful_items)
        type3 = len(null_set.critical_items)
        return f"Score reduced by {type3} critical and {type2} meaningful null violations."

    def _explain_reasoning(self, score: float, clauses: List[LogicClause]) -> str:
        if not clauses:
            return "No symbolic inference performed."
        unsatisfied = [c for c in clauses if not c.satisfied]
        if not unsatisfied:
            return f"All {len(clauses)} logic clauses satisfied."
        return f"{len(unsatisfied)} of {len(clauses)} clauses unsatisfied."

    def _explain_boundary(self, score: float) -> str:
        if score >= 0.8:
            return "Query appears within system's competence."
        if score >= 0.5:
            return "Query partially outside training distribution."
        return "Query may be significantly out-of-distribution."

    # ─── CALIBRATION DATA ACCESS ───────────────────────────────────

    @property
    def calibration_data(self) -> List[Tuple[float, bool]]:
        """Returns (predicted_confidence, was_actually_correct) pairs."""
        return list(self._calibration_log)

    def reset_calibration_log(self) -> None:
        self._calibration_log.clear()
