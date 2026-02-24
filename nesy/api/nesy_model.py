"""
nesy/api/nesy_model.py
======================
The main developer-facing API for NeSy-Core.

This is the "PyTorch feel" interface: simple, composable, powerful.
A developer with any background can use NeSy-Core through this
single class without understanding the internals.

Public API:
    nesy = NeSyModel(domain="medical")
    nesy.add_rule(rule)
    output = nesy.reason("What is the likely diagnosis?", facts)
    nesy.learn(new_rule)
    print(output.summary())

Revolutionary Features:
    PCAP  — Proof Capsule export (deterministic, auditable JSON)
    CFG   — Counterfactual Fix Generator (what-if suggestions)
    TB    — Trust Budget (finite confidence currency)
    DCV   — Dual-Channel Verdict (decision + compliance grade)
    ECS   — Edge Consistency Seal (compression-proof guard)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from nesy.continual.learner import ContinualLearner
from nesy.core.exceptions import CriticalNullViolation, NeSyError, SymbolicConflict
from nesy.core.types import (
    ConceptEdge,
    ConfidenceReport,
    NSIOutput,
    NullSet,
    NullType,
    OutputStatus,
    Predicate,
    PresentSet,
    ReasoningStep,
    ReasoningTrace,
    SymbolicRule,
    UnsatCore,
)
from nesy.core.validators import clamp_probability, safe_divide
from nesy.metacognition.fingerprint import compute_reasoning_fingerprint
from nesy.metacognition.monitor import MetaCognitionMonitor
from nesy.neural.nsil import IntegrityReport
from nesy.nsi.concept_graph import ConceptGraphEngine
from nesy.symbolic.engine import SymbolicEngine
from nesy.symbolic.unsat_explanation import (
    enrich_with_null_set,
    explain_constraint_violations,
    explain_unsat_core,
    format_contradiction_report,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  PROOF CAPSULE (PCAP) — serialisable audit packet
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ProofCapsule:
    """Deterministic, self-contained proof artifact.

    Contains everything needed to reproduce or audit a reasoning result
    *without* access to the live model.

    Fields:
        version:     PCAP format version ("1.0").
        domain:      Domain string at time of reasoning.
        answer:      Final answer string.
        status:      OutputStatus value string.
        confidence:  3-dim confidence dict.
        steps:       Ordered reasoning steps.
        null_items:  Null-set items with types.
        flags:       Warning flags.
        checksum:    SHA-256 of canonical JSON (excluding checksum field).
        timestamp:   ISO-8601 UTC.
        request_id:  Unique request ID from NSIOutput.
    """

    version:    str
    domain:     str
    answer:     str
    status:     str
    confidence: Dict[str, float]
    steps:      List[Dict[str, Any]]
    null_items: List[Dict[str, Any]]
    flags:      List[str]
    checksum:   str
    timestamp:  str
    request_id: str
    reasoning_fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        """Return plain dict suitable for ``json.dumps()``."""
        return {
            "version":    self.version,
            "domain":     self.domain,
            "answer":     self.answer,
            "status":     self.status,
            "confidence": self.confidence,
            "steps":      self.steps,
            "null_items": self.null_items,
            "flags":      self.flags,
            "checksum":   self.checksum,
            "timestamp":  self.timestamp,
            "request_id": self.request_id,
            "reasoning_fingerprint": self.reasoning_fingerprint,
        }


def _compute_pcap_checksum(data: Dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON representation (checksum field zeroed)."""
    canonical = dict(data)
    canonical["checksum"] = ""
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════════════════
#  COUNTERFACTUAL FIX (CFG) — minimal hitting-set suggestions
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CounterfactualFix:
    """A single what-if suggestion.

    Attributes:
        missing_concept:   The absent concept that would improve confidence.
        predicted_uplift:  Estimated confidence uplift if concept were present.
        source_null_type:  NullType value of the missing item.
        explanation:       Human-readable suggestion text.
    """

    missing_concept:  str
    predicted_uplift: float
    source_null_type: str
    explanation:      str


# ═══════════════════════════════════════════════════════════════════
#  DUAL-CHANNEL VERDICT (DCV)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DualChannelVerdict:
    """Decision channel + Compliance channel combined.

    Attributes:
        decision:          The answer string (same as NSIOutput.answer).
        decision_status:   OutputStatus value string.
        compliance_grade:  Letter grade: A / B / C / D.
        compliance_detail: Per-dimension compliance breakdown.
        overall_pass:      True if grade is A or B.
    """

    decision:          str
    decision_status:   str
    compliance_grade:  str
    compliance_detail: Dict[str, str]
    overall_pass:      bool


# ═══════════════════════════════════════════════════════════════════
#  TRUST BUDGET (TB)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrustBudgetResult:
    """Result of a trust-budget-aware inference.

    Attributes:
        output:            The full NSIOutput.
        cost:              How much trust budget this inference consumed.
        remaining_budget:  Budget left after this inference.
        budget_exceeded:   True if inference was capped by budget.
    """

    output:           NSIOutput
    cost:             float
    remaining_budget: float
    budget_exceeded:  bool


# ═══════════════════════════════════════════════════════════════════
#  CONTRADICTION REPORT — Unsat-Core → Human Explanation output
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ContradictionReport:
    """Structured output from ``explain_contradiction()``.

    Contains the rejected output, the unsat core analysis,
    a human-readable report, and actionable CFG-integrated repairs.

    Attributes:
        output:           The REJECTED NSIOutput.
        unsat_core:       The minimal conflict set with explanations.
        report:           Formatted human-readable report string.
        fixes:            CFG-enriched repair suggestions.
        conflicting_rules: Shortcut to the conflicting rule IDs.
    """

    output:            NSIOutput
    unsat_core:        UnsatCore
    report:            str
    fixes:             List["CounterfactualFix"]
    conflicting_rules: List[str]


class NeSyModel:
    """The unified NeSy-Core reasoning model.
    
    Composes all five layers into a single, usable object:
        - SymbolicEngine      (logic + rules)
        - ConceptGraphEngine  (NSI null set computation)
        - MetaCognitionMonitor (confidence + self-doubt)
        - ContinualLearner    (no catastrophic forgetting)
    
    Quick start:
        from nesy import NeSyModel, SymbolicRule, Predicate, ConceptEdge
        
        model = NeSyModel(domain="medical")
        
        # Add domain knowledge
        model.add_rule(SymbolicRule(
            id="fever_implies_infection_possible",
            antecedents=[Predicate("HasSymptom", ("?p", "fever"))],
            consequents=[Predicate("PossiblyHas", ("?p", "infection"))],
            weight=0.8,
            description="Fever is a common symptom of infection",
        ))
        
        # Add concept relationships for NSI
        model.add_concept_edge(ConceptEdge(
            source="fever", target="blood_test",
            cooccurrence_prob=0.85, causal_strength=1.0, temporal_stability=1.0,
        ))
        
        # Reason
        facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
        output = model.reason(facts=facts, context_type="medical")
        
        print(output.summary())
        print(output.confidence.explanation)
    """

    def __init__(
        self,
        domain:           str   = "general",
        doubt_threshold:  float = 0.60,
        strict_mode:      bool  = False,
        lambda_ewc:       float = 1000.0,
        shadow_enabled:   bool  = True,
        shadow_policy:    str   = "none",
        shadow_critical_distance: int = 1,
        shadow_apply_domains: Optional[List[str]] = None,
    ):
        self.domain = domain

        # Auto-configure shadow policy for safety-critical domains
        effective_policy = shadow_policy
        if shadow_policy == "none" and domain in ("medical", "legal"):
            effective_policy = "flag"

        # Initialise all layers
        self._symbolic  = SymbolicEngine(domain=domain)
        self._cge       = ConceptGraphEngine(domain=domain)
        self._monitor   = MetaCognitionMonitor(
            doubt_threshold=doubt_threshold,
            strict_mode=strict_mode,
            shadow_enabled=shadow_enabled,
            shadow_policy=effective_policy,
            shadow_critical_distance=shadow_critical_distance,
            shadow_apply_domains=shadow_apply_domains or ["medical", "legal"],
            domain=domain,
        )
        self._learner   = ContinualLearner(lambda_ewc=lambda_ewc)

    # ─── KNOWLEDGE LOADING ─────────────────────────────────────────

    def add_rule(self, rule: SymbolicRule) -> "NeSyModel":
        """Add a symbolic rule. Chainable."""
        self._symbolic.add_rule(rule)
        return self

    def add_rules(self, rules: List[SymbolicRule]) -> "NeSyModel":
        for rule in rules:
            self.add_rule(rule)
        return self

    def add_concept_edge(self, edge: ConceptEdge) -> "NeSyModel":
        """Add a concept graph edge for NSI null set computation. Chainable."""
        self._cge.add_edge(edge)
        return self

    def add_concept_edges(self, edges: List[ConceptEdge]) -> "NeSyModel":
        for edge in edges:
            self.add_concept_edge(edge)
        return self

    def register_critical_concept(self, concept: str, class_label: str) -> "NeSyModel":
        """Register a concept as critical (Type3 null if absent). Chainable."""
        self._cge.register_concept_class(concept, class_label)
        return self

    def anchor(self, rule: SymbolicRule) -> "NeSyModel":
        """Add a rule as an immutable symbolic anchor. Chainable."""
        self._learner.add_symbolic_anchor(rule)
        self._symbolic.add_rule(rule)
        return self

    # ─── MAIN REASONING INTERFACE ──────────────────────────────────

    def reason(
        self,
        facts:            Set[Predicate],
        context_type:     str             = "general",
        neural_confidence: float          = 0.90,
        raw_input:        Optional[str]   = None,
    ) -> NSIOutput:
        """Main reasoning entry point.
        
        Pipeline:
            1. Forward chain over symbolic rules
            2. Compute null set N(X) via concept graph
            3. Run metacognition monitor
            4. Return NSIOutput with full audit trail
        
        Args:
            facts:             Set of Predicate objects representing known facts
            context_type:      Domain context ("medical", "legal", "code", "general")
            neural_confidence: Confidence from upstream neural model (if any), else 0.9
            raw_input:         Original text/query for trace
        
        Returns:
            NSIOutput with answer, confidence, trace, null_set, status, flags
        """
        # ── Step 1: Symbolic reasoning ────────────────────────────
        try:
            derived_facts, reasoning_steps, symbolic_confidence = self._symbolic.reason(facts)
        except SymbolicConflict as e:
            rejected = self._make_rejected_output(
                facts=facts,
                context_type=context_type,
                reason=f"Symbolic conflict: {e}",
            )
            # Attach the UnsatCore to the rejected output for later retrieval
            rejected._unsat_core = getattr(e, "unsat_core", None)
            rejected._conflicting_rules = e.conflicting_rules
            return rejected

        # ── Step 2: Build answer from derived facts ────────────────
        answer = self._facts_to_answer(derived_facts, facts)

        # ── Step 3: NSI null set computation ──────────────────────
        present_concepts = {p.name for p in facts} | {a for p in facts for a in p.args}
        present_set = PresentSet(
            concepts=present_concepts,
            context_type=context_type,
            raw_input=raw_input,
        )
        null_set = self._cge.compute_null_set(present_set)

        # ── Step 4: Metacognition ──────────────────────────────────
        # Build logic_clauses from the reasoning steps returned by SymbolicEngine.
        # Each step (except step 0 = initial facts) corresponds to
        # a forward-chain derivation with an associated LogicClause.
        from nesy.core.types import LogicClause, LogicConnective
        logic_clauses: List[LogicClause] = []
        for step in reasoning_steps:
            if step.rule_applied is not None:
                logic_clauses.append(LogicClause(
                    predicates=step.predicates,
                    connective=LogicConnective.IMPLIES,
                    satisfied=True,
                    weight=step.confidence,
                    source_rule=step.rule_applied,
                ))

        try:
            confidence, trace, status, flags = self._monitor.evaluate(
                answer=answer,
                neural_confidence=neural_confidence,
                symbolic_confidence=symbolic_confidence,
                reasoning_steps=reasoning_steps,
                logic_clauses=logic_clauses,
                null_set=null_set,
                query=raw_input,
                input_facts=facts,
                derived_facts=derived_facts,
                rules=list(self._symbolic.rules),
            )
        except CriticalNullViolation as e:
            return self._make_rejected_output(
                facts=facts,
                context_type=context_type,
                reason=f"Critical null violation: {e}",
            )

        return NSIOutput(
            answer=answer,
            confidence=confidence,
            reasoning_trace=trace,
            null_set=null_set,
            status=status,
            flags=flags,
            reasoning_fingerprint=compute_reasoning_fingerprint(output=NSIOutput(
                answer=answer,
                confidence=confidence,
                reasoning_trace=trace,
                null_set=null_set,
                status=status,
                flags=flags,
            )),
        )

    def learn(
        self,
        new_rule:         SymbolicRule,
        make_anchor:      bool = False,
    ) -> "NeSyModel":
        """Learn a new rule without forgetting existing knowledge.
        
        Args:
            new_rule:    The new SymbolicRule to learn
            make_anchor: If True, make this rule an immutable anchor
        
        If make_anchor=True, this rule joins the permanent
        symbolic store and can never be modified or removed.
        """
        if make_anchor:
            self.anchor(new_rule)
        else:
            self._symbolic.add_rule(new_rule)

        logger.info(
            f"Learned new rule: '{new_rule.id}' "
            f"(anchor={make_anchor}, weight={new_rule.weight})"
        )
        return self

    def explain(self, output: NSIOutput) -> str:
        """Return a human-readable explanation of an NSIOutput.
        
        Useful for debugging, logging, and end-user transparency.
        """
        lines = [
            "=" * 60,
            f"NeSy-Core Explanation",
            "=" * 60,
            f"Answer:  {output.answer}",
            f"Status:  {output.status.value.upper()}",
            "",
            "── Confidence ──────────────────────────────────────────",
            f"  Factual:            {output.confidence.factual:.3f}",
            f"  Reasoning:          {output.confidence.reasoning:.3f}",
            f"  Knowledge Boundary: {output.confidence.knowledge_boundary:.3f}",
            f"  Overall (min):      {output.confidence.minimum:.3f}",
            "",
        ]

        if output.flags:
            lines.append("── Flags ───────────────────────────────────────────────")
            for flag in output.flags:
                lines.append(f"  ⚠ {flag}")
            lines.append("")

        lines.append("── Reasoning Steps ─────────────────────────────────────")
        for step in output.reasoning_trace.steps:
            lines.append(f"  Step {step.step_number}: {step.description}")

        if output.null_set.critical_items:
            lines.append("")
            lines.append("── Critical Nulls ──────────────────────────────────────")
            for item in output.null_set.critical_items:
                lines.append(f"  ✗ CRITICAL: '{item.concept}' is absent")

        if output.null_set.meaningful_items:
            lines.append("")
            lines.append("── Meaningful Nulls ─────────────────────────────────────")
            for item in output.null_set.meaningful_items:
                lines.append(f"  → '{item.concept}' may be relevant (weight={item.weight:.2f})")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ─── SAVE / LOAD ───────────────────────────────────────────────

    def save_concept_graph(self, path: str) -> None:
        self._cge.save(path)

    def load_concept_graph(self, path: str) -> "NeSyModel":
        self._cge = ConceptGraphEngine.load(path)
        return self

    # ─── PRIVATE HELPERS ───────────────────────────────────────────

    def _facts_to_answer(
        self,
        derived: Set[Predicate],
        original: Set[Predicate],
    ) -> str:
        """Build a human-readable answer from derived facts.
        
        Returns the new facts (derived minus original) as the 'answer'.
        In a full implementation this would use a neural decoder
        to generate natural language.
        """
        new_facts = derived - original
        if not new_facts:
            return "No new conclusions can be derived from the given facts."
        conclusions = ", ".join(str(p) for p in sorted(new_facts, key=str))
        return f"Derived: {conclusions}"

    def _make_rejected_output(
        self,
        facts: Set[Predicate],
        context_type: str,
        reason: str,
    ) -> NSIOutput:
        """Build a REJECTED NSIOutput when hard failure occurs."""
        from nesy.core.types import (
            ConfidenceReport,
            NullSet,
            ReasoningTrace,
            PresentSet,
        )
        present_set = PresentSet(
            concepts={p.name for p in facts},
            context_type=context_type,
        )
        null_set = NullSet(items=[], present_set=present_set)
        trace = ReasoningTrace(
            steps=[ReasoningStep(
                step_number=0,
                description=f"Reasoning halted: {reason}",
                rule_applied=None,
                predicates=[],
                confidence=0.0,
            )],
            rules_activated=[],
            neural_confidence=0.0,
            symbolic_confidence=0.0,
            null_violations=[],
            logic_clauses=[],
        )
        confidence = ConfidenceReport(
            factual=0.0, reasoning=0.0, knowledge_boundary=0.0
        )
        rejected = NSIOutput(
            answer="",
            confidence=confidence,
            reasoning_trace=trace,
            null_set=null_set,
            status=OutputStatus.REJECTED,
            flags=[reason],
        )
        rejected.reasoning_fingerprint = compute_reasoning_fingerprint(output=rejected)
        return rejected

    # ─── PROPERTIES ────────────────────────────────────────────────

    @property
    def rule_count(self) -> int:
        return len(self._symbolic.rules)

    @property
    def concept_graph_stats(self) -> Dict:
        return self._cge.stats

    @property
    def anchored_rules(self) -> int:
        return self._learner.anchor_count

    # ═══════════════════════════════════════════════════════════════
    #  REVOLUTIONARY FEATURE 1: PROOF CAPSULE (PCAP)
    # ═══════════════════════════════════════════════════════════════

    def export_proof_capsule(
        self,
        output: NSIOutput,
        path: Optional[str] = None,
    ) -> ProofCapsule:
        """Export a deterministic, self-contained proof capsule.

        The capsule contains everything needed to audit or reproduce
        a reasoning result without access to the live model.

        Args:
            output: An ``NSIOutput`` returned by ``reason()``.
            path:   Optional filesystem path. If given, writes a
                    ``*.pcap.json`` file.

        Returns:
            ``ProofCapsule`` with SHA-256 integrity checksum.
        """
        steps = [
            {
                "step_number": s.step_number,
                "description": s.description,
                "rule_applied": s.rule_applied,
                "predicates": [str(p) for p in s.predicates],
                "confidence": s.confidence,
            }
            for s in output.reasoning_trace.steps
        ]
        null_items = [
            {
                "concept": ni.concept,
                "weight": ni.weight,
                "null_type": ni.null_type.name,
                "expected_because_of": ni.expected_because_of,
            }
            for ni in output.null_set.items
        ]
        confidence_dict = {
            "factual": output.confidence.factual,
            "reasoning": output.confidence.reasoning,
            "knowledge_boundary": output.confidence.knowledge_boundary,
        }

        capsule_data: Dict[str, Any] = {
            "version":    "1.0",
            "domain":     self.domain,
            "answer":     output.answer,
            "status":     output.status.value,
            "confidence": confidence_dict,
            "steps":      steps,
            "null_items": null_items,
            "flags":      list(output.flags),
            "checksum":   "",
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "request_id": output.request_id,
            "reasoning_fingerprint": output.reasoning_fingerprint or "",
        }

        capsule_data["checksum"] = _compute_pcap_checksum(capsule_data)

        capsule = ProofCapsule(**capsule_data)

        if path is not None:
            Path(path).write_text(
                json.dumps(capsule.to_dict(), indent=2, sort_keys=True)
            )
            logger.info(f"Proof capsule written: {path}")

        return capsule

    @staticmethod
    def verify_proof_capsule(capsule: ProofCapsule) -> bool:
        """Verify the integrity of a proof capsule via SHA-256 checksum.

        Returns:
            ``True`` if the checksum matches, ``False`` if tampered.
        """
        expected = _compute_pcap_checksum(capsule.to_dict())
        return expected == capsule.checksum

    @staticmethod
    def load_proof_capsule(path: str) -> ProofCapsule:
        """Load a proof capsule from a ``.pcap.json`` file.

        Args:
            path: Filesystem path to the JSON file.

        Returns:
            ``ProofCapsule`` instance.

        Raises:
            NeSyError: If the file cannot be read or parsed.
        """
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise NeSyError(
                f"Failed to load proof capsule from {path}: {exc}",
                context={"path": path},
            ) from exc
        return ProofCapsule(**data)

    # ═══════════════════════════════════════════════════════════════
    #  REVOLUTIONARY FEATURE 2: COUNTERFACTUAL FIX GENERATOR (CFG)
    # ═══════════════════════════════════════════════════════════════

    def suggest_fixes(self, output: NSIOutput) -> List[CounterfactualFix]:
        """Generate actionable what-if suggestions from an NSIOutput.

        Uses a minimal hitting-set approach: for every Type2/Type3 null
        item, compute the predicted confidence uplift if that concept
        were added to the present set.

        The uplift formula:
            uplift = anomaly_contribution / (1 + total_anomaly)

        Returns:
            Sorted list of ``CounterfactualFix`` (highest uplift first).
        """
        fixes: List[CounterfactualFix] = []
        total_anomaly = output.null_set.total_anomaly_score

        for item in output.null_set.items:
            if item.null_type == NullType.TYPE1_EXPECTED:
                continue

            uplift = safe_divide(item.anomaly_contribution, 1.0 + total_anomaly)
            uplift = clamp_probability(uplift, label=f"uplift({item.concept})")

            if item.null_type == NullType.TYPE3_CRITICAL:
                explanation = (
                    f"CRITICAL: Adding '{item.concept}' would fill a critical "
                    f"gap (expected because of: "
                    f"{', '.join(item.expected_because_of)}). "
                    f"Predicted uplift: +{uplift:.3f}"
                )
            else:
                explanation = (
                    f"Adding '{item.concept}' may improve confidence "
                    f"(weight={item.weight:.2f}, expected because of: "
                    f"{', '.join(item.expected_because_of)}). "
                    f"Predicted uplift: +{uplift:.3f}"
                )

            fixes.append(CounterfactualFix(
                missing_concept=item.concept,
                predicted_uplift=uplift,
                source_null_type=item.null_type.name,
                explanation=explanation,
            ))

        fixes.sort(key=lambda f: -f.predicted_uplift)
        return fixes

    # ═══════════════════════════════════════════════════════════════
    #  REVOLUTIONARY FEATURE 6: UNSAT-CORE → HUMAN EXPLANATION (UCE)
    # ═══════════════════════════════════════════════════════════════

    def explain_contradiction(
        self,
        output: NSIOutput,
        include_null_set: bool = True,
    ) -> ContradictionReport:
        """Produce a full contradiction report from a REJECTED output.

        This is the "reasoning auditor + repair agent" feature.
        When ``reason()`` returns a REJECTED output (due to a symbolic
        conflict), this method:

          1. Extracts the ``UnsatCore`` from the rejected output.
          2. Enriches it with NSI null-set insights (critical absences).
          3. Generates CFG-integrated repair suggestions.
          4. Formats a human-readable contradiction report.

        If the output is NOT rejected (no contradiction), returns a
        report with an empty core and "No contradiction" message.

        Args:
            output:           An ``NSIOutput`` (typically REJECTED).
            include_null_set: Whether to enrich repairs with NSI nulls.

        Returns:
            ``ContradictionReport`` with core, report text, and fixes.
        """
        # ── Extract or build UnsatCore ─────────────────────────
        unsat_core = getattr(output, "_unsat_core", None)
        conflicting_rules = getattr(output, "_conflicting_rules", [])

        if unsat_core is None:
            if output.status == OutputStatus.REJECTED:
                # Build a basic UnsatCore from the rejection flags
                unsat_core = UnsatCore(
                    conflicting_rule_ids=list(conflicting_rules),
                    explanation=(
                        "Symbolic conflict detected. "
                        + (output.flags[0] if output.flags else "Unknown cause.")
                    ),
                )
            else:
                # No contradiction — return empty report
                unsat_core = UnsatCore(
                    explanation="No contradiction detected.",
                )

        # ── Enrich with NSI null set ───────────────────────────
        if include_null_set and output.null_set.items:
            unsat_core = enrich_with_null_set(unsat_core, output.null_set)

        # ── Generate CFG-integrated fixes ──────────────────────
        cfg_fixes = self.suggest_fixes(output)

        # Add unsat-core repair suggestions as additional fixes
        for ra in unsat_core.repair_actions:
            concept = ra.get("concept", "")
            # Avoid duplicate concepts already in CFG
            if not any(f.missing_concept == concept for f in cfg_fixes):
                cfg_fixes.append(CounterfactualFix(
                    missing_concept=concept,
                    predicted_uplift=0.0,  # unsat-core repairs are qualitative
                    source_null_type="UNSAT_CORE",
                    explanation=ra.get("reason", f"Add '{concept}' to resolve conflict."),
                ))

        # ── Format report ──────────────────────────────────────
        report = format_contradiction_report(unsat_core)

        return ContradictionReport(
            output=output,
            unsat_core=unsat_core,
            report=report,
            fixes=cfg_fixes,
            conflicting_rules=list(unsat_core.conflicting_rule_ids),
        )

    # ═══════════════════════════════════════════════════════════════
    #  REVOLUTIONARY FEATURE 3: TRUST BUDGET (TB)
    # ═══════════════════════════════════════════════════════════════

    def trust_budget_reason(
        self,
        facts: Set[Predicate],
        budget: float,
        context_type: str = "general",
        neural_confidence: float = 0.90,
        raw_input: Optional[str] = None,
    ) -> TrustBudgetResult:
        """Reason with a finite trust budget.

        Each inference consumes trust proportional to the complexity
        of the reasoning chain and the null-set anomaly score.

        Cost formula:
            cost = (num_steps * 0.1) + (total_anomaly * 0.2) + base_cost

        base_cost = 0.05 (minimum cost for any reasoning call).

        If the budget is exhausted the inference still runs but
        ``budget_exceeded`` is set so callers can throttle.

        Args:
            facts:              Input facts.
            budget:             Available trust budget (> 0).
            context_type:       Domain context string.
            neural_confidence:  Upstream neural confidence.
            raw_input:          Original query text.

        Returns:
            ``TrustBudgetResult`` with output, cost, remaining budget.

        Raises:
            NeSyError: if budget ≤ 0.
        """
        if budget <= 0:
            raise NeSyError(
                "Trust budget must be positive",
                context={"budget": budget},
            )

        output = self.reason(
            facts=facts,
            context_type=context_type,
            neural_confidence=neural_confidence,
            raw_input=raw_input,
        )

        base_cost = 0.05
        step_cost = len(output.reasoning_trace.steps) * 0.1
        anomaly_cost = output.null_set.total_anomaly_score * 0.2
        cost = base_cost + step_cost + anomaly_cost

        remaining = budget - cost
        exceeded = remaining < 0

        return TrustBudgetResult(
            output=output,
            cost=cost,
            remaining_budget=max(0.0, remaining),
            budget_exceeded=exceeded,
        )

    # ═══════════════════════════════════════════════════════════════
    #  REVOLUTIONARY FEATURE 4: DUAL-CHANNEL VERDICT (DCV)
    # ═══════════════════════════════════════════════════════════════

    def dual_channel_verdict(self, output: NSIOutput) -> DualChannelVerdict:
        """Produce a dual-channel verdict: Decision + Compliance.

        Decision channel:  The answer and status from the NSIOutput.
        Compliance channel: A letter grade (A/B/C/D) based on:
            - Confidence dimensions
            - Null-set severity
            - Flag count

        Grade thresholds:
            A — min_confidence ≥ 0.8 AND no critical nulls AND ≤ 1 flag
            B — min_confidence ≥ 0.6 AND no critical nulls
            C — min_confidence ≥ 0.4 OR status not REJECTED
            D — everything else (REJECTED or min_confidence < 0.4)

        Args:
            output: An ``NSIOutput`` from ``reason()``.

        Returns:
            ``DualChannelVerdict`` with grade and detail breakdown.
        """
        min_conf = output.confidence.minimum
        n_critical = len(output.null_set.critical_items)
        n_flags = len(output.flags)

        # Compliance grading
        if min_conf >= 0.8 and n_critical == 0 and n_flags <= 1:
            grade = "A"
        elif min_conf >= 0.6 and n_critical == 0:
            grade = "B"
        elif min_conf >= 0.4 or output.status != OutputStatus.REJECTED:
            grade = "C"
        else:
            grade = "D"

        detail = {
            "factual": f"{output.confidence.factual:.3f}",
            "reasoning": f"{output.confidence.reasoning:.3f}",
            "knowledge_boundary": f"{output.confidence.knowledge_boundary:.3f}",
            "critical_nulls": str(n_critical),
            "flags": str(n_flags),
        }

        return DualChannelVerdict(
            decision=output.answer,
            decision_status=output.status.value,
            compliance_grade=grade,
            compliance_detail=detail,
            overall_pass=grade in ("A", "B"),
        )

    # ═══════════════════════════════════════════════════════════════
    #  REVOLUTIONARY FEATURE 5: EDGE CONSISTENCY SEAL (ECS)
    # ═══════════════════════════════════════════════════════════════

    def edge_consistency_seal(
        self,
        facts: Set[Predicate],
        context_type: str = "general",
        neural_confidence: float = 0.90,
    ) -> Dict[str, Any]:
        """Compute a consistency seal over the concept graph.

        The seal captures:
            1. Full reasoning output on the given facts.
            2. SHA-256 checksum of the concept graph edges (sorted).
            3. Null-set summary.

        After compression (NeSy-Lite) the seal can be recomputed on
        the compressed graph and compared to detect semantic drift.

        Args:
            facts:              Input facts for probe reasoning.
            context_type:       Domain context string.
            neural_confidence:  Upstream neural confidence.

        Returns:
            Dict with ``graph_checksum``, ``answer``, ``status``,
            ``confidence_minimum``, ``null_count``, ``seal_timestamp``.
        """
        output = self.reason(
            facts=facts,
            context_type=context_type,
            neural_confidence=neural_confidence,
        )

        # Deterministic graph checksum: sort edges then SHA-256
        edge_strs: List[str] = []
        for source_dict in self._cge._graph.values():
            for edge in source_dict.values():
                edge_strs.append(
                    f"{edge.source}|{edge.target}|"
                    f"{edge.cooccurrence_prob}|"
                    f"{edge.causal_strength}|"
                    f"{edge.temporal_stability}"
                )
        edge_strs.sort()
        graph_hash = hashlib.sha256(
            "\n".join(edge_strs).encode("utf-8")
        ).hexdigest()

        return {
            "graph_checksum": graph_hash,
            "answer": output.answer,
            "status": output.status.value,
            "confidence_minimum": output.confidence.minimum,
            "null_count": len(output.null_set.items),
            "seal_timestamp": datetime.now(timezone.utc).isoformat(),
        }