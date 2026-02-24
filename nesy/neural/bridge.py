"""
nesy/neural/bridge.py
=====================
Bidirectional Neural-Symbolic Bridge.

Direction 1 (Neural → Symbolic):
    embedding → grounded predicates → symbolic engine input

Direction 2 (Symbolic → Neural):
    symbolic constraints → loss penalty → gradient signal

This is where the two worlds meet. The bridge is stateless —
it transforms representations, it does not store state.

Mathematical basis (Symbolic → Neural loss):
    L_total = L_task + α × L_symbolic

    L_symbolic = Σᵣ wᵣ × max(0, 1 - satisfaction_score(r, output))

    Where:
        wᵣ               = rule weight
        satisfaction_score = how well output satisfies rule r
        α                = symbolic loss coefficient (hyperparameter)

    This allows symbolic rules to directly shape gradient updates
    without modifying the symbolic engine — the bridge translates
    constraint violations into differentiable loss terms.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set, Tuple

from nesy.core.types import GroundedSymbol, NullSet, Predicate, SymbolicRule
from nesy.neural.grounding import SymbolGrounder
from nesy.neural.nsil import IntegrityReport, compute_integrity_report

logger = logging.getLogger(__name__)


class NeuralSymbolicBridge:
    """Connects neural encoder output to the symbolic reasoning engine.

    Usage:
        bridge = NeuralSymbolicBridge(grounder, symbolic_loss_alpha=0.5)

        # Neural → Symbolic
        predicates = bridge.neural_to_symbolic(embedding, domain="medical")

        # Symbolic → Loss (during training)
        penalty = bridge.symbolic_to_loss(output_embedding, violated_rules)
    """

    def __init__(
        self,
        grounder: SymbolGrounder,
        symbolic_loss_alpha: float = 0.5,
    ):
        self.grounder = grounder
        self.alpha = symbolic_loss_alpha

    # ── Direction 1: Neural → Symbolic ───────────────────────────

    def neural_to_symbolic(
        self,
        embedding: List[float],
        domain: Optional[str] = None,
    ) -> Tuple[Set[Predicate], float]:
        """Convert neural embedding to a set of symbolic predicates.

        Returns:
            predicates:           Set of grounded Predicates for symbolic engine
            grounding_confidence: How confidently we made this conversion
        """
        grounded_symbols, confidence = self.grounder.ground_with_confidence(embedding)

        if not grounded_symbols:
            logger.debug("No predicates grounded from embedding (below threshold).")
            return set(), 0.0

        predicates = {gs.predicate for gs in grounded_symbols}
        return predicates, confidence

    # ── Direction 2: Symbolic → Neural Loss ──────────────────────

    def symbolic_to_loss(
        self,
        output_embedding: List[float],
        violated_rules: List[SymbolicRule],
    ) -> float:
        """Compute symbolic constraint loss for gradient update.

        L_symbolic = α × Σᵣ wᵣ × hinge(satisfaction)

        hinge(s) = max(0, 1 - s)  — zero if satisfied, positive if violated

        Args:
            output_embedding: The neural model's output embedding
            violated_rules:   Rules whose constraints were not satisfied

        Returns:
            Scalar loss penalty (add to task loss during training)
        """
        if not violated_rules:
            return 0.0

        total = 0.0
        for rule in violated_rules:
            # Satisfaction score proxy: 0.0 (completely violated)
            # In a full implementation this would compare output_embedding
            # to the prototype of the consequent predicate
            satisfaction = 0.0
            hinge = max(0.0, 1.0 - satisfaction)
            total += rule.weight * hinge

        return self.alpha * total

    def symbolic_constraint_gradient(
        self,
        embedding: List[float],
        violated_rules: List[SymbolicRule],
        step_size: float = 0.01,
    ) -> List[float]:
        """Compute gradient of symbolic loss w.r.t. embedding.

        Used when backpropagation is available (PyTorch wrapper).
        Returns a gradient vector to add to the embedding's gradient.

        For violated rules, we push the embedding toward the
        consequent predicate's prototype — making the output
        more consistent with what the rule demands.
        """
        if not violated_rules:
            return [0.0] * len(embedding)

        gradient = [0.0] * len(embedding)

        for rule in violated_rules:
            for consequent in rule.consequents:
                # Find prototype for the consequent
                proto = self._find_prototype(consequent)
                if proto is None:
                    continue
                # Gradient: push embedding toward prototype
                for i in range(min(len(embedding), len(proto))):
                    gradient[i] += rule.weight * self.alpha * step_size * (proto[i] - embedding[i])

        return gradient

    def _find_prototype(self, predicate: Predicate) -> Optional[List[float]]:
        """Find the prototype vector for a predicate from the grounder."""
        for proto_obj in self.grounder._prototypes:
            if proto_obj.predicate.name == predicate.name:
                return proto_obj.prototype
        return None

    # ── Direction 3: NSIL (Neural–Symbolic Integrity Link) ───────

    def assess_integrity(
        self,
        grounded: List[GroundedSymbol],
        activated_rules: List[SymbolicRule],
        derived_facts: Set[Predicate],
        null_set: Optional[NullSet] = None,
        is_passthrough: bool = False,
        doubt_threshold: float = 0.6,
    ) -> IntegrityReport:
        """Compute the NSIL integrity report.

        Bridges the gap between neural grounding evidence and
        symbolic constraint expectations.  The returned
        ``IntegrityReport`` contains per-schema residuals and
        an overall integrity score.

        Parameters
        ----------
        grounded : list[GroundedSymbol]
            Grounding evidence from the symbol grounder.
        activated_rules : list[SymbolicRule]
            Rules that fired during symbolic reasoning.
        derived_facts : set[Predicate]
            Facts derived by forward chaining.
        null_set : NullSet, optional
            NSI null set (used for enrichment suggestions).
        is_passthrough : bool
            True when using PassthroughBackbone (symbolic-only).
        doubt_threshold : float
            ``NSIL_LOW_INTEGRITY`` flag fires below this value.

        Returns
        -------
        IntegrityReport
        """
        return compute_integrity_report(
            grounded=grounded,
            activated_rules=activated_rules,
            derived_facts=derived_facts,
            null_set=null_set,
            is_passthrough=is_passthrough,
            doubt_threshold=doubt_threshold,
        )
