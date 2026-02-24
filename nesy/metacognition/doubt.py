"""
nesy/metacognition/doubt.py
============================
SelfDoubtLayer — decides when the system should hold output.

The key innovation: the system has a computable definition of
"I don't know enough to answer this reliably."

Three doubt triggers:
    1. Confidence below threshold (soft doubt)
    2. Critical null violations present (factual doubt)  
    3. Reasoning chain has disconnected components β₀ > 1 (logical doubt)
"""
from __future__ import annotations
from typing import List, Tuple
from nesy.core.types import ConfidenceReport, NullSet, OutputStatus


class SelfDoubtLayer:
    """Determines whether the system should doubt its own output.
    
    This is the 'epistemic humility' module — the system knows
    when it doesn't know.
    """

    def __init__(self, threshold: float = 0.60, betti_warn: int = 2):
        self.threshold  = threshold
        self.betti_warn = betti_warn

    def evaluate(
        self,
        confidence: ConfidenceReport,
        null_set:   NullSet,
        betti_0:    int = 1,
    ) -> Tuple[OutputStatus, List[str]]:
        """Evaluate whether to doubt the output.
        
        Returns (status, flags):
            REJECTED:  Critical nulls present — output cannot be trusted
            FLAGGED:   Min confidence < threshold OR β₀ > warn level
            UNCERTAIN: Confidence borderline (threshold to 0.75)
            OK:        Confidence ≥ 0.75 and no violations
        """
        flags = []

        # Trigger 1: Critical null violations
        if null_set.critical_items:
            for item in null_set.critical_items:
                flags.append(
                    f"CRITICAL DOUBT: '{item.concept}' is absent but causally required. "
                    f"Output cannot be trusted without this information."
                )
            return OutputStatus.REJECTED, flags

        # Trigger 2: Topological doubt
        if betti_0 >= self.betti_warn:
            flags.append(
                f"TOPOLOGICAL DOUBT: Reasoning has {betti_0} disconnected components. "
                f"Logic chain is not unified — possible hidden information."
            )

        # Trigger 3: Confidence threshold
        min_conf = confidence.minimum
        if min_conf < self.threshold:
            weakest = min(
                [("factual", confidence.factual),
                 ("reasoning", confidence.reasoning),
                 ("knowledge_boundary", confidence.knowledge_boundary)],
                key=lambda x: x[1]
            )
            flags.append(
                f"SELF-DOUBT: Minimum confidence {min_conf:.3f} < {self.threshold:.3f}. "
                f"Weakest dimension: {weakest[0]} = {weakest[1]:.3f}"
            )
            return OutputStatus.FLAGGED, flags

        if min_conf < 0.75:
            return OutputStatus.UNCERTAIN, flags

        return OutputStatus.OK, flags
