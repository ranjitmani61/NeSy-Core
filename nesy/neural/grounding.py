"""
nesy/neural/grounding.py
========================
Symbol Grounding — the bridge from continuous neural space to
discrete symbolic predicates.

Mathematical basis:
    Grounding function G: ℝᵈ → P(Predicates)
    
    For each candidate predicate p with prototype vector μₚ:
        similarity(e, p) = cosine(e, μₚ) = (e · μₚ) / (‖e‖ × ‖μₚ‖)
    
    Grounded predicates = { p | similarity(e, p) > θ_ground }
    Grounding confidence = max similarity over matched predicates
    
    This is the Smolensky-Elman grounding approach:
    symbolic representations emerge from sub-symbolic vectors
    when similarity crosses a threshold.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nesy.core.types import GroundedSymbol, Predicate

logger = logging.getLogger(__name__)


@dataclass
class PredicatePrototype:
    """A predicate with its prototype embedding vector.
    
    The prototype is the 'canonical' embedding for this predicate —
    built by averaging embeddings of known instances during training.
    """
    predicate:  Predicate
    prototype:  List[float]     # canonical embedding vector μₚ
    domain:     Optional[str] = None


class SymbolGrounder:
    """Ground neural embeddings to symbolic predicates.
    
    Usage:
        grounder = SymbolGrounder(threshold=0.75)
        grounder.register(PredicatePrototype(
            predicate=Predicate("HasSymptom", ("?p", "fever")),
            prototype=[0.1, 0.8, 0.3, ...],
            domain="medical",
        ))
        grounded = grounder.ground(embedding)
    """

    def __init__(self, threshold: float = 0.72):
        """
        Args:
            threshold: Minimum cosine similarity for grounding.
                       Below this → no predicate is assigned.
                       Typical range: 0.65 – 0.85
        """
        assert 0.0 < threshold < 1.0
        self.threshold = threshold
        self._prototypes: List[PredicatePrototype] = []

    def register(self, proto: PredicatePrototype) -> None:
        """Register a predicate prototype."""
        self._prototypes.append(proto)

    def register_batch(self, protos: List[PredicatePrototype]) -> None:
        for p in protos:
            self.register(p)

    def ground(
        self,
        embedding: List[float],
        domain: Optional[str] = None,
        top_k: int = 5,
    ) -> List[GroundedSymbol]:
        """Ground an embedding vector to its most similar predicates.
        
        Args:
            embedding: Neural embedding vector ∈ ℝᵈ
            domain:    If set, only consider predicates from this domain
            top_k:     Return at most k grounded symbols
        
        Returns:
            List of GroundedSymbol, sorted by grounding_confidence descending.
            Empty list if no predicate exceeds the threshold.
        """
        candidates = self._prototypes
        if domain:
            candidates = [p for p in candidates if p.domain is None or p.domain == domain]

        scored: List[Tuple[float, PredicatePrototype]] = []
        for proto in candidates:
            sim = self._cosine_similarity(embedding, proto.prototype)
            if sim >= self.threshold:
                scored.append((sim, proto))

        scored.sort(key=lambda x: -x[0])
        scored = scored[:top_k]

        return [
            GroundedSymbol(
                predicate=proto.predicate,
                embedding=embedding,
                grounding_confidence=sim,
            )
            for sim, proto in scored
        ]

    def ground_with_confidence(self, embedding: List[float]) -> Tuple[List[GroundedSymbol], float]:
        """Ground and return overall grounding confidence.
        
        Overall confidence = max similarity across all matched predicates.
        Returns 0.0 if no predicate matched.
        """
        grounded = self.ground(embedding)
        if not grounded:
            return [], 0.0
        best = max(g.grounding_confidence for g in grounded)
        return grounded, best

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity: (a · b) / (‖a‖ × ‖b‖)
        
        Returns 0.0 for zero vectors.
        """
        if len(a) != len(b):
            raise ValueError(f"Embedding dimension mismatch: {len(a)} vs {len(b)}")
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)

    def build_prototype_from_examples(
        self,
        predicate: Predicate,
        example_embeddings: List[List[float]],
        domain: Optional[str] = None,
    ) -> PredicatePrototype:
        """Build a prototype by averaging example embeddings.
        
        μₚ = (1/n) Σᵢ eᵢ  then L2-normalise
        """
        if not example_embeddings:
            raise ValueError("Cannot build prototype from empty examples")
        dim = len(example_embeddings[0])
        mean = [
            sum(e[i] for e in example_embeddings) / len(example_embeddings)
            for i in range(dim)
        ]
        # L2 normalise
        norm = math.sqrt(sum(x * x for x in mean))
        if norm > 1e-10:
            mean = [x / norm for x in mean]

        proto = PredicatePrototype(predicate=predicate, prototype=mean, domain=domain)
        self.register(proto)
        return proto

    @property
    def prototype_count(self) -> int:
        return len(self._prototypes)
