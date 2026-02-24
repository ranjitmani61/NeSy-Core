"""
nesy/nsi/concept_graph.py
=========================
Concept Graph Engine (CGE) — the world model for Negative Space Intelligence.

Mathematical basis:
    G = (V, E, W) directed weighted graph

    V = concept nodes (strings)
    E = directed edges (source → target)
    W: E → [0,1]   edge weight formula:
        W(a→b) = P(b|a) × causal_strength × temporal_stability

    Expected Set E(X):
        E(X) = { c ∈ V | ∃e ∈ P(X) : W(e→c) > threshold }

    Null Set N(X):
        N(X) = E(X) − P(X)

The concept graph is built ONCE and queried at inference time.
Query time complexity: O(k) where k = average out-degree of concept nodes.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from nesy.core.types import ConceptEdge, NullItem, NullType, PresentSet, NullSet

logger = logging.getLogger(__name__)

# Context-specific thresholds:
# Medical reasoning requires high causal certainty → higher threshold
# General conversation allows looser expected associations → lower threshold
CONTEXT_THRESHOLDS: Dict[str, float] = {
    "medical": 0.35,
    "legal": 0.30,
    "code": 0.25,
    "general": 0.15,
}

# Criticality multipliers per domain-specific concept class
CRITICAL_CONCEPT_CLASSES: Dict[str, Set[str]] = {
    "medical": {
        "vital_signs",
        "medication",
        "allergy",
        "contraindication",
        "emergency_symptom",
        "diagnostic_test",
    },
    "code": {
        "null_check",
        "error_handling",
        "input_validation",
        "authentication",
        "authorization",
    },
}


class ConceptGraphEngine:
    """Build, store, and query the weighted concept graph.

    This is the persistent world-model: built once, queried many times.
    Supports incremental updates without full rebuild.
    """

    def __init__(self, domain: str = "general"):
        self.domain = domain
        # Adjacency: source → {target → ConceptEdge}
        self._graph: Dict[str, Dict[str, ConceptEdge]] = defaultdict(dict)
        self._concept_classes: Dict[str, str] = {}  # concept → class label
        self._total_concepts: int = 0

    # ─── GRAPH CONSTRUCTION ────────────────────────────────────────

    def add_edge(self, edge: ConceptEdge) -> None:
        """Add a single directed edge to the graph."""
        self._graph[edge.source][edge.target] = edge
        self._total_concepts = len(self._graph)

    def add_edges(self, edges: List[ConceptEdge]) -> None:
        for edge in edges:
            self.add_edge(edge)

    def register_concept_class(self, concept: str, class_label: str) -> None:
        """Register a concept as belonging to a critical class.

        Example: register_concept_class("blood_pressure", "vital_signs")
        This makes 'blood_pressure' a Type3 critical null in medical context.
        """
        self._concept_classes[concept] = class_label

    def update_edge(self, source: str, target: str, **kwargs) -> None:
        """Incrementally update edge properties without full rebuild."""
        if target in self._graph.get(source, {}):
            existing = self._graph[source][target]
            updated = ConceptEdge(
                source=source,
                target=target,
                cooccurrence_prob=kwargs.get("cooccurrence_prob", existing.cooccurrence_prob),
                causal_strength=kwargs.get("causal_strength", existing.causal_strength),
                temporal_stability=kwargs.get("temporal_stability", existing.temporal_stability),
            )
            self._graph[source][target] = updated

    # ─── GRAPH QUERIES ─────────────────────────────────────────────

    def get_neighbors(
        self,
        concept: str,
        threshold: float,
    ) -> List[Tuple[str, float]]:
        """Get all concepts reachable from 'concept' with weight > threshold.

        Returns list of (neighbor_concept, edge_weight) pairs,
        sorted by weight descending.
        """
        neighbors = []
        for target, edge in self._graph.get(concept, {}).items():
            if edge.weight > threshold:
                neighbors.append((target, edge.weight))
        return sorted(neighbors, key=lambda x: -x[1])

    def get_edge_weight(self, source: str, target: str) -> Optional[float]:
        edge = self._graph.get(source, {}).get(target)
        return edge.weight if edge else None

    # ─── NULL SET COMPUTATION ──────────────────────────────────────

    def compute_null_set(self, present_set: PresentSet) -> NullSet:
        """Core NSI computation: N(X) = E(X) − P(X).

        Algorithm:
            1. For each concept c in P(X):
               collect all neighbors above context threshold
            2. Merge into Expected Set E(X) with max weight
               (if multiple present concepts expect the same absent concept,
               take the maximum edge weight — strongest expectation wins)
            3. Subtract P(X) to get N(X)
            4. Classify each item in N(X) as Type 1/2/3

        Time: O(|P(X)| × k) where k = average out-degree
        """
        threshold = CONTEXT_THRESHOLDS.get(present_set.context_type, 0.15)

        # Step 1 & 2: Build Expected Set with max weights
        expected: Dict[str, Dict] = {}  # concept → {weight, triggered_by}

        for concept in present_set.concepts:
            neighbors = self.get_neighbors(concept, threshold)
            for neighbor, weight in neighbors:
                if neighbor not in present_set.concepts:
                    if neighbor not in expected or weight > expected[neighbor]["weight"]:
                        expected[neighbor] = {
                            "weight": weight,
                            "triggered_by": [concept],
                        }
                    else:
                        expected[neighbor]["triggered_by"].append(concept)

        # Step 3 & 4: Subtract P(X) and classify
        null_items = []
        for concept, info in expected.items():
            null_type = self._classify_null(
                concept=concept,
                weight=info["weight"],
                context_type=present_set.context_type,
                present_concepts=present_set.concepts,
            )
            criticality = self._get_criticality(concept, present_set.context_type)

            null_items.append(
                NullItem(
                    concept=concept,
                    weight=info["weight"],
                    null_type=null_type,
                    expected_because_of=info["triggered_by"],
                    criticality=criticality,
                )
            )

        # Sort: Type3 first, then Type2, then Type1
        null_items.sort(key=lambda x: (-x.null_type.value, -x.weight))

        return NullSet(items=null_items, present_set=present_set)

    def _classify_null(
        self,
        concept: str,
        weight: float,
        context_type: str,
        present_concepts: Set[str],
    ) -> NullType:
        """Classify an absent concept as Type1 / Type2 / Type3.

        Classification logic:

        Type3 (Critical): The absent concept belongs to a critical class
                          for the current domain. Its absence MUST be addressed.
                          e.g., missing vital signs in a medical context.

        Type2 (Meaningful): High edge weight (≥ 0.6 × threshold × 10)
                            but not in a critical class. Should be investigated.

        Type1 (Expected):  Low weight OR ubiquitous background concept.
                          Normal absence — do not flag.
        """
        concept_class = self._concept_classes.get(concept)
        critical_classes = CRITICAL_CONCEPT_CLASSES.get(context_type, set())

        if concept_class in critical_classes:
            return NullType.TYPE3_CRITICAL

        if weight >= 0.50:
            return NullType.TYPE2_MEANINGFUL

        return NullType.TYPE1_EXPECTED

    def _get_criticality(self, concept: str, context_type: str) -> float:
        """Domain-specific criticality multiplier for anomaly score."""
        concept_class = self._concept_classes.get(concept)
        critical_classes = CRITICAL_CONCEPT_CLASSES.get(context_type, set())
        if concept_class in critical_classes:
            return 2.0
        return 1.0

    # ─── PERSISTENCE ───────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist graph to JSON for reuse across sessions."""
        data = {
            "domain": self.domain,
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "cooccurrence_prob": e.cooccurrence_prob,
                    "causal_strength": e.causal_strength,
                    "temporal_stability": e.temporal_stability,
                }
                for source_dict in self._graph.values()
                for e in source_dict.values()
            ],
            "concept_classes": self._concept_classes,
        }
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info(f"Concept graph saved: {len(data['edges'])} edges → {path}")

    @classmethod
    def load(cls, path: str) -> "ConceptGraphEngine":
        """Load a previously saved concept graph."""
        data = json.loads(Path(path).read_text())
        cge = cls(domain=data["domain"])
        for ed in data["edges"]:
            cge.add_edge(
                ConceptEdge(
                    source=ed["source"],
                    target=ed["target"],
                    cooccurrence_prob=ed["cooccurrence_prob"],
                    causal_strength=ed["causal_strength"],
                    temporal_stability=ed["temporal_stability"],
                )
            )
        cge._concept_classes = data.get("concept_classes", {})
        logger.info(f"Concept graph loaded: {len(data['edges'])} edges from {path}")
        return cge

    # ─── DIAGNOSTICS ───────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        total_edges = sum(len(v) for v in self._graph.values())
        return {
            "concepts": len(self._graph),
            "edges": total_edges,
            "domain": self.domain,
            "avg_out_degree": total_edges / max(len(self._graph), 1),
        }
