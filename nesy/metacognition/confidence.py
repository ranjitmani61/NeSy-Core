"""
nesy/metacognition/confidence.py
=================================
Three-dimensional confidence computation — independent of monitor.py.
Can be used standalone for confidence-only queries.

Dimensions:
    Factual:            1 / (1 + anomaly_score_from_null_set)
    Reasoning:          geometric_mean(rule_weights_in_chain)
    KnowledgeBoundary:  neural_confidence × (1 - background_gap_penalty)
"""

from __future__ import annotations
import math
from typing import List
from nesy.core.types import ConfidenceReport, LogicClause, NullSet, NullType


def compute_factual(null_set: NullSet) -> float:
    """C_factual = 1 / (1 + A(X))
    A(X) = total anomaly score from null set.
    """
    return 1.0 / (1.0 + null_set.total_anomaly_score)


def compute_reasoning(
    symbolic_confidence: float,
    clauses: List[LogicClause],
) -> float:
    """C_reasoning = sqrt(symbolic_confidence × satisfied_ratio)"""
    if not clauses:
        return symbolic_confidence
    satisfied = sum(1 for c in clauses if c.satisfied)
    ratio = satisfied / len(clauses)
    return math.sqrt(max(0.0, symbolic_confidence * ratio))


def compute_boundary(
    neural_confidence: float,
    null_set: NullSet,
) -> float:
    """C_boundary = neural × (1 - 0.3 × background_gap)"""
    total = len(null_set.items) if null_set.items else 1
    type1_count = sum(1 for i in null_set.items if i.null_type == NullType.TYPE1_EXPECTED)
    gap = type1_count / total
    return neural_confidence * (1.0 - 0.3 * gap)


def build_confidence_report(
    null_set: NullSet,
    symbolic_confidence: float,
    neural_confidence: float,
    clauses: List[LogicClause],
) -> ConfidenceReport:
    """Build a complete ConfidenceReport from all components."""
    f = compute_factual(null_set)
    r = compute_reasoning(symbolic_confidence, clauses)
    b = compute_boundary(neural_confidence, null_set)
    return ConfidenceReport(
        factual=f,
        reasoning=r,
        knowledge_boundary=b,
    )
