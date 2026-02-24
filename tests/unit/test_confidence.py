"""
tests/unit/test_confidence.py
===============================
Tests for metacognition confidence computation.

Mathematical basis:
    C_factual   = 1 / (1 + A(X))
    C_reasoning = √(symbolic_conf × satisfied_ratio)
    C_boundary  = neural_conf × (1 − 0.3 × background_gap)
"""

import math
import pytest
from nesy.metacognition.confidence import (
    compute_factual,
    compute_reasoning,
    compute_boundary,
    build_confidence_report,
)
from nesy.core.types import (
    ConfidenceReport,
    LogicClause,
    LogicConnective,
    NullItem,
    NullSet,
    NullType,
    Predicate,
    PresentSet,
)


def _ps():
    return PresentSet(concepts={"fever"}, context_type="medical")


def _null_set(type1=0, type2=0, type3=0):
    items = []
    for i in range(type1):
        items.append(NullItem(f"t1_{i}", 0.5, NullType.TYPE1_EXPECTED, ["fever"]))
    for i in range(type2):
        items.append(NullItem(f"t2_{i}", 0.6, NullType.TYPE2_MEANINGFUL, ["fever"]))
    for i in range(type3):
        items.append(NullItem(f"t3_{i}", 0.8, NullType.TYPE3_CRITICAL, ["fever"], 2.0))
    return NullSet(items=items, present_set=_ps())


def _clause(satisfied: bool, weight: float = 0.9) -> LogicClause:
    return LogicClause(
        predicates=[Predicate("X", ())],
        connective=LogicConnective.AND,
        satisfied=satisfied,
        weight=weight,
    )


# ─── compute_factual ──────────────────────────────────────────


class TestComputeFactual:
    def test_no_nulls_returns_one(self):
        assert compute_factual(_null_set()) == pytest.approx(1.0)

    def test_type1_nulls_do_not_penalise(self):
        """Type1 has multiplier 0.0 → zero anomaly score."""
        ns = _null_set(type1=5)
        assert compute_factual(ns) == pytest.approx(1.0)

    def test_type2_reduces_score(self):
        ns = _null_set(type2=2)
        score = compute_factual(ns)
        assert 0.0 < score < 1.0

    def test_type3_reduces_score_more(self):
        """Type3 has higher multiplier → lower factual confidence."""
        ns2 = _null_set(type2=1)
        ns3 = _null_set(type3=1)
        assert compute_factual(ns3) < compute_factual(ns2)

    def test_formula_1_over_1_plus_A(self):
        """Verify C_factual = 1 / (1 + A(X)) exactly."""
        ns = _null_set(type2=1)
        A = ns.total_anomaly_score
        expected = 1.0 / (1.0 + A)
        assert compute_factual(ns) == pytest.approx(expected)


# ─── compute_reasoning ────────────────────────────────────────


class TestComputeReasoning:
    def test_all_satisfied_high_confidence(self):
        clauses = [_clause(True), _clause(True)]
        score = compute_reasoning(1.0, clauses)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_none_satisfied_zero(self):
        clauses = [_clause(False), _clause(False)]
        score = compute_reasoning(1.0, clauses)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_partial_satisfaction(self):
        clauses = [_clause(True), _clause(False)]
        score = compute_reasoning(1.0, clauses)
        # sqrt(1.0 × 0.5) ≈ 0.707
        assert score == pytest.approx(math.sqrt(0.5), abs=0.01)

    def test_empty_clauses_returns_symbolic(self):
        assert compute_reasoning(0.85, []) == 0.85

    def test_low_symbolic_confidence(self):
        clauses = [_clause(True)]
        score = compute_reasoning(0.25, clauses)
        assert score == pytest.approx(math.sqrt(0.25), abs=0.01)


# ─── compute_boundary ─────────────────────────────────────────


class TestComputeBoundary:
    def test_no_nulls(self):
        ns = _null_set()
        score = compute_boundary(0.90, ns)
        # When no items, total=1 (denominator guard), type1_count=0 → gap=0
        # boundary = 0.9 × (1 - 0.3 × 0) = 0.9
        assert score == pytest.approx(0.90, abs=0.01)

    def test_type1_nulls_reduce_boundary(self):
        """Type1 = 'background gap' → reduces boundary confidence."""
        ns = _null_set(type1=3)
        score = compute_boundary(0.90, ns)
        # gap = 3/3 = 1.0 → boundary = 0.9 × (1 - 0.3) = 0.63
        assert score == pytest.approx(0.63, abs=0.01)

    def test_neural_confidence_scales(self):
        ns = _null_set()
        high = compute_boundary(1.0, ns)
        low = compute_boundary(0.5, ns)
        assert high > low


# ─── build_confidence_report ──────────────────────────────────


class TestBuildConfidenceReport:
    def test_returns_confidence_report(self):
        ns = _null_set()
        report = build_confidence_report(
            null_set=ns,
            symbolic_confidence=0.9,
            neural_confidence=0.9,
            clauses=[_clause(True)],
        )
        assert isinstance(report, ConfidenceReport)
        assert 0.0 <= report.factual <= 1.0
        assert 0.0 <= report.reasoning <= 1.0
        assert 0.0 <= report.knowledge_boundary <= 1.0

    def test_minimum_tracks_weakest(self):
        ns = _null_set(type3=3)
        report = build_confidence_report(
            null_set=ns,
            symbolic_confidence=0.1,
            neural_confidence=0.9,
            clauses=[_clause(False)],
        )
        assert report.minimum == min(report.factual, report.reasoning, report.knowledge_boundary)
