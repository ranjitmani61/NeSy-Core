"""
tests/unit/test_betti_full.py
=============================
Comprehensive tests for nesy/symbolic/betti.py — BettiAnalyser.

The ``betti_0`` function itself lives in ``nesy.symbolic.logic`` and is
re-exported.  This file tests the higher-level ``BettiAnalyser`` class
which provides components, coherence_score, diagnose, and from_trace.

Target: 100% line coverage for nesy/symbolic/betti.py.
"""

from nesy.core.types import Predicate, ReasoningStep
from nesy.symbolic.betti import BettiAnalyser, betti_0


# ═══════════════════════════════════════════════════════════════════
#  Re-exported betti_0
# ═══════════════════════════════════════════════════════════════════


class TestBetti0ReExport:
    """Verify the re-exported betti_0 behaves identically to logic.betti_0."""

    def test_empty(self):
        assert betti_0([]) == 0

    def test_single_predicate(self):
        assert betti_0([Predicate("A", ("x",))]) == 1

    def test_connected(self):
        """Two predicates sharing ground arg → β₀ = 1."""
        preds = [
            Predicate("HasSymptom", ("p1", "fever")),
            Predicate("RequiresTest", ("p1", "blood_test")),
        ]
        assert betti_0(preds) == 1

    def test_disconnected(self):
        """Two predicates with no shared ground args → β₀ = 2."""
        preds = [
            Predicate("HasSymptom", ("p1", "fever")),
            Predicate("HasSymptom", ("p2", "cough")),
        ]
        assert betti_0(preds) == 2

    def test_variables_do_not_connect(self):
        """Arguments starting with '?' are variables and should NOT connect."""
        preds = [
            Predicate("A", ("?x", "foo")),
            Predicate("B", ("?x", "bar")),
        ]
        # ?x is variable → not ground → does not connect A and B
        # "foo" only in A, "bar" only in B → β₀ = 2
        assert betti_0(preds) == 2

    def test_three_connected_chain(self):
        """A-B-C all connected through shared args → β₀ = 1."""
        preds = [
            Predicate("Step1", ("a", "b")),
            Predicate("Step2", ("b", "c")),
            Predicate("Step3", ("c", "d")),
        ]
        assert betti_0(preds) == 1

    def test_mixed_connected_disconnected(self):
        """Two connected clusters → β₀ = 2."""
        preds = [
            Predicate("Step1", ("a", "b")),
            Predicate("Step2", ("b", "c")),
            Predicate("Unrelated", ("x", "y")),  # different cluster
        ]
        assert betti_0(preds) == 2


# ═══════════════════════════════════════════════════════════════════
#  BettiAnalyser.compute
# ═══════════════════════════════════════════════════════════════════


class TestBettiCompute:
    def test_compute_empty(self):
        assert BettiAnalyser.compute([]) == 0

    def test_compute_single(self):
        assert BettiAnalyser.compute([Predicate("A", ("x",))]) == 1

    def test_compute_connected(self):
        preds = [
            Predicate("A", ("x",)),
            Predicate("B", ("x",)),
        ]
        assert BettiAnalyser.compute(preds) == 1


# ═══════════════════════════════════════════════════════════════════
#  BettiAnalyser.components
# ═══════════════════════════════════════════════════════════════════


class TestBettiComponents:
    def test_empty(self):
        assert BettiAnalyser.components([]) == []

    def test_single_component(self):
        preds = [
            Predicate("A", ("x",)),
            Predicate("B", ("x",)),
        ]
        comps = BettiAnalyser.components(preds)
        assert len(comps) == 1
        assert len(comps[0]) == 2

    def test_two_components(self):
        preds = [
            Predicate("A", ("x",)),
            Predicate("B", ("y",)),
        ]
        comps = BettiAnalyser.components(preds)
        assert len(comps) == 2

    def test_three_components(self):
        preds = [
            Predicate("A", ("x",)),
            Predicate("B", ("y",)),
            Predicate("C", ("z",)),
        ]
        comps = BettiAnalyser.components(preds)
        assert len(comps) == 3

    def test_components_content(self):
        p1 = Predicate("A", ("x",))
        p2 = Predicate("B", ("x",))
        p3 = Predicate("C", ("y",))
        comps = BettiAnalyser.components([p1, p2, p3])
        assert len(comps) == 2
        # p1 and p2 should be in the same component
        found = False
        for comp in comps:
            if p1 in comp and p2 in comp:
                found = True
        assert found

    def test_variables_do_not_connect_components(self):
        """Variables ('?x') should not create connections."""
        preds = [
            Predicate("A", ("?x", "foo")),
            Predicate("B", ("?x", "bar")),
        ]
        comps = BettiAnalyser.components(preds)
        assert len(comps) == 2

    def test_chain_is_single_component(self):
        """A chain a→b→c→d is one component."""
        preds = [
            Predicate("P1", ("a", "b")),
            Predicate("P2", ("b", "c")),
            Predicate("P3", ("c", "d")),
        ]
        comps = BettiAnalyser.components(preds)
        assert len(comps) == 1
        assert len(comps[0]) == 3


# ═══════════════════════════════════════════════════════════════════
#  BettiAnalyser.coherence_score
# ═══════════════════════════════════════════════════════════════════


class TestBettiCoherence:
    def test_coherence_empty(self):
        """β₀ = 0 → coherence = 1.0 (vacuously coherent)."""
        assert BettiAnalyser.coherence_score([]) == 1.0

    def test_coherence_single_component(self):
        """β₀ = 1 → coherence = 1.0."""
        preds = [
            Predicate("A", ("x",)),
            Predicate("B", ("x",)),
        ]
        assert BettiAnalyser.coherence_score(preds) == 1.0

    def test_coherence_two_components(self):
        """β₀ = 2 → coherence = 0.5."""
        preds = [
            Predicate("A", ("x",)),
            Predicate("B", ("y",)),
        ]
        assert abs(BettiAnalyser.coherence_score(preds) - 0.5) < 1e-10

    def test_coherence_three_components(self):
        """β₀ = 3 → coherence = 1/3."""
        preds = [
            Predicate("A", ("x",)),
            Predicate("B", ("y",)),
            Predicate("C", ("z",)),
        ]
        assert abs(BettiAnalyser.coherence_score(preds) - 1.0 / 3.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════
#  BettiAnalyser.diagnose
# ═══════════════════════════════════════════════════════════════════


class TestBettiDiagnose:
    def test_diagnose_empty(self):
        result = BettiAnalyser.diagnose([])
        assert "No predicates" in result

    def test_diagnose_single_component(self):
        preds = [
            Predicate("A", ("x",)),
            Predicate("B", ("x",)),
        ]
        result = BettiAnalyser.diagnose(preds)
        assert "β₀ = 1" in result
        assert "coherent" in result.lower()

    def test_diagnose_disconnected(self):
        preds = [
            Predicate("A", ("x",)),
            Predicate("B", ("y",)),
        ]
        result = BettiAnalyser.diagnose(preds)
        assert "β₀ = 2" in result
        assert "disconnected" in result.lower() or "WARNING" in result

    def test_diagnose_three_disconnected_has_components(self):
        preds = [
            Predicate("A", ("x",)),
            Predicate("B", ("y",)),
            Predicate("C", ("z",)),
        ]
        result = BettiAnalyser.diagnose(preds)
        assert "β₀ = 3" in result
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result
        assert "WARNING" in result


# ═══════════════════════════════════════════════════════════════════
#  BettiAnalyser.from_trace
# ═══════════════════════════════════════════════════════════════════


class TestBettiFromTrace:
    def test_from_trace_empty(self):
        b0, score, diag = BettiAnalyser.from_trace([])
        assert b0 == 0
        assert score == 1.0
        assert "No predicates" in diag

    def test_from_trace_single_step(self):
        steps = [
            ReasoningStep(
                step_number=0,
                description="Initial facts",
                rule_applied=None,
                predicates=[Predicate("A", ("x",))],
                confidence=1.0,
            )
        ]
        b0, score, diag = BettiAnalyser.from_trace(steps)
        assert b0 == 1
        assert score == 1.0
        assert "β₀ = 1" in diag

    def test_from_trace_connected_steps(self):
        steps = [
            ReasoningStep(
                step_number=0,
                description="Step 0",
                rule_applied=None,
                predicates=[Predicate("A", ("x",))],
                confidence=1.0,
            ),
            ReasoningStep(
                step_number=1,
                description="Step 1",
                rule_applied="rule_1",
                predicates=[Predicate("B", ("x",))],
                confidence=0.9,
            ),
        ]
        b0, score, diag = BettiAnalyser.from_trace(steps)
        assert b0 == 1
        assert score == 1.0

    def test_from_trace_disconnected_steps(self):
        steps = [
            ReasoningStep(
                step_number=0,
                description="Step 0",
                rule_applied=None,
                predicates=[Predicate("A", ("x",))],
                confidence=1.0,
            ),
            ReasoningStep(
                step_number=1,
                description="Step 1",
                rule_applied="rule_1",
                predicates=[Predicate("B", ("y",))],
                confidence=0.9,
            ),
        ]
        b0, score, diag = BettiAnalyser.from_trace(steps)
        assert b0 == 2
        assert abs(score - 0.5) < 1e-10
        assert "WARNING" in diag


# ═══════════════════════════════════════════════════════════════════
#  Union-Find edge cases (covers betti.py lines 85, 88)
# ═══════════════════════════════════════════════════════════════════


class TestBettiUnionFindEdgeCases:
    def test_already_same_component_returns_early(self):
        """Two predicates sharing 2 args → union is called twice for same pair.
        Second call hits the 'ra == rb: return' branch (line 85)."""
        preds = [
            Predicate("A", ("x", "y")),
            Predicate("B", ("x", "y")),
        ]
        comps = BettiAnalyser.components(preds)
        assert len(comps) == 1

    def test_rank_swap_in_union(self):
        """Force the rank[ra] < rank[rb] swap path (line 88).

        Construct predicates whose union-find processing creates a tall
        tree, then attempts to merge a singleton into it from the
        LEFT side of union(a, b) where find(a).rank < find(b).rank.

        Predicates:
            P0: (a, b)   idx 0
            P1: (a,)     idx 1
            P2: (c,)     idx 2
            P3: (c, b)   idx 3

        arg_to_indices: a→[0,1], b→[0,3], c→[2,3]
        Step 1: union(0,1) via 'a' → rank[0]=1
        Step 2: union(0,3) via 'b' → rank[0]>rank[3], no swap
        Step 3: union(2,3) via 'c' → find(2)=2(rank 0), find(3)=0(rank 1)
                rank[2] < rank[0] → SWAP → parent[2]=0
        """
        preds = [
            Predicate("P0", ("a", "b")),
            Predicate("P1", ("a",)),
            Predicate("P2", ("c",)),
            Predicate("P3", ("c", "b")),
        ]
        comps = BettiAnalyser.components(preds)
        assert len(comps) == 1
        assert len(comps[0]) == 4
