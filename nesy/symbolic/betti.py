"""
nesy/symbolic/betti.py
======================
Topological analysis of reasoning chains via Betti numbers.

Mathematical basis:
    β₀ = number of connected components in reasoning graph.

    Two predicates P₁, P₂ are connected if they share at least
    one ground argument (non-variable term).

    β₀ = 1 → all derivations form one connected chain → coherent
    β₀ > 1 → multiple disconnected reasoning chains → warning

    The Betti number is computed via Union-Find over a graph
    whose vertices are predicates and edges connect predicates
    sharing at least one ground argument.

    Reference: Edelsbrunner & Harer (2010) "Computational Topology".

The core ``betti_0`` function lives in ``nesy.symbolic.logic`` to avoid
circular imports.  This module re-exports it AND provides a higher-level
``BettiAnalyser`` for richer topological diagnostics.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from nesy.core.types import Predicate, ReasoningStep
from nesy.symbolic.logic import betti_0  # canonical implementation

logger = logging.getLogger(__name__)

__all__ = ["betti_0", "BettiAnalyser"]


class BettiAnalyser:
    """Higher-level topological analysis of reasoning traces.

    Provides:
        - β₀ computation (delegates to ``betti_0``)
        - Connected component extraction
        - Coherence scoring
        - Human-readable diagnostics
    """

    # ─── PUBLIC API ────────────────────────────────────────────────

    @staticmethod
    def compute(predicates: List[Predicate]) -> int:
        """Compute β₀ for a list of predicates.

        Returns 0 for empty input, otherwise the number of
        connected components (≥ 1).
        """
        return betti_0(predicates)

    @staticmethod
    def components(predicates: List[Predicate]) -> List[Set[Predicate]]:
        """Return each connected component as a set of predicates.

        Two predicates are in the same component iff they share
        at least one ground argument.

        Time complexity: effectively O(n·α(n)) via Union-Find.
        """
        if not predicates:
            return []

        parent: Dict[int, int] = {i: i for i in range(len(predicates))}
        rank: Dict[int, int] = {i: 0 for i in range(len(predicates))}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            # union by rank
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        # Build argument → indices mapping (ground terms only)
        arg_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, p in enumerate(predicates):
            for arg in p.args:
                if not arg.startswith("?"):
                    arg_to_indices[arg].append(idx)

        for indices in arg_to_indices.values():
            for i in range(len(indices) - 1):
                union(indices[i], indices[i + 1])

        # Group by root
        groups: Dict[int, Set[Predicate]] = defaultdict(set)
        for idx, p in enumerate(predicates):
            groups[find(idx)].add(p)

        return list(groups.values())

    @staticmethod
    def coherence_score(predicates: List[Predicate]) -> float:
        """Coherence ∈ (0, 1].  1.0 = single connected component.

        Formula:
            coherence = 1 / β₀

        This is the multiplicative penalty applied by ``SymbolicEngine``
        to symbolic confidence when β₀ > 1.
        """
        b0 = betti_0(predicates)
        if b0 == 0:
            return 1.0
        return 1.0 / b0

    @classmethod
    def diagnose(cls, predicates: List[Predicate]) -> str:
        """Return a human-readable diagnostic string."""
        b0 = betti_0(predicates)
        if b0 == 0:
            return "No predicates to analyse."
        if b0 == 1:
            return (
                f"β₀ = 1 — all {len(predicates)} predicates form "
                "a single connected reasoning chain. Topologically coherent."
            )
        comps = cls.components(predicates)
        lines = [
            f"β₀ = {b0} — reasoning has {b0} disconnected components "
            f"across {len(predicates)} predicates.",
            "Components:",
        ]
        for i, comp in enumerate(comps, 1):
            preds_str = ", ".join(str(p) for p in sorted(comp, key=str))
            lines.append(f"  [{i}] {preds_str}")
        lines.append(
            "WARNING: disconnected components indicate possible hidden "
            "information or deceptive input (NSI principle)."
        )
        return "\n".join(lines)

    @classmethod
    def from_trace(cls, steps: List[ReasoningStep]) -> Tuple[int, float, str]:
        """Convenience: analyse a full reasoning trace.

        Returns:
            (beta_0, coherence_score, diagnostic_text)
        """
        all_preds: List[Predicate] = []
        for step in steps:
            all_preds.extend(step.predicates)

        b0 = betti_0(all_preds)
        score = cls.coherence_score(all_preds)
        diag = cls.diagnose(all_preds)
        return b0, score, diag
