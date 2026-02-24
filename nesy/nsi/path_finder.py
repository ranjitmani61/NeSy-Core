"""
nesy/nsi/path_finder.py
========================
ConceptPathFinder — find paths between concepts in the concept graph.

Used for:
    1. EXPLANATION: "Why is concept B expected given concept A?"
       → find the shortest weighted path from A to B
    2. COUNTERFACTUAL: "What concepts would need to be present to avoid null X?"
       → find which present concepts trigger the null
    3. INFLUENCE CHAINS: "What does concept A transitively affect?"
       → all concepts reachable from A within k hops

Algorithm: Modified Dijkstra on the weighted directed graph.
    Edge weight as distance = 1 - W(a→b)
    (higher weight = shorter distance = stronger connection)

Shortest path in this space = strongest causal/semantic connection.

Mathematical basis:
    Distance d(a,b) = 1 - W(a→b)
    Dijkstra minimises Σ d(aᵢ, aᵢ₊₁) along a path
    = maximises Σ W(aᵢ, aᵢ₊₁) (strongest path)
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ConceptPath:
    """A path through the concept graph."""

    source: str
    target: str
    nodes: List[str]  # [source, intermediate..., target]
    weights: List[float]  # edge weights along path
    total_weight: float  # sum of edge weights (strength)
    hops: int  # len(nodes) - 1

    def explain(self) -> str:
        if not self.nodes:
            return f"No path from {self.source} to {self.target}"
        parts = []
        for i in range(len(self.nodes) - 1):
            w = self.weights[i] if i < len(self.weights) else 0.0
            parts.append(f"{self.nodes[i]} →[{w:.2f}]→ {self.nodes[i + 1]}")
        return " ".join(parts)


class ConceptPathFinder:
    """Find paths between concepts using weighted Dijkstra.

    Usage:
        finder = ConceptPathFinder(graph._graph)

        path = finder.shortest_path("fever", "blood_test")
        print(path.explain())

        reachable = finder.reachable_from("fever", max_hops=3)
    """

    def __init__(self, graph: Dict[str, Dict[str, any]]):
        """
        Args:
            graph: ConceptGraphEngine._graph (adjacency dict)
        """
        self._graph = graph

    def shortest_path(
        self,
        source: str,
        target: str,
        max_hops: int = 10,
    ) -> Optional[ConceptPath]:
        """Find strongest-connection path from source to target.

        Uses Dijkstra with distance = 1 - W(a→b).

        Returns None if target is unreachable within max_hops.
        """
        if source == target:
            return ConceptPath(
                source=source, target=target, nodes=[source], weights=[], total_weight=0.0, hops=0
            )

        if source not in self._graph:
            return None

        # Dijkstra: priority queue of (distance, node, path, weights)
        heap = [(0.0, source, [source], [])]
        visited: Set[str] = set()

        while heap:
            dist, node, path, edge_weights = heapq.heappop(heap)

            if node in visited:
                continue
            visited.add(node)

            if len(path) - 1 > max_hops:
                continue

            for neighbor, edge in self._graph.get(node, {}).items():
                if neighbor in visited:
                    continue
                new_dist = dist + (1.0 - edge.weight)
                new_path = path + [neighbor]
                new_weights = edge_weights + [edge.weight]

                if neighbor == target:
                    total_w = sum(new_weights)
                    return ConceptPath(
                        source=source,
                        target=target,
                        nodes=new_path,
                        weights=new_weights,
                        total_weight=total_w,
                        hops=len(new_path) - 1,
                    )

                heapq.heappush(heap, (new_dist, neighbor, new_path, new_weights))

        return None  # unreachable

    def all_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 5,
        min_weight: float = 0.10,
    ) -> List[ConceptPath]:
        """Find ALL paths from source to target (up to max_hops).

        Returns paths sorted by total_weight descending.
        Useful for explanation: "there are 3 reasons why B is expected".
        """
        results = []
        self._dfs(
            current=source,
            target=target,
            path=[source],
            weights=[],
            visited={source},
            max_hops=max_hops,
            min_weight=min_weight,
            results=results,
        )
        results.sort(key=lambda p: -p.total_weight)
        return results

    def _dfs(
        self,
        current: str,
        target: str,
        path: List[str],
        weights: List[float],
        visited: Set[str],
        max_hops: int,
        min_weight: float,
        results: List[ConceptPath],
    ) -> None:
        if len(path) - 1 >= max_hops:
            return
        for neighbor, edge in self._graph.get(current, {}).items():
            if neighbor in visited:
                continue
            if edge.weight < min_weight:
                continue
            new_path = path + [neighbor]
            new_weights = weights + [edge.weight]
            if neighbor == target:
                results.append(
                    ConceptPath(
                        source=path[0],
                        target=target,
                        nodes=new_path,
                        weights=new_weights,
                        total_weight=sum(new_weights),
                        hops=len(new_path) - 1,
                    )
                )
            else:
                self._dfs(
                    neighbor,
                    target,
                    new_path,
                    new_weights,
                    visited | {neighbor},
                    max_hops,
                    min_weight,
                    results,
                )

    def reachable_from(
        self,
        source: str,
        max_hops: int = 3,
        min_weight: float = 0.10,
    ) -> Dict[str, float]:
        """Find all concepts reachable from source within max_hops.

        Returns dict: concept → best (max) path weight to reach it.
        Used for null set explanation: "fever transitively expects ECG via BP".
        """
        reachable: Dict[str, float] = {}
        heap = [(0.0, source, 0)]  # (neg_weight, node, hops) — negate for max-heap
        visited: Set[str] = set()

        while heap:
            neg_w, node, hops = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            if node != source:
                reachable[node] = -neg_w

            if hops >= max_hops:
                continue

            for neighbor, edge in self._graph.get(node, {}).items():
                if neighbor in visited:
                    continue
                if edge.weight < min_weight:
                    continue
                # Accumulate weight as product along path
                path_weight = (-neg_w) * edge.weight if node != source else edge.weight
                heapq.heappush(heap, (-path_weight, neighbor, hops + 1))

        return reachable

    def explain_null(
        self,
        absent_concept: str,
        present_concepts: Set[str],
    ) -> str:
        """Explain WHY an absent concept is in the null set.

        Finds the strongest path from any present concept to the absent one.
        """
        best_path = None
        best_total = -1.0

        for pc in present_concepts:
            path = self.shortest_path(pc, absent_concept)
            if path and path.total_weight > best_total:
                best_total = path.total_weight
                best_path = path

        if best_path is None:
            return f"'{absent_concept}' is expected but no direct path found."

        return (
            f"'{absent_concept}' is expected because: {best_path.explain()}\n"
            f"(path strength: {best_total:.3f})"
        )
