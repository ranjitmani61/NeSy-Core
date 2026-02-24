"""
nesy/metacognition/boundary.py
================================
KnowledgeBoundaryEstimator — estimate whether a query is within
the model's competence (in-distribution vs. out-of-distribution).

Mathematical basis:
    K-Nearest Neighbours distance in embedding space:
    
    C_boundary(q) = 1 / (1 + d_k(q, D_train))
    
    Where:
        d_k(q, D_train) = mean distance to k nearest training examples
        d(a, b)         = 1 - cosine_similarity(a, b)
    
    In-distribution:  d_k ≈ 0 → C_boundary ≈ 1.0
    Out-of-distribution: d_k ≈ 1 → C_boundary ≈ 0.5
    
    Threshold: if C_boundary < boundary_threshold, the query is
    flagged as out-of-distribution — the system should not be trusted.

This is the rigorous implementation of what monitor.py approximates
with the neural_confidence proxy. Use this when:
    - You have a training dataset of embeddings
    - The neural backbone produces embeddings
    - You want to detect distribution shift

If you don't have training embeddings, use the neural_confidence
proxy in monitor.py (default behavior).
"""
from __future__ import annotations

import heapq
import logging
import math
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class KNNBoundaryEstimator:
    """K-Nearest Neighbours knowledge boundary estimator.
    
    Usage:
        # During training/setup:
        estimator = KNNBoundaryEstimator(k=5)
        estimator.fit(training_embeddings)
        
        # At inference:
        confidence = estimator.estimate(query_embedding)
    """

    def __init__(self, k: int = 5, threshold: float = 0.50):
        """
        Args:
            k:         Number of nearest neighbours to consider
            threshold: C_boundary below this → OOD flag
        """
        self.k = k
        self.threshold = threshold
        self._training_embeddings: List[List[float]] = []
        self._fitted = False

    def fit(self, embeddings: List[List[float]]) -> "KNNBoundaryEstimator":
        """Register training embeddings as the reference distribution."""
        self._training_embeddings = embeddings
        self._fitted = True
        logger.info(f"KNNBoundaryEstimator fitted on {len(embeddings)} embeddings")
        return self

    def estimate(self, query_embedding: List[float]) -> float:
        """Estimate C_boundary for a query embedding.
        
        Returns:
            float ∈ [0, 1]: higher = more in-distribution
            
        If not fitted, returns 0.9 (permissive default).
        """
        if not self._fitted or not self._training_embeddings:
            return 0.90   # default: assume in-distribution

        distances = self._k_nearest_distances(query_embedding)
        if not distances:
            return 0.90

        mean_dist = sum(distances) / len(distances)
        return 1.0 / (1.0 + mean_dist)

    def is_in_distribution(self, query_embedding: List[float]) -> bool:
        return self.estimate(query_embedding) >= self.threshold

    def _k_nearest_distances(self, query: List[float]) -> List[float]:
        """Find k smallest cosine distances from query to training set.
        
        Uses a max-heap for efficient top-k tracking.
        O(n × d) where n = training set size, d = embedding dimension.
        """
        heap: List[Tuple[float, int]] = []   # max-heap (negate to simulate)

        for i, emb in enumerate(self._training_embeddings):
            dist = 1.0 - self._cosine_similarity(query, emb)
            if len(heap) < self.k:
                heapq.heappush(heap, (-dist, i))
            elif -heap[0][0] > dist:
                heapq.heapreplace(heap, (-dist, i))

        return [-d for d, _ in heap]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot   = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)

    @property
    def training_size(self) -> int:
        return len(self._training_embeddings)


class DensityBoundaryEstimator:
    """Alternative boundary estimator using Gaussian density estimation.
    
    Models training distribution as a mixture of Gaussians.
    Faster than KNN for large training sets.
    
    p_train(q) = (1/N) Σᵢ K(q - xᵢ)  (kernel density estimate)
    K(u) = exp(-‖u‖² / 2σ²) / (σ√2π)  (Gaussian kernel)
    
    C_boundary(q) = p_train(q) / p_max  (normalised density)
    """

    def __init__(self, bandwidth: float = 0.1):
        self.bandwidth = bandwidth
        self._training: List[List[float]] = []
        self._max_density: Optional[float] = None

    def fit(self, embeddings: List[List[float]]) -> "DensityBoundaryEstimator":
        self._training = embeddings
        # Estimate max density (from training examples themselves)
        densities = [self._density(e) for e in embeddings[:100]]  # sample
        self._max_density = max(densities) if densities else 1.0
        return self

    def estimate(self, query_embedding: List[float]) -> float:
        if not self._training:
            return 0.90
        density = self._density(query_embedding)
        normalised = density / max(self._max_density, 1e-8)
        return min(1.0, normalised)

    def _density(self, query: List[float]) -> float:
        """Gaussian kernel density estimate at query point."""
        n = len(self._training)
        if n == 0:
            return 0.0
        total = 0.0
        for emb in self._training:
            sq_dist = sum((a - b) ** 2 for a, b in zip(query, emb))
            total  += math.exp(-sq_dist / (2 * self.bandwidth ** 2))
        return total / n
