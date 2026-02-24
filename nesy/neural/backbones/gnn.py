"""
nesy/neural/backbones/gnn.py
=============================
Graph Neural Network backbone for structured/relational input.

Used when input is a graph (knowledge graph, molecule, AST, scene graph).
Particularly useful for NSI: the concept graph itself can be encoded
via GNN to produce richer embeddings than text-only models.

Mathematical basis (Graph Convolutional Network):
    h⁽ˡ⁺¹⁾ᵥ = σ( Σᵤ∈N(v) (1/√dᵥdᵤ) W⁽ˡ⁾ h⁽ˡ⁾ᵤ )
    
    Where:
        h⁽ˡ⁾ᵥ  = node v's embedding at layer l
        N(v)    = neighbours of v
        dᵥ      = degree of v (normalisation)
        W⁽ˡ⁾    = learnable weight matrix at layer l
        σ       = ReLU activation
    
    After L layers, h⁽ᴸ⁾ᵥ captures L-hop neighbourhood structure.
    Graph embedding = mean pooling over all node embeddings.
"""
from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

from nesy.neural.base import NeSyBackbone

logger = logging.getLogger(__name__)


class GraphInput:
    """Structured graph input for GNN backbone."""
    def __init__(
        self,
        node_features: List[List[float]],   # shape: [n_nodes, feature_dim]
        edge_list:     List[Tuple[int, int]], # list of (src, tgt) indices
        edge_weights:  Optional[List[float]] = None,
    ):
        self.node_features = node_features
        self.edge_list     = edge_list
        self.edge_weights  = edge_weights or [1.0] * len(edge_list)
        self.n_nodes       = len(node_features)


class SimpleGNNBackbone(NeSyBackbone):
    """Pure-Python GCN implementation (no PyTorch needed).
    
    For production use with large graphs, replace with
    PyTorch Geometric (see integrations/pytorch_lightning.py).
    
    This implementation:
    - 2-layer GCN
    - Mean pooling for graph-level embedding
    - Suitable for small graphs (< 10K nodes)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        self._input_dim  = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        # Random initialisation — replace with trained weights in production
        self._W1 = self._random_matrix(input_dim,  hidden_dim)
        self._W2 = self._random_matrix(hidden_dim, output_dim)

    def encode(self, input_data: Any) -> List[float]:
        """Encode a GraphInput to a graph-level embedding."""
        if not isinstance(input_data, GraphInput):
            raise TypeError(f"GNNBackbone requires GraphInput, got {type(input_data)}")
        return self._gcn_forward(input_data)

    def _gcn_forward(self, graph: GraphInput) -> List[float]:
        """2-layer GCN forward pass → graph embedding."""
        # Build adjacency (with self-loops)
        n = graph.n_nodes
        adj: Dict[int, List[Tuple[int, float]]] = {i: [(i, 1.0)] for i in range(n)}
        for (src, tgt), w in zip(graph.edge_list, graph.edge_weights):
            adj[src].append((tgt, w))
            adj[tgt].append((src, w))   # undirected

        # Degree computation for normalisation
        degree = {i: sum(w for _, w in neighbors) for i, neighbors in adj.items()}

        # Layer 1: h¹ = ReLU(A_norm × X × W1)
        h = graph.node_features
        h = self._gcn_layer(h, adj, degree, self._W1, activation="relu")
        # Layer 2: h² = A_norm × h¹ × W2 (no activation for last layer)
        h = self._gcn_layer(h, adj, degree, self._W2, activation="none")

        # Mean pooling over all nodes → graph embedding
        graph_emb = [
            sum(h[i][j] for i in range(n)) / n
            for j in range(self._output_dim)
        ]
        return graph_emb

    def _gcn_layer(
        self,
        h:          List[List[float]],
        adj:        Dict[int, List[Tuple[int, float]]],
        degree:     Dict[int, float],
        W:          List[List[float]],
        activation: str,
    ) -> List[List[float]]:
        """One GCN layer: h_new = σ(A_norm × h × W)"""
        n = len(h)
        out_dim = len(W[0])
        result = [[0.0] * out_dim for _ in range(n)]

        for v in range(n):
            for u, w_edge in adj[v]:
                # Symmetric normalisation: 1/√(dᵥ × dᵤ)
                norm = 1.0 / math.sqrt(max(degree[v], 1) * max(degree[u], 1))
                # Multiply h[u] by W
                for j in range(out_dim):
                    for k in range(len(h[u])):
                        result[v][j] += norm * w_edge * h[u][k] * W[k][j]

            # Activation
            if activation == "relu":
                result[v] = [max(0.0, x) for x in result[v]]

        return result

    def confidence(self, embedding: List[float]) -> float:
        norm = math.sqrt(sum(x * x for x in embedding))
        return min(1.0, norm / math.sqrt(self._output_dim))

    @staticmethod
    def _random_matrix(rows: int, cols: int) -> List[List[float]]:
        """Xavier-uniform initialisation: U(-√(6/(r+c)), √(6/(r+c)))"""
        import random
        limit = math.sqrt(6.0 / (rows + cols))
        return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]

    @property
    def embedding_dim(self) -> int:
        return self._output_dim

    @property
    def name(self) -> str:
        return f"gnn:gcn({self._input_dim}→{self._hidden_dim}→{self._output_dim})"
