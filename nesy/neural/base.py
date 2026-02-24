"""
nesy/neural/base.py
===================
Abstract base class for all neural backbones in NeSy-Core.

Every neural model that plugs into NeSy-Core must implement
this interface. This decouples the symbolic reasoning layer
from any specific ML framework (PyTorch, JAX, etc.).

Mathematical contract:
    encode(input) → embedding ∈ ℝᵈ
    confidence(embedding) → scalar ∈ [0,1]
    
The bridge.py module uses these two outputs to:
    1. Ground the embedding to symbolic predicates
    2. Feed neural confidence into MetaCognitionMonitor
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class NeSyBackbone(ABC):
    """Abstract neural backbone interface.
    
    Implementors: TransformerBackbone, GNNBackbone, custom models.
    
    Contract:
        - encode() must return a list of floats (framework-agnostic)
        - confidence() must return a float in [0, 1]
        - Both methods must be deterministic given same input + same model state
    """

    @abstractmethod
    def encode(self, input_data: Any) -> List[float]:
        """Encode input into a fixed-dimensional embedding vector.
        
        Args:
            input_data: Raw input (text string, token list, image tensor, etc.)
        
        Returns:
            List of floats representing the embedding.
            Dimensionality must be consistent for a given backbone instance.
        """
        ...

    @abstractmethod
    def confidence(self, embedding: List[float]) -> float:
        """Estimate the model's confidence in its encoding.
        
        This is NOT the task-specific prediction confidence.
        It is the backbone's estimate of how well it understood the input:
            - Is this input in-distribution?
            - Is the embedding geometrically stable?
        
        Returns: float ∈ [0, 1]
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this backbone."""
        ...

    def encode_batch(self, inputs: List[Any]) -> List[List[float]]:
        """Default batch encoding (sequential). Override for efficiency."""
        return [self.encode(inp) for inp in inputs]


class PassthroughBackbone(NeSyBackbone):
    """Minimal backbone for testing: passes through pre-computed embeddings.
    
    Useful when embeddings are computed externally (e.g., from OpenAI API)
    and you only need the symbolic + NSI layers of NeSy-Core.
    """

    def __init__(self, dim: int = 768):
        self._dim = dim

    def encode(self, input_data: Any) -> List[float]:
        """Expects input_data to already be a List[float]."""
        if isinstance(input_data, list):
            return input_data
        raise TypeError(f"PassthroughBackbone expects List[float], got {type(input_data)}")

    def confidence(self, embedding: List[float]) -> float:
        """Confidence = L2-normalised magnitude proxy.
        Well-formed embeddings cluster around unit norm.
        """
        import math
        norm = math.sqrt(sum(x * x for x in embedding))
        # Map norm to [0,1]: norm ≈ 1.0 → confidence ≈ 1.0
        return min(1.0, norm)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "passthrough"
