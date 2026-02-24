"""
nesy/neural/backbones/transformer.py
=====================================
Transformer backbone wrapper for NeSy-Core.
Wraps HuggingFace sentence-transformers for embedding.
Requires: pip install transformers sentence-transformers
"""
from __future__ import annotations

import logging
import math
from typing import Any, List, Optional

from nesy.neural.base import NeSyBackbone

logger = logging.getLogger(__name__)


class TransformerBackbone(NeSyBackbone):
    """HuggingFace transformer backbone.
    
    Produces sentence/document embeddings via mean pooling
    over the last hidden state.
    
    Default model: 'sentence-transformers/all-MiniLM-L6-v2'
    (384-dim, fast, good quality for symbolic grounding tasks)
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._tokenizer = None
        self._dim: Optional[int] = None

    def _load(self):
        """Lazy load — only download model when first used."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                self._dim = self._model.get_sentence_embedding_dimension()
                logger.info(f"Loaded transformer backbone: {self._model_name} (dim={self._dim})")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "pip install sentence-transformers"
                )

    def encode(self, input_data: Any) -> List[float]:
        """Encode text to embedding vector."""
        self._load()
        embedding = self._model.encode(str(input_data), convert_to_numpy=True)
        return embedding.tolist()

    def encode_batch(self, inputs: List[Any]) -> List[List[float]]:
        """Efficient batch encoding."""
        self._load()
        embeddings = self._model.encode(
            [str(x) for x in inputs],
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return [e.tolist() for e in embeddings]

    def confidence(self, embedding: List[float]) -> float:
        """Confidence via embedding norm stability.
        
        Well-formed sentence embeddings from this model cluster
        around norm ≈ 1.0 (they are approximately normalised).
        """
        norm = math.sqrt(sum(x * x for x in embedding))
        # Confidence = 1 - |norm - 1|, clipped to [0, 1]
        return max(0.0, min(1.0, 1.0 - abs(norm - 1.0)))

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim

    @property
    def name(self) -> str:
        return f"transformer:{self._model_name}"
