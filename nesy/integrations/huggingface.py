"""
nesy/integrations/huggingface.py
==================================
Wrap any HuggingFace model with NeSy-Core reasoning.

This allows existing HuggingFace pipelines to gain:
    - Symbolic constraint checking
    - NSI null set analysis
    - Three-dimensional confidence
    - Self-doubt layer

Usage:
    from transformers import pipeline
    from nesy.integrations.huggingface import NeSyHFWrapper

    hf_model = pipeline("text-classification", model="bert-base-uncased")
    nesy_model = NeSyHFWrapper(hf_model, rules=medical_rules)
    output = nesy_model.reason(text="Patient has fever")
"""

from __future__ import annotations
import logging
from typing import Any, List, Optional
from nesy.api.nesy_model import NeSyModel
from nesy.core.types import ConceptEdge, NSIOutput, SymbolicRule

logger = logging.getLogger(__name__)


class NeSyHFWrapper:
    """Wraps a HuggingFace model with NeSy reasoning."""

    def __init__(
        self,
        hf_pipeline: Any,
        rules: Optional[List[SymbolicRule]] = None,
        edges: Optional[List[ConceptEdge]] = None,
        domain: str = "general",
    ):
        self._hf = hf_pipeline
        self._nesy = NeSyModel(domain=domain)
        if rules:
            self._nesy.add_rules(rules)
        if edges:
            self._nesy.add_concept_edges(edges)

    def reason(self, text: str, context_type: str = "general") -> NSIOutput:
        """Run HF model → extract confidence → pass through NeSy reasoning."""
        # Run HF model
        try:
            hf_result = self._hf(text)
            if isinstance(hf_result, list) and hf_result:
                neural_conf = hf_result[0].get("score", 0.85)
            else:
                neural_conf = 0.85
        except Exception as e:
            logger.warning(f"HF model call failed: {e}. Using default confidence.")
            neural_conf = 0.75

        # For now, use empty facts — in production, extract facts from HF output
        facts: set = set()
        return self._nesy.reason(
            facts=facts,
            context_type=context_type,
            neural_confidence=neural_conf,
            raw_input=text,
        )
