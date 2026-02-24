"""
nesy/api/pipeline.py
=====================
NeSyPipeline — composable end-to-end pipeline.

Allows chaining preprocessing → backbone → grounding → symbolic → NSI → metacognition
in a clean, configurable way.

Usage:
    pipeline = (NeSyPipeline()
                .with_backbone(TransformerBackbone())
                .with_rules(medical_rules)
                .with_concept_edges(medical_edges)
                .with_doubt_threshold(0.65)
                .build())

    output = pipeline.run("Patient has fever and elevated WBC.")
"""

from __future__ import annotations
import logging
from typing import Any, List, Optional, Set

from nesy.core.types import ConceptEdge, NSIOutput, Predicate, SymbolicRule
from nesy.neural.base import NeSyBackbone, PassthroughBackbone
from nesy.neural.grounding import SymbolGrounder

logger = logging.getLogger(__name__)


class NeSyPipeline:
    """Fluent builder for an end-to-end NeSy pipeline."""

    def __init__(self):
        self._backbone: Optional[NeSyBackbone] = None
        self._rules: List[SymbolicRule] = []
        self._edges: List[ConceptEdge] = []
        self._domain: str = "general"
        self._threshold: float = 0.60
        self._strict: bool = False

    def with_backbone(self, backbone: NeSyBackbone) -> "NeSyPipeline":
        self._backbone = backbone
        return self

    def with_rules(self, rules: List[SymbolicRule]) -> "NeSyPipeline":
        self._rules.extend(rules)
        return self

    def with_concept_edges(self, edges: List[ConceptEdge]) -> "NeSyPipeline":
        self._edges.extend(edges)
        return self

    def with_domain(self, domain: str) -> "NeSyPipeline":
        self._domain = domain
        return self

    def with_doubt_threshold(self, t: float) -> "NeSyPipeline":
        self._threshold = t
        return self

    def strict(self) -> "NeSyPipeline":
        self._strict = True
        return self

    def build(self) -> "BuiltPipeline":
        from nesy.api.nesy_model import NeSyModel

        model = NeSyModel(
            domain=self._domain,
            doubt_threshold=self._threshold,
            strict_mode=self._strict,
        )
        model.add_rules(self._rules)
        model.add_concept_edges(self._edges)
        backbone = self._backbone or PassthroughBackbone()
        return BuiltPipeline(model=model, backbone=backbone, domain=self._domain)


class BuiltPipeline:
    """Executable pipeline built by NeSyPipeline."""

    def __init__(self, model, backbone: NeSyBackbone, domain: str):
        self._model = model
        self._backbone = backbone
        self._domain = domain
        self._grounder = SymbolGrounder()

    def run(self, input_data: Any, facts: Optional[Set[Predicate]] = None) -> NSIOutput:
        """Run the full pipeline on input.

        If facts are provided directly, skips the neural encoding step.
        If only input_data provided, encodes to embedding → grounds to predicates.
        """
        if facts is None:
            # Encode input
            embedding = self._backbone.encode(input_data)
            neural_conf = self._backbone.confidence(embedding)
            # Ground to predicates
            grounded, ground_conf = self._grounder.ground_with_confidence(embedding)
            facts = {gs.predicate for gs in grounded}
            if not facts:
                logger.warning("No predicates grounded from input. Using empty fact set.")
                facts = set()
            neural_confidence = min(neural_conf, ground_conf) if grounded else neural_conf
        else:
            neural_confidence = 0.90

        return self._model.reason(
            facts=facts,
            context_type=self._domain,
            neural_confidence=neural_confidence,
            raw_input=str(input_data),
        )
