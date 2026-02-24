"""
nesy/integrations/langchain.py
================================
LangChain tool integration for NeSy-Core.

Allows NeSy reasoning to be used as a LangChain tool in agent pipelines.
A LangChain agent can call ``NeSyReasoningTool`` to get symbolic reasoning
with confidence scores, null set analysis, and self-doubt.

Requires: ``pip install langchain-core``

Usage:
    from nesy import NeSyModel
    from nesy.integrations.langchain import NeSyReasoningTool

    model = NeSyModel(domain="medical")
    tool  = NeSyReasoningTool(model)

    # Use with a LangChain agent
    result = tool._run('{"facts": [{"name": "HasSymptom", "args": ["p1", "fever"]}]}')
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Type

from nesy.api.nesy_model import NeSyModel
from nesy.core.types import NSIOutput, Predicate

logger = logging.getLogger(__name__)


class NeSyReasoningTool:
    """LangChain-compatible tool wrapping NeSy-Core reasoning.

    Implements the LangChain ``BaseTool`` interface (name, description, _run).
    Can be used standalone or injected into any LangChain agent.

    Input format (JSON string):
        {
            "facts": [{"name": "HasSymptom", "args": ["p1", "fever"]}],
            "context_type": "medical",      // optional, default "general"
            "neural_confidence": 0.90       // optional, default 0.90
        }

    Output format (JSON string):
        {
            "answer": "Derived: ...",
            "status": "ok",
            "confidence": 0.85,
            "factual": 0.90,
            "reasoning": 0.88,
            "boundary": 0.85,
            "trustworthy": true,
            "critical_nulls": 0,
            "flags": []
        }

    Args:
        model: A configured NeSyModel instance.

    Raises:
        TypeError: if model is not a NeSyModel instance.
    """

    name: str = "nesy_reasoning"
    description: str = (
        "Symbolic reasoning with confidence scores and self-doubt. "
        "Input: JSON with 'facts' list of {name, args} objects. "
        "Output: reasoning result with three-dimensional confidence."
    )

    def __init__(self, model: NeSyModel) -> None:
        if not isinstance(model, NeSyModel):
            raise TypeError(
                f"model must be a NeSyModel, got {type(model).__name__}"
            )
        self._model = model
        logger.info("NeSyReasoningTool initialised.")

    @property
    def model(self) -> NeSyModel:
        """The underlying NeSyModel instance."""
        return self._model

    def _run(self, tool_input: str) -> str:
        """Execute NeSy reasoning on the provided facts.

        Args:
            tool_input: JSON string with 'facts' key.

        Returns:
            JSON string with answer, confidence breakdown, and flags.
        """
        try:
            data = json.loads(tool_input)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON input to NeSyReasoningTool: %s", e)
            return json.dumps({"error": f"Invalid JSON: {e}"})

        try:
            facts: Set[Predicate] = set()
            for f in data.get("facts", []):
                name = f.get("name")
                if not name:
                    logger.warning("Fact missing 'name' field, skipping: %s", f)
                    continue
                facts.add(
                    Predicate(
                        name=name,
                        args=tuple(f.get("args", [])),
                    )
                )

            context_type = data.get("context_type", "general")
            neural_confidence = data.get("neural_confidence", 0.90)

            # Clamp neural_confidence to [0, 1]
            neural_confidence = min(1.0, max(0.0, float(neural_confidence)))

            logger.debug(
                "NeSyReasoningTool: %d facts, context=%s, neural_conf=%.3f",
                len(facts),
                context_type,
                neural_confidence,
            )

            output = self._model.reason(
                facts=facts,
                context_type=context_type,
                neural_confidence=neural_confidence,
            )

            return json.dumps({
                "answer": output.answer,
                "status": output.status.value,
                "confidence": output.confidence.minimum,
                "factual": output.confidence.factual,
                "reasoning": output.confidence.reasoning,
                "boundary": output.confidence.knowledge_boundary,
                "trustworthy": output.is_trustworthy(),
                "critical_nulls": len(output.null_set.critical_items),
                "flags": output.flags,
            })
        except Exception as e:
            logger.error("NeSy reasoning error: %s", e, exc_info=True)
            return json.dumps({"error": f"NeSy reasoning error: {e}"})

    async def _arun(self, tool_input: str) -> str:
        """Async version — delegates to sync _run.

        Symbolic reasoning is CPU-bound, so async wrapper just
        calls the synchronous method directly.
        """
        return self._run(tool_input)

    @property
    def args_schema(self) -> Dict[str, Any]:
        """Schema describing the expected input format."""
        return {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "args": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["name"],
                    },
                },
                "context_type": {"type": "string", "default": "general"},
                "neural_confidence": {"type": "number", "default": 0.90},
            },
            "required": ["facts"],
        }
