"""
nesy/integrations/openai.py
==============================
Post-process OpenAI API responses through NeSy-Core reasoning.

The NeSy layer adds symbolic verification on top of LLM outputs:
    1. Extract facts from LLM response
    2. Check for null set violations
    3. Return NSIOutput with full confidence analysis

Requires: ``pip install openai``

Usage:
    from nesy.integrations.openai import NeSyOpenAIWrapper
    from nesy import NeSyModel

    model = NeSyModel(domain="medical")
    wrapper = NeSyOpenAIWrapper(openai_client, model)
    result = wrapper.chat_with_reasoning(messages=[...], facts={...})
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from nesy.api.nesy_model import NeSyModel
from nesy.core.types import NSIOutput, Predicate

logger = logging.getLogger(__name__)


class NeSyOpenAIWrapper:
    """Wraps an OpenAI-compatible client to add NeSy post-processing.

    After the LLM generates a response, NeSy-Core runs symbolic
    reasoning + NSI analysis over the provided facts and returns
    a three-dimensional confidence report alongside the LLM text.

    The wrapper does NOT modify the LLM response — it augments it
    with epistemic confidence and null-set analysis.

    Args:
        openai_client: Any object with a ``chat.completions.create()``
                       method (openai.OpenAI(), or a mock).
        nesy_model:    Configured NeSyModel instance.

    Raises:
        TypeError: if nesy_model is not a NeSyModel instance.
    """

    def __init__(self, openai_client: Any, nesy_model: NeSyModel) -> None:
        if not isinstance(nesy_model, NeSyModel):
            raise TypeError(
                f"nesy_model must be a NeSyModel, got {type(nesy_model).__name__}"
            )
        self._client = openai_client
        self._nesy = nesy_model
        logger.info("NeSyOpenAIWrapper initialised.")

    @property
    def model(self) -> NeSyModel:
        """The underlying NeSyModel instance."""
        return self._nesy

    def chat_with_reasoning(
        self,
        messages: List[Dict[str, str]],
        facts: Optional[Set[Predicate]] = None,
        context_type: str = "general",
        **openai_kwargs: Any,
    ) -> Dict[str, Any]:
        """Call OpenAI chat, then run NeSy reasoning on result.

        Args:
            messages:       OpenAI-format message list.
            facts:          Set of Predicate objects for symbolic reasoning.
                            If None, an empty set is used.
            context_type:   Domain context for NSI analysis.
            **openai_kwargs: Forwarded to ``chat.completions.create()``.

        Returns:
            Dict with keys:
                llm_response: The LLM's text response.
                nesy_output:  NSIOutput with confidence + reasoning trace.
                trustworthy:  Whether the output passes self-doubt.
                confidence:   Minimum confidence across all three dimensions.
        """
        logger.debug(
            "chat_with_reasoning: %d messages, context=%s",
            len(messages),
            context_type,
        )

        response = self._client.chat.completions.create(
            messages=messages,
            **openai_kwargs,
        )

        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        # Penalise truncated completions (finish_reason == "length")
        neural_conf = 1.0 - (0.2 if finish_reason == "length" else 0.0)

        reasoning_facts = facts if facts is not None else set()

        nesy_output = self._nesy.reason(
            facts=reasoning_facts,
            context_type=context_type,
            neural_confidence=neural_conf,
            raw_input=content,
        )

        logger.info(
            "NeSy post-processing: status=%s confidence=%.3f",
            nesy_output.status.value,
            nesy_output.confidence.minimum,
        )

        return {
            "llm_response": content,
            "nesy_output": nesy_output,
            "trustworthy": nesy_output.is_trustworthy(),
            "confidence": nesy_output.confidence.minimum,
        }
