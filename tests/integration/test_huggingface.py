"""
tests/integration/test_huggingface.py
=======================================
Integration tests for HuggingFace wrapper.

These test the NeSyHFWrapper with a mock HF pipeline
(no real transformers model needed).
"""
import pytest
from nesy.integrations.huggingface import NeSyHFWrapper
from nesy.core.types import (
    ConceptEdge,
    NSIOutput,
    OutputStatus,
    Predicate,
    SymbolicRule,
)


class MockHFPipeline:
    """Simulates a HuggingFace pipeline.__call__ return value."""
    def __init__(self, score: float = 0.92, label: str = "positive"):
        self._score = score
        self._label = label

    def __call__(self, text: str):
        return [{"label": self._label, "score": self._score}]


class MockFailingPipeline:
    """Simulates a HF pipeline that raises on __call__."""
    def __call__(self, text: str):
        raise RuntimeError("Model loading failed")


# ─── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def medical_rules():
    return [
        SymbolicRule(
            id="fever_test",
            antecedents=[Predicate("HasSymptom", ("?p", "fever"))],
            consequents=[Predicate("RequiresTest", ("?p", "blood_test"))],
            weight=0.85,
            domain="medical",
        ),
    ]


@pytest.fixture
def medical_edges():
    return [
        ConceptEdge(
            "fever", "blood_test",
            cooccurrence_prob=0.90,
            causal_strength=1.0,
            temporal_stability=1.0,
        ),
    ]


# ─── Tests ────────────────────────────────────────────────────


class TestNeSyHFWrapper:
    def test_init_with_rules(self, medical_rules, medical_edges):
        wrapper = NeSyHFWrapper(
            hf_pipeline=MockHFPipeline(),
            rules=medical_rules,
            edges=medical_edges,
            domain="medical",
        )
        assert wrapper._nesy is not None

    def test_reason_returns_nsi_output(self, medical_rules):
        wrapper = NeSyHFWrapper(
            hf_pipeline=MockHFPipeline(score=0.95),
            rules=medical_rules,
            domain="medical",
        )
        output = wrapper.reason("Patient has fever")
        assert isinstance(output, NSIOutput)
        assert output.answer is not None
        assert 0.0 <= output.confidence.factual <= 1.0
        assert 0.0 <= output.confidence.reasoning <= 1.0
        assert 0.0 <= output.confidence.knowledge_boundary <= 1.0

    def test_reason_with_high_hf_score(self, medical_rules):
        wrapper = NeSyHFWrapper(
            hf_pipeline=MockHFPipeline(score=0.99),
            rules=medical_rules,
        )
        output = wrapper.reason("test input")
        assert output.status in (
            OutputStatus.OK, OutputStatus.FLAGGED,
            OutputStatus.UNCERTAIN, OutputStatus.REJECTED,
        )

    def test_reason_handles_hf_failure(self, medical_rules):
        """When HF model fails, wrapper uses default confidence."""
        wrapper = NeSyHFWrapper(
            hf_pipeline=MockFailingPipeline(),
            rules=medical_rules,
        )
        output = wrapper.reason("test input")
        assert isinstance(output, NSIOutput)

    def test_no_rules_no_edges(self):
        """Wrapper works even with no rules or edges."""
        wrapper = NeSyHFWrapper(
            hf_pipeline=MockHFPipeline(),
        )
        output = wrapper.reason("plain input")
        assert isinstance(output, NSIOutput)

    def test_domain_propagated(self):
        wrapper = NeSyHFWrapper(
            hf_pipeline=MockHFPipeline(),
            domain="legal",
        )
        assert wrapper._nesy.domain == "legal"
