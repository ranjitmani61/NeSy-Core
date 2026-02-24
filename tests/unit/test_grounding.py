"""
tests/unit/test_grounding.py
==============================
Tests for neural → symbolic grounding via cosine similarity.

Mathematical basis:
    similarity(e, μₚ) = (e · μₚ) / (‖e‖ × ‖μₚ‖)
    Grounded iff similarity ≥ threshold.
"""

import math
import pytest
from nesy.neural.grounding import SymbolGrounder, PredicatePrototype
from nesy.core.types import Predicate


# ─── SymbolGrounder ─────────────────────────────────────────────


class TestSymbolGrounderInit:
    def test_default_threshold(self):
        g = SymbolGrounder()
        assert 0.0 < g.threshold < 1.0

    def test_custom_threshold(self):
        g = SymbolGrounder(threshold=0.80)
        assert g.threshold == 0.80

    def test_invalid_threshold_zero(self):
        with pytest.raises(AssertionError):
            SymbolGrounder(threshold=0.0)

    def test_invalid_threshold_one(self):
        with pytest.raises(AssertionError):
            SymbolGrounder(threshold=1.0)


class TestGrounding:
    @pytest.fixture
    def grounder(self):
        g = SymbolGrounder(threshold=0.70)
        g.register(
            PredicatePrototype(
                predicate=Predicate("HasSymptom", ("?p", "fever")),
                prototype=[1.0, 0.0, 0.0],
                domain="medical",
            )
        )
        g.register(
            PredicatePrototype(
                predicate=Predicate("HasSymptom", ("?p", "cough")),
                prototype=[0.0, 1.0, 0.0],
                domain="medical",
            )
        )
        return g

    def test_identical_vector_grounded(self, grounder):
        """Cosine(e, e) = 1.0 → always above threshold."""
        result = grounder.ground([1.0, 0.0, 0.0])
        assert len(result) >= 1
        assert result[0].grounding_confidence == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vector_not_grounded(self, grounder):
        """Cosine of orthogonal vectors = 0.0 → below any reasonable threshold."""
        result = grounder.ground([0.0, 0.0, 1.0])
        assert len(result) == 0

    def test_grounded_symbols_sorted_by_confidence(self, grounder):
        """Results must be sorted descending by grounding_confidence."""
        # Emit a vector between the two prototypes, closer to first
        result = grounder.ground([0.9, 0.3, 0.0])
        if len(result) >= 2:
            assert result[0].grounding_confidence >= result[1].grounding_confidence

    def test_domain_filter(self, grounder):
        """Non-matching domain should be excluded."""
        grounder.register(
            PredicatePrototype(
                predicate=Predicate("CodeError", ("?f",)),
                prototype=[1.0, 0.0, 0.0],
                domain="code",
            )
        )
        result = grounder.ground([1.0, 0.0, 0.0], domain="medical")
        pred_names = {r.predicate.name for r in result}
        assert "CodeError" not in pred_names

    def test_top_k_limits_results(self, grounder):
        """top_k controls max returned results."""
        result = grounder.ground([0.8, 0.6, 0.0], top_k=1)
        assert len(result) <= 1

    def test_grounding_with_confidence(self, grounder):
        """ground_with_confidence returns (list, overall_conf)."""
        symbols, conf = grounder.ground_with_confidence([1.0, 0.0, 0.0])
        assert len(symbols) >= 1
        assert conf == pytest.approx(1.0, abs=1e-6)

    def test_grounding_empty_returns_zero(self, grounder):
        """No match → confidence = 0.0."""
        symbols, conf = grounder.ground_with_confidence([0.0, 0.0, 1.0])
        assert len(symbols) == 0
        assert conf == 0.0


class TestCosine:
    def test_identical_unit_vectors(self):
        assert SymbolGrounder._cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        assert SymbolGrounder._cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        assert SymbolGrounder._cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        assert SymbolGrounder._cosine_similarity([0, 0], [1, 0]) == 0.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            SymbolGrounder._cosine_similarity([1, 0], [1, 0, 0])


class TestPrototypeFromExamples:
    def test_builds_normalised_prototype(self):
        g = SymbolGrounder()
        proto = g.build_prototype_from_examples(
            predicate=Predicate("Test", ()),
            example_embeddings=[[2.0, 0.0], [0.0, 2.0]],
        )
        # Mean is [1, 1], L2-normalised → [1/√2, 1/√2]
        expected = 1.0 / math.sqrt(2)
        assert proto.prototype[0] == pytest.approx(expected, abs=1e-6)
        assert proto.prototype[1] == pytest.approx(expected, abs=1e-6)

    def test_prototype_registered(self):
        g = SymbolGrounder()
        g.build_prototype_from_examples(
            predicate=Predicate("Test", ()),
            example_embeddings=[[1.0, 0.0]],
        )
        assert g.prototype_count == 1

    def test_empty_examples_raises(self):
        g = SymbolGrounder()
        with pytest.raises(ValueError, match="empty"):
            g.build_prototype_from_examples(
                predicate=Predicate("Test", ()),
                example_embeddings=[],
            )


class TestRegisterBatch:
    def test_batch_registers_all(self):
        g = SymbolGrounder()
        protos = [
            PredicatePrototype(predicate=Predicate("A", ()), prototype=[1.0, 0.0]),
            PredicatePrototype(predicate=Predicate("B", ()), prototype=[0.0, 1.0]),
        ]
        g.register_batch(protos)
        assert g.prototype_count == 2
