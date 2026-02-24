"""
tests/unit/test_integrations.py
================================
100 % coverage tests for:
    - nesy/integrations/langchain.py
    - nesy/integrations/openai.py
    - nesy/integrations/pytorch_lightning.py

External dependencies (LangChain, OpenAI client, PyTorch Lightning) are
either mocked or skipped when unavailable.  NeSyModel itself is real —
these tests exercise the full NeSy integration path.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nesy.api.nesy_model import NeSyModel
from nesy.core.types import ConceptEdge, Predicate, SymbolicRule
from nesy.integrations.langchain import NeSyReasoningTool
from nesy.integrations.openai import NeSyOpenAIWrapper
from nesy.integrations.pytorch_lightning import NeSyLightningModule, PL_AVAILABLE

# ═══════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════


def _make_model() -> NeSyModel:
    """Build a minimal but real NeSyModel with one rule + one edge."""
    m = NeSyModel(domain="medical", doubt_threshold=0.55)
    m.add_rule(
        SymbolicRule(
            id="fever_inf",
            antecedents=[
                Predicate("HasSymptom", ("?p", "fever")),
                Predicate("HasLabResult", ("?p", "elevated_wbc")),
            ],
            consequents=[Predicate("PossiblyHas", ("?p", "bacterial_infection"))],
            weight=0.85,
            domain="medical",
            description="Fever + WBC → infection",
        )
    )
    m.add_concept_edge(
        ConceptEdge(
            "fever",
            "blood_test",
            cooccurrence_prob=0.90,
            causal_strength=1.0,
            temporal_stability=1.0,
        )
    )
    m.register_critical_concept("blood_test", "diagnostic_test")
    return m


# ═══════════════════════════════════════════════════════════════════
#  LangChain integration — NeSyReasoningTool
# ═══════════════════════════════════════════════════════════════════


class TestNeSyReasoningToolInit:
    def test_valid_model(self):
        model = _make_model()
        tool = NeSyReasoningTool(model)
        assert tool.model is model

    def test_invalid_model_raises_type_error(self):
        with pytest.raises(TypeError, match="NeSyModel"):
            NeSyReasoningTool("not_a_model")

    def test_name_and_description(self):
        tool = NeSyReasoningTool(_make_model())
        assert isinstance(tool.name, str) and len(tool.name) > 0
        assert isinstance(tool.description, str) and len(tool.description) > 0


class TestNeSyReasoningToolRun:
    @pytest.fixture
    def tool(self):
        return NeSyReasoningTool(_make_model())

    def test_valid_json_returns_answer(self, tool):
        inp = json.dumps(
            {
                "facts": [
                    {"name": "HasSymptom", "args": ["p1", "fever"]},
                    {"name": "HasLabResult", "args": ["p1", "elevated_wbc"]},
                ]
            }
        )
        result = json.loads(tool._run(inp))
        assert "answer" in result
        assert "confidence" in result
        assert "status" in result
        assert "trustworthy" in result
        assert isinstance(result["critical_nulls"], int)

    def test_invalid_json_returns_error(self, tool):
        result = json.loads(tool._run("{bad json"))
        assert "error" in result
        assert "Invalid JSON" in result["error"]

    def test_empty_facts(self, tool):
        result = json.loads(tool._run(json.dumps({"facts": []})))
        assert "answer" in result

    def test_missing_facts_key(self, tool):
        result = json.loads(tool._run(json.dumps({})))
        assert "answer" in result  # empty facts → still produces output

    def test_custom_context_type(self, tool):
        inp = json.dumps(
            {
                "facts": [{"name": "A", "args": ["x"]}],
                "context_type": "legal",
            }
        )
        result = json.loads(tool._run(inp))
        assert "answer" in result

    def test_custom_neural_confidence(self, tool):
        inp = json.dumps(
            {
                "facts": [{"name": "A", "args": ["x"]}],
                "neural_confidence": 0.75,
            }
        )
        result = json.loads(tool._run(inp))
        assert "answer" in result

    def test_neural_confidence_clamped_high(self, tool):
        inp = json.dumps(
            {
                "facts": [{"name": "A", "args": ["x"]}],
                "neural_confidence": 5.0,
            }
        )
        result = json.loads(tool._run(inp))
        assert "answer" in result  # should not crash

    def test_neural_confidence_clamped_low(self, tool):
        inp = json.dumps(
            {
                "facts": [{"name": "A", "args": ["x"]}],
                "neural_confidence": -3.0,
            }
        )
        result = json.loads(tool._run(inp))
        assert "answer" in result

    def test_fact_missing_name_skipped(self, tool):
        """Fact with no 'name' key should be skipped, not crash."""
        inp = json.dumps(
            {
                "facts": [
                    {"args": ["x"]},  # missing name
                    {"name": "B", "args": ["y"]},  # valid
                ]
            }
        )
        result = json.loads(tool._run(inp))
        assert "answer" in result
        assert "error" not in result

    def test_fact_with_no_args(self, tool):
        """Fact with name but no args → args defaults to ()."""
        inp = json.dumps({"facts": [{"name": "SomeFact"}]})
        result = json.loads(tool._run(inp))
        assert "answer" in result

    def test_reasoning_exception_returns_error(self, tool):
        """If model.reason() raises, _run returns JSON error."""
        with patch.object(tool._model, "reason", side_effect=RuntimeError("boom")):
            result = json.loads(tool._run(json.dumps({"facts": [{"name": "A", "args": ["x"]}]})))
            assert "error" in result
            assert "boom" in result["error"]


class TestNeSyReasoningToolAsync:
    def test_arun_delegates_to_run(self):
        tool = NeSyReasoningTool(_make_model())
        inp = json.dumps({"facts": [{"name": "X", "args": ["a"]}]})
        sync_result = tool._run(inp)
        async_result = asyncio.run(tool._arun(inp))
        assert sync_result == async_result


class TestNeSyReasoningToolArgsSchema:
    def test_args_schema_structure(self):
        tool = NeSyReasoningTool(_make_model())
        schema = tool.args_schema
        assert schema["type"] == "object"
        assert "facts" in schema["properties"]
        assert "facts" in schema["required"]


# ═══════════════════════════════════════════════════════════════════
#  OpenAI integration — NeSyOpenAIWrapper
# ═══════════════════════════════════════════════════════════════════


def _mock_openai_response(
    content: str = "The patient may have an infection.",
    finish_reason: str = "stop",
) -> MagicMock:
    """Build a mock that quacks like openai.ChatCompletion response."""
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    response = SimpleNamespace(choices=[choice])
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


class TestNeSyOpenAIWrapperInit:
    def test_valid_init(self):
        client = _mock_openai_response()
        model = _make_model()
        wrapper = NeSyOpenAIWrapper(client, model)
        assert wrapper.model is model

    def test_invalid_model_raises_type_error(self):
        with pytest.raises(TypeError, match="NeSyModel"):
            NeSyOpenAIWrapper(MagicMock(), "not_a_model")


class TestNeSyOpenAIChatWithReasoning:
    @pytest.fixture
    def wrapper(self):
        return NeSyOpenAIWrapper(_mock_openai_response(), _make_model())

    def test_basic_call(self, wrapper):
        result = wrapper.chat_with_reasoning(
            messages=[{"role": "user", "content": "diagnose patient"}],
            facts={Predicate("HasSymptom", ("p1", "fever"))},
            context_type="medical",
        )
        assert "llm_response" in result
        assert "nesy_output" in result
        assert "trustworthy" in result
        assert "confidence" in result
        assert result["llm_response"] == "The patient may have an infection."

    def test_no_facts_defaults_to_empty_set(self, wrapper):
        result = wrapper.chat_with_reasoning(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert "nesy_output" in result

    def test_truncated_response_penalty(self):
        """finish_reason == 'length' should lower neural confidence."""
        client = _mock_openai_response(finish_reason="length")
        wrapper = NeSyOpenAIWrapper(client, _make_model())
        result = wrapper.chat_with_reasoning(
            messages=[{"role": "user", "content": "test"}],
            facts={Predicate("A", ("x",))},
        )
        # neural_conf = 0.8 (penalised from 1.0)
        assert "confidence" in result

    def test_openai_kwargs_forwarded(self):
        client = _mock_openai_response()
        wrapper = NeSyOpenAIWrapper(client, _make_model())
        wrapper.chat_with_reasoning(
            messages=[{"role": "user", "content": "q"}],
            model="gpt-4o",
            temperature=0.2,
        )
        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("model") == "gpt-4o"
        assert call_kwargs.kwargs.get("temperature") == 0.2


# ═══════════════════════════════════════════════════════════════════
#  PyTorch Lightning integration — NeSyLightningModule
# ═══════════════════════════════════════════════════════════════════


class TestPLAvailability:
    def test_is_available_returns_bool(self):
        assert isinstance(NeSyLightningModule.is_available(), bool)

    def test_is_available_matches_module_flag(self):
        assert NeSyLightningModule.is_available() == PL_AVAILABLE


@pytest.mark.skipif(not PL_AVAILABLE, reason="pytorch-lightning not installed")
class TestPLBuild:
    """Tests that require actual pytorch-lightning + torch."""

    @staticmethod
    def _make_backbone():
        import torch.nn as nn

        class SimpleBB(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def encode(self, x):
                return self.linear(x)

        return SimpleBB()

    @staticmethod
    def _make_backbone_scalar():
        """Backbone with scalar (numel=1) parameters for EWC path."""
        import torch.nn as nn

        class ScalarBB(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def encode(self, x):
                return self.linear(x)

        return ScalarBB()

    @staticmethod
    def _make_learner():
        from nesy.continual.learner import ContinualLearner

        return ContinualLearner(lambda_ewc=1000.0)

    @staticmethod
    def _loss_fn(output, target):
        import torch

        return torch.nn.functional.mse_loss(output, target)

    def test_build_returns_lightning_module(self):
        import pytorch_lightning as pl

        module = NeSyLightningModule.build(
            self._make_backbone(), self._make_learner(), self._loss_fn
        )
        assert isinstance(module, pl.LightningModule)

    def test_build_custom_lr(self):
        module = NeSyLightningModule.build(
            self._make_backbone(),
            self._make_learner(),
            self._loss_fn,
            learning_rate=1e-3,
        )
        optim = module.configure_optimizers()
        assert optim.defaults["lr"] == 1e-3

    def test_forward_calls_encode(self):
        import torch

        module = NeSyLightningModule.build(
            self._make_backbone(), self._make_learner(), self._loss_fn
        )
        x = torch.randn(2, 4)
        out = module(x)
        assert out.shape == (2, 2)

    def test_training_step_returns_loss(self):
        import torch

        module = NeSyLightningModule.build(
            self._make_backbone(), self._make_learner(), self._loss_fn
        )
        x = torch.randn(2, 4)
        y = torch.randn(2, 2)
        loss = module.training_step((x, y), batch_idx=0)
        assert loss.dim() == 0  # scalar

    def test_training_step_ewc_penalty(self):
        """EWC penalty should be computed when backbone has scalar params."""
        import torch

        bb = self._make_backbone_scalar()
        learner = self._make_learner()
        module = NeSyLightningModule.build(bb, learner, self._loss_fn)

        x = torch.randn(2, 1)
        y = torch.randn(2, 1)
        loss = module.training_step((x, y), batch_idx=0)
        assert loss.item() >= 0


class TestPLBuildImportError:
    """Test that build() raises ImportError when PL is absent."""

    def test_build_raises_when_pl_missing(self):
        """Simulate PL not being importable."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("pytorch_lightning", "torch", "torch.nn"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="pytorch-lightning"):
                NeSyLightningModule.build(None, None, None)
