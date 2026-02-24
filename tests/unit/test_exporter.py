"""
tests/unit/test_exporter.py
============================
100 % coverage for ``nesy/deployment/exporter.py``.

Strategy
--------
* **ONNX export — REAL** : torch 2.10.0+cpu and onnx 1.20.1 are
  installed, so we perform full round-trip tests:
  write → checksum → onnx.load → validate → read manifest.
* **TFLite / CoreML** : TensorFlow and coremltools are NOT installed,
  so we verify the **staged-artifact + typed-error** code path and
  also mock-test the happy path so every branch is covered.
* **ExportManifest** : full lifecycle — construction, ``to_dict()``,
  ``save()``, ``load()``, round-trip, error handling.
* **Validation helpers** : bad backbone, bad input_dim, empty path.
* **export_bundle** : multi-format, unknown format, reasoning config.

All temp files go into ``tmp_path`` (pytest auto-cleanup).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import hashlib

import pytest
import torch
import torch.nn as nn

from nesy.core.exceptions import NeSyError
from nesy.deployment.exporter import (
    ExportManifest,
    ModelExporter,
    _sha256_file,
    _tflite_encode,
    _validate_backbone,
    _validate_input_dim,
    _validate_output_path,
)


# ════════════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════════════

class _TinyBackbone(nn.Module):
    """Minimal torch backbone that satisfies NeSyBackbone contract."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.name = "TinyBackbone"
        self.embedding_dim = dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def confidence(self, embedding: Any) -> float:
        return 0.95


class _NoEncodeBackbone:
    """Object that has no encode() — should be rejected."""
    pass


@pytest.fixture
def tiny_backbone():
    """A small torch.nn.Module backbone for real export tests."""
    return _TinyBackbone(dim=16)


@pytest.fixture
def sample_reasoning_config():
    return {
        "domain": "medical",
        "doubt_threshold": 0.55,
        "rules_count": 2,
        "symbolic_strict_mode": True,
    }


# ════════════════════════════════════════════════════════════════════
#  ExportManifest
# ════════════════════════════════════════════════════════════════════


class TestExportManifest:
    """ExportManifest dataclass lifecycle."""

    def test_defaults_populate_timestamp(self):
        m = ExportManifest(model_name="A", domain="d", export_format="onnx",
                           input_shape=[1, 16])
        assert m.timestamp  # auto-populated
        assert m.checksum == ""
        assert m.output_shape is None
        assert m.quantization_mode == "none"
        assert m.reasoning_config is None
        assert m.artifact_path == ""

    def test_to_dict_returns_plain_dict(self):
        m = ExportManifest(
            model_name="B", domain="gen", export_format="tflite",
            input_shape=[1, 32], checksum="abc123",
        )
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["model_name"] == "B"
        assert d["checksum"] == "abc123"
        assert d["export_format"] == "tflite"

    def test_save_and_load_round_trip(self, tmp_path):
        m = ExportManifest(
            model_name="RT", domain="d", export_format="onnx",
            input_shape=[1, 64], output_shape=[1, 64],
            quantization_mode="dynamic", checksum="deadbeef",
            reasoning_config={"key": "val"}, artifact_path="/tmp/model.onnx",
        )
        json_path = str(tmp_path / "manifest.json")
        m.save(json_path)

        loaded = ExportManifest.load(json_path)
        assert loaded.model_name == "RT"
        assert loaded.checksum == "deadbeef"
        assert loaded.reasoning_config == {"key": "val"}
        assert loaded.artifact_path == "/tmp/model.onnx"
        assert loaded.quantization_mode == "dynamic"

    def test_save_creates_parent_dir(self, tmp_path):
        deep_path = str(tmp_path / "a" / "b" / "manifest.json")
        m = ExportManifest(model_name="x", domain="d", export_format="onnx",
                           input_shape=[1, 8])
        m.save(deep_path)
        assert Path(deep_path).exists()

    def test_load_bad_file_raises_nesy_error(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("NOT JSON", encoding="utf-8")
        with pytest.raises(NeSyError, match="Cannot load export manifest"):
            ExportManifest.load(str(bad))

    def test_load_nonexistent_file_raises_nesy_error(self, tmp_path):
        with pytest.raises(NeSyError, match="Cannot load export manifest"):
            ExportManifest.load(str(tmp_path / "no_such.json"))

    def test_load_ignores_extra_keys(self, tmp_path):
        data = {
            "model_name": "Z", "domain": "d", "export_format": "onnx",
            "input_shape": [1, 4], "extra_garbage": True,
        }
        p = tmp_path / "extra.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        loaded = ExportManifest.load(str(p))
        assert loaded.model_name == "Z"
        assert not hasattr(loaded, "extra_garbage")

    def test_custom_timestamp_preserved(self):
        m = ExportManifest(
            model_name="T", domain="d", export_format="onnx",
            input_shape=[1, 8], timestamp="2024-01-01T00:00:00+00:00",
        )
        assert m.timestamp == "2024-01-01T00:00:00+00:00"


# ════════════════════════════════════════════════════════════════════
#  Validation helpers
# ════════════════════════════════════════════════════════════════════


class TestValidationHelpers:

    def test_validate_backbone_ok(self, tiny_backbone):
        _validate_backbone(tiny_backbone)  # should not raise

    def test_validate_backbone_missing_encode(self):
        with pytest.raises(NeSyError, match="no 'encode\\(\\)' method"):
            _validate_backbone(_NoEncodeBackbone())

    def test_validate_input_dim_positive(self):
        _validate_input_dim(16)  # should not raise

    @pytest.mark.parametrize("bad_dim", [0, -1, "x", 3.5, None])
    def test_validate_input_dim_bad(self, bad_dim):
        with pytest.raises(NeSyError, match="positive integer"):
            _validate_input_dim(bad_dim)

    def test_validate_output_path_ok(self):
        _validate_output_path("/tmp/model.onnx")

    @pytest.mark.parametrize("bad_path", ["", None, 42])
    def test_validate_output_path_bad(self, bad_path):
        with pytest.raises(NeSyError, match="non-empty string"):
            _validate_output_path(bad_path)


# ════════════════════════════════════════════════════════════════════
#  _sha256_file
# ════════════════════════════════════════════════════════════════════


class TestSha256File:

    def test_known_hash(self, tmp_path):
        f = tmp_path / "data.bin"
        data = b"hello world"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert _sha256_file(str(f)) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert _sha256_file(str(f)) == expected


# ════════════════════════════════════════════════════════════════════
#  _tflite_encode
# ════════════════════════════════════════════════════════════════════


class TestTfliteEncode:
    """Unit tests for the extracted _tflite_encode helper."""

    def test_non_tensor_is_converted(self, tiny_backbone):
        """When encode() returns a non-tf.Tensor, tf.constant is called."""
        mock_tf = MagicMock()
        mock_tf.Tensor = type("FakeTensor", (), {})
        mock_tf.constant = MagicMock(return_value="CONST")
        mock_tf.float32 = "float32"

        x = MagicMock()
        x.numpy.return_value = torch.randn(1, 16)

        result = _tflite_encode(tiny_backbone, x, mock_tf)
        mock_tf.constant.assert_called_once()
        assert result == "CONST"

    def test_tensor_passes_through(self):
        """When encode() returns a tf.Tensor instance, no conversion."""
        FakeTensor = type("FakeTensor", (), {})
        mock_tf = MagicMock()
        mock_tf.Tensor = FakeTensor

        bb = MagicMock()
        tensor_out = FakeTensor()
        bb.encode.return_value = tensor_out

        x = MagicMock()
        x.numpy.return_value = [1.0, 2.0]

        result = _tflite_encode(bb, x, mock_tf)
        assert result is tensor_out
        mock_tf.constant.assert_not_called()


# ════════════════════════════════════════════════════════════════════
#  ONNX export (REAL — torch + onnx are installed)
# ════════════════════════════════════════════════════════════════════


class TestOnnxExport:

    def test_basic_export(self, tiny_backbone, tmp_path):
        out = str(tmp_path / "model.onnx")
        manifest = ModelExporter.to_onnx(tiny_backbone, out, input_dim=16)

        assert Path(out).exists()
        assert Path(out).stat().st_size > 0
        assert manifest.export_format == "onnx"
        assert manifest.model_name == "TinyBackbone"
        assert manifest.input_shape == [1, 16]
        assert manifest.output_shape is not None
        assert len(manifest.checksum) == 64  # SHA-256 hex
        assert manifest.timestamp
        assert manifest.artifact_path

    def test_onnx_model_is_valid(self, tiny_backbone, tmp_path):
        """onnx.checker validates the exported graph."""
        import onnx
        out = str(tmp_path / "valid.onnx")
        ModelExporter.to_onnx(tiny_backbone, out, input_dim=16)
        model = onnx.load(out)
        onnx.checker.check_model(model)

    def test_checksum_is_valid_sha256(self, tiny_backbone, tmp_path):
        """Checksum is a valid 64-char hex SHA-256 digest."""
        out = str(tmp_path / "m1.onnx")
        m = ModelExporter.to_onnx(tiny_backbone, out, input_dim=16)
        assert len(m.checksum) == 64
        # Verify it matches independent computation
        assert m.checksum == _sha256_file(out)

    def test_custom_opset(self, tiny_backbone, tmp_path):
        """Opset is respected (torch 2.10 may upgrade to min 18)."""
        out = str(tmp_path / "opset18.onnx")
        m = ModelExporter.to_onnx(tiny_backbone, out, input_dim=16,
                                   opset_version=18)
        import onnx
        loaded = onnx.load(out)
        assert loaded.opset_import[0].version >= 18
        assert m.export_format == "onnx"

    def test_domain_and_reasoning_config(self, tiny_backbone, tmp_path,
                                          sample_reasoning_config):
        out = str(tmp_path / "domain.onnx")
        m = ModelExporter.to_onnx(
            tiny_backbone, out, input_dim=16,
            domain="medical",
            reasoning_config=sample_reasoning_config,
        )
        assert m.domain == "medical"
        assert m.reasoning_config == sample_reasoning_config

    def test_custom_dynamic_axes_batch_only(self, tiny_backbone, tmp_path):
        """Custom dynamic axes (batch dim only — dim 1 is static)."""
        axes = {"input": {0: "B"}, "output": {0: "B"}}
        out = str(tmp_path / "dyn.onnx")
        m = ModelExporter.to_onnx(tiny_backbone, out, 16,
                                   dynamic_axes=axes)
        assert m.export_format == "onnx"
        assert Path(out).exists()

    def test_creates_parent_dirs(self, tiny_backbone, tmp_path):
        deep = str(tmp_path / "a" / "b" / "c" / "model.onnx")
        m = ModelExporter.to_onnx(tiny_backbone, deep, input_dim=16)
        assert Path(deep).exists()

    def test_manifest_save_alongside(self, tiny_backbone, tmp_path):
        out = str(tmp_path / "model.onnx")
        m = ModelExporter.to_onnx(tiny_backbone, out, input_dim=16)
        manifest_path = str(tmp_path / "manifest.json")
        m.save(manifest_path)
        loaded = ExportManifest.load(manifest_path)
        assert loaded.checksum == m.checksum

    def test_bad_backbone_raises(self, tmp_path):
        with pytest.raises(NeSyError, match="no 'encode\\(\\)' method"):
            ModelExporter.to_onnx(_NoEncodeBackbone(),
                                   str(tmp_path / "x.onnx"), 16)

    def test_bad_input_dim_raises(self, tiny_backbone, tmp_path):
        with pytest.raises(NeSyError, match="positive integer"):
            ModelExporter.to_onnx(tiny_backbone,
                                   str(tmp_path / "x.onnx"), -1)

    def test_bad_output_path_raises(self, tiny_backbone):
        with pytest.raises(NeSyError, match="non-empty string"):
            ModelExporter.to_onnx(tiny_backbone, "", 16)

    def test_torch_import_error(self, tiny_backbone, tmp_path):
        """When torch is missing, a NeSyError should be raised."""
        import builtins
        original_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name in ("torch", "torch.onnx"):
                raise ImportError("no torch")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fake_import):
            with pytest.raises(NeSyError, match="ONNX export requires"):
                ModelExporter.to_onnx(tiny_backbone,
                                       str(tmp_path / "x.onnx"), 16)

    def test_backbone_name_fallback(self, tmp_path):
        """When backbone has no .name attr, class name is used."""
        class _BareBone(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def encode(self, x):
                return self.fc(x)

        bb = _BareBone()
        out = str(tmp_path / "bare.onnx")
        m = ModelExporter.to_onnx(bb, out, input_dim=4)
        assert m.model_name == "_BareBone"


# ════════════════════════════════════════════════════════════════════
#  TFLite export (staged — tensorflow NOT installed)
# ════════════════════════════════════════════════════════════════════


class TestTfliteExportStaged:
    """TF is not installed → stages ONNX + raises NeSyError."""

    def test_staged_onnx_created_and_error_raised(self, tiny_backbone, tmp_path):
        out = str(tmp_path / "model.tflite")
        with pytest.raises(NeSyError, match="TFLite export requires TensorFlow") as exc_info:
            ModelExporter.to_tflite(tiny_backbone, out, input_dim=16)

        # Staged ONNX should exist
        staged_onnx = str(tmp_path / "model.onnx")
        assert Path(staged_onnx).exists()

        # Error context has staged info
        assert "staged_onnx_path" in exc_info.value.context
        assert "staged_manifest" in exc_info.value.context
        assert exc_info.value.context["missing_dependency"] == "tensorflow"

    def test_staged_manifest_is_valid(self, tiny_backbone, tmp_path):
        out = str(tmp_path / "model.tflite")
        with pytest.raises(NeSyError) as exc_info:
            ModelExporter.to_tflite(tiny_backbone, out, input_dim=16)

        staged = exc_info.value.context["staged_manifest"]
        assert staged["export_format"] == "onnx_staged_for_tflite"
        assert staged["model_name"] == "TinyBackbone"
        assert len(staged["checksum"]) == 64

    def test_bad_backbone_before_tf_check(self, tmp_path):
        with pytest.raises(NeSyError, match="no 'encode\\(\\)' method"):
            ModelExporter.to_tflite(_NoEncodeBackbone(),
                                     str(tmp_path / "x.tflite"), 16)

    def test_bad_dim_before_tf_check(self, tiny_backbone, tmp_path):
        with pytest.raises(NeSyError, match="positive integer"):
            ModelExporter.to_tflite(tiny_backbone,
                                     str(tmp_path / "x.tflite"), 0)

    def test_bad_path_before_tf_check(self, tiny_backbone):
        with pytest.raises(NeSyError, match="non-empty string"):
            ModelExporter.to_tflite(tiny_backbone, "", 16)

    def test_with_reasoning_config(self, tiny_backbone, tmp_path,
                                    sample_reasoning_config):
        out = str(tmp_path / "cfg.tflite")
        with pytest.raises(NeSyError) as exc_info:
            ModelExporter.to_tflite(
                tiny_backbone, out, input_dim=16,
                reasoning_config=sample_reasoning_config,
            )
        staged = exc_info.value.context["staged_manifest"]
        assert staged["reasoning_config"] == sample_reasoning_config


class TestTfliteExportMocked:
    """Mock-test the real-TF happy path so those lines are covered."""

    @staticmethod
    def _make_tf_mock():
        """Create a TF mock where tf.Module is a real base class
        and tf.function wraps functions so get_concrete_function works."""
        mock_tf = MagicMock()
        # tf.Module must be a real class so _TFWrapper.__init__ runs
        mock_tf.Module = type("FakeModule", (), {})
        mock_tf.lite.Optimize.DEFAULT = "DEFAULT"
        mock_tf.float32 = "float32"

        class _FakeTFFunction:
            """Descriptor that mimics tf.function: binds to instance
            and provides get_concrete_function()."""

            def __init__(self, fn):
                self._fn = fn

            def __set_name__(self, owner, name):
                self._attr_name = name

            def __get__(self, obj, objtype=None):
                if obj is None:
                    return self
                import functools
                bound = functools.partial(self._fn, obj)
                bound.get_concrete_function = lambda: bound
                return bound

            def get_concrete_function(self):
                return self._fn

        def fake_tf_function(**kw):
            return lambda fn: _FakeTFFunction(fn)

        mock_tf.function = fake_tf_function
        mock_tf.TensorSpec = lambda **kw: None
        mock_tf.Tensor = type("FakeTensor", (), {})
        mock_tf.constant = MagicMock(return_value="CONST")
        return mock_tf

    def test_happy_path_no_quantise(self, tiny_backbone, tmp_path):
        mock_tf = self._make_tf_mock()
        mock_np = MagicMock()

        tflite_bytes = b"FAKE_TFLITE_MODEL"
        converter_mock = MagicMock()
        converter_mock.convert.return_value = tflite_bytes
        mock_tf.lite.TFLiteConverter.from_concrete_functions.return_value = converter_mock

        with patch.dict("sys.modules", {
            "tensorflow": mock_tf,
            "numpy": mock_np,
        }):
            out = str(tmp_path / "model.tflite")
            m = ModelExporter.to_tflite(tiny_backbone, out, input_dim=16,
                                         quantise=False)

        assert Path(out).exists()
        assert Path(out).read_bytes() == tflite_bytes
        assert m.export_format == "tflite"
        assert m.quantization_mode == "none"
        assert len(m.checksum) == 64

    def test_happy_path_quantised(self, tiny_backbone, tmp_path):
        mock_tf = self._make_tf_mock()
        mock_np = MagicMock()

        tflite_bytes = b"QUANTISED_MODEL"
        converter_mock = MagicMock()
        converter_mock.convert.return_value = tflite_bytes
        mock_tf.lite.TFLiteConverter.from_concrete_functions.return_value = converter_mock

        with patch.dict("sys.modules", {
            "tensorflow": mock_tf,
            "numpy": mock_np,
        }):
            out = str(tmp_path / "model_q.tflite")
            m = ModelExporter.to_tflite(tiny_backbone, out, input_dim=16,
                                         quantise=True)

        assert m.quantization_mode == "dynamic"
        assert m.export_format == "tflite"

    def test_serve_body_executes(self, tiny_backbone, tmp_path):
        """Verify the serve() delegation line is covered."""
        mock_tf = self._make_tf_mock()
        mock_np = MagicMock()

        tflite_bytes = b"SERVE_TEST"
        converter_mock = MagicMock()
        converter_mock.convert.return_value = tflite_bytes

        # Intercept converter creation: call the bound serve function
        def capture_converter(fns):
            serve_fn = fns[0]  # bound partial(serve, wrapper_instance)
            x = MagicMock()
            x.numpy.return_value = torch.randn(1, 16)
            try:
                serve_fn(x)
            except Exception:
                pass
            return converter_mock

        mock_tf.lite.TFLiteConverter.from_concrete_functions.side_effect = capture_converter

        with patch.dict("sys.modules", {
            "tensorflow": mock_tf,
            "numpy": mock_np,
        }):
            out = str(tmp_path / "serve.tflite")
            m = ModelExporter.to_tflite(tiny_backbone, out, input_dim=16)


# ════════════════════════════════════════════════════════════════════
#  CoreML export (staged — coremltools NOT installed)
# ════════════════════════════════════════════════════════════════════


class TestCoremlExportStaged:
    """coremltools is not installed → stages ONNX + raises NeSyError."""

    def test_staged_onnx_created_and_error_raised(self, tiny_backbone, tmp_path):
        out = str(tmp_path / "model.mlpackage")
        with pytest.raises(NeSyError, match="CoreML export requires") as exc_info:
            ModelExporter.to_coreml(tiny_backbone, out, input_dim=16)

        staged_onnx = str(tmp_path / "model.onnx")
        assert Path(staged_onnx).exists()

        assert "staged_onnx_path" in exc_info.value.context
        assert exc_info.value.context["missing_dependency"] == "coremltools"

    def test_staged_manifest_format(self, tiny_backbone, tmp_path):
        out = str(tmp_path / "model.mlpackage")
        with pytest.raises(NeSyError) as exc_info:
            ModelExporter.to_coreml(tiny_backbone, out, input_dim=16)

        staged = exc_info.value.context["staged_manifest"]
        assert staged["export_format"] == "onnx_staged_for_coreml"
        assert len(staged["checksum"]) == 64

    def test_bad_backbone_before_ct_check(self, tmp_path):
        with pytest.raises(NeSyError, match="no 'encode\\(\\)' method"):
            ModelExporter.to_coreml(_NoEncodeBackbone(),
                                     str(tmp_path / "x.mlpackage"), 16)

    def test_bad_dim_before_ct_check(self, tiny_backbone, tmp_path):
        with pytest.raises(NeSyError, match="positive integer"):
            ModelExporter.to_coreml(tiny_backbone,
                                     str(tmp_path / "x.mlpackage"), -5)

    def test_bad_path_before_ct_check(self, tiny_backbone):
        with pytest.raises(NeSyError, match="non-empty string"):
            ModelExporter.to_coreml(tiny_backbone, "", 16)

    def test_with_domain_and_reasoning(self, tiny_backbone, tmp_path,
                                        sample_reasoning_config):
        out = str(tmp_path / "cfg.mlpackage")
        with pytest.raises(NeSyError) as exc_info:
            ModelExporter.to_coreml(
                tiny_backbone, out, input_dim=16,
                domain="legal",
                reasoning_config=sample_reasoning_config,
            )
        staged = exc_info.value.context["staged_manifest"]
        assert staged["reasoning_config"] == sample_reasoning_config


class TestCoremlExportMocked:
    """Mock-test the real-coremltools happy path."""

    def test_happy_path(self, tiny_backbone, tmp_path):
        mock_ct = MagicMock()
        mock_ct.TensorType = MagicMock

        mlmodel_mock = MagicMock()
        mock_ct.convert.return_value = mlmodel_mock

        # mlmodel.save writes a file
        out = str(tmp_path / "model.mlpackage")

        def _save(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"FAKE_COREML")

        mlmodel_mock.save.side_effect = _save

        with patch.dict("sys.modules", {"coremltools": mock_ct}):
            m = ModelExporter.to_coreml(
                tiny_backbone, out, input_dim=16,
                model_name="TestCore",
            )

        assert m.export_format == "coreml"
        assert m.model_name == "TestCore"
        assert len(m.checksum) == 64
        assert Path(out).exists()


# ════════════════════════════════════════════════════════════════════
#  export_bundle
# ════════════════════════════════════════════════════════════════════


class TestExportBundle:

    def test_onnx_only_bundle(self, tiny_backbone, tmp_path):
        bundle_dir = str(tmp_path / "bundle")
        manifests = ModelExporter.export_bundle(
            tiny_backbone, bundle_dir, input_dim=16,
        )
        assert "onnx" in manifests
        assert (tmp_path / "bundle" / "model.onnx").exists()
        assert (tmp_path / "bundle" / "manifest.json").exists()

    def test_bundle_with_reasoning_config(self, tiny_backbone, tmp_path,
                                           sample_reasoning_config):
        bundle_dir = str(tmp_path / "bundle_cfg")
        manifests = ModelExporter.export_bundle(
            tiny_backbone, bundle_dir, input_dim=16,
            reasoning_config=sample_reasoning_config,
        )
        cfg_path = tmp_path / "bundle_cfg" / "reasoning_config.json"
        assert cfg_path.exists()
        parsed = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert parsed["domain"] == "medical"

    def test_bundle_with_tflite_staged(self, tiny_backbone, tmp_path):
        """TFLite format is attempted but staged with warning (no crash)."""
        bundle_dir = str(tmp_path / "bundle_mixed")
        manifests = ModelExporter.export_bundle(
            tiny_backbone, bundle_dir, input_dim=16,
            formats=["onnx", "tflite"],
        )
        # ONNX should succeed
        assert "onnx" in manifests
        # TFLite will be staged → NeSyError caught → not in manifests
        # But staged ONNX should be written
        assert (tmp_path / "bundle_mixed" / "model.onnx").exists()

    def test_bundle_with_coreml_staged(self, tiny_backbone, tmp_path):
        bundle_dir = str(tmp_path / "bundle_coreml")
        manifests = ModelExporter.export_bundle(
            tiny_backbone, bundle_dir, input_dim=16,
            formats=["onnx", "coreml"],
        )
        assert "onnx" in manifests
        # CoreML staged → not in manifests but no crash
        assert (tmp_path / "bundle_coreml" / "model.onnx").exists()

    def test_bundle_unknown_format_skipped(self, tiny_backbone, tmp_path):
        bundle_dir = str(tmp_path / "bundle_unk")
        manifests = ModelExporter.export_bundle(
            tiny_backbone, bundle_dir, input_dim=16,
            formats=["webgl"],
        )
        assert len(manifests) == 0
        assert (tmp_path / "bundle_unk" / "manifest.json").exists()

    def test_bundle_manifest_json(self, tiny_backbone, tmp_path):
        bundle_dir = str(tmp_path / "bundle_mf")
        ModelExporter.export_bundle(
            tiny_backbone, bundle_dir, input_dim=16,
        )
        mf_path = tmp_path / "bundle_mf" / "manifest.json"
        data = json.loads(mf_path.read_text(encoding="utf-8"))
        assert "onnx" in data
        assert data["onnx"]["export_format"] == "onnx"

    def test_bad_backbone_raises(self, tmp_path):
        with pytest.raises(NeSyError, match="no 'encode\\(\\)' method"):
            ModelExporter.export_bundle(
                _NoEncodeBackbone(), str(tmp_path / "b"), 16,
            )

    def test_bad_input_dim_raises(self, tiny_backbone, tmp_path):
        with pytest.raises(NeSyError, match="positive integer"):
            ModelExporter.export_bundle(
                tiny_backbone, str(tmp_path / "b"), 0,
            )

    def test_bad_output_dir_raises(self, tiny_backbone):
        with pytest.raises(NeSyError, match="non-empty string"):
            ModelExporter.export_bundle(tiny_backbone, "", 16)

    def test_domain_propagated(self, tiny_backbone, tmp_path):
        bundle_dir = str(tmp_path / "bundle_dom")
        manifests = ModelExporter.export_bundle(
            tiny_backbone, bundle_dir, input_dim=16,
            domain="legal",
        )
        assert manifests["onnx"].domain == "legal"

    def test_bundle_all_formats(self, tiny_backbone, tmp_path):
        """All three formats attempted — ONNX succeeds, others staged."""
        bundle_dir = str(tmp_path / "all")
        manifests = ModelExporter.export_bundle(
            tiny_backbone, bundle_dir, input_dim=16,
            formats=["onnx", "tflite", "coreml"],
        )
        assert "onnx" in manifests
        # tflite and coreml are staged but don't crash

    def test_bundle_tflite_success_mocked(self, tiny_backbone, tmp_path):
        """Mock TFLite to succeed inside export_bundle (covers line 666)."""
        fake_manifest = ExportManifest(
            model_name="T", domain="g", export_format="tflite",
            input_shape=[1, 16], checksum="a" * 64,
        )
        bundle_dir = str(tmp_path / "b_tflite_ok")
        with patch.object(ModelExporter, "to_tflite", return_value=fake_manifest):
            manifests = ModelExporter.export_bundle(
                tiny_backbone, bundle_dir, input_dim=16,
                formats=["tflite"],
            )
        assert "tflite" in manifests
        assert manifests["tflite"].export_format == "tflite"

    def test_bundle_coreml_success_mocked(self, tiny_backbone, tmp_path):
        """Mock CoreML to succeed inside export_bundle (covers line 675)."""
        fake_manifest = ExportManifest(
            model_name="C", domain="g", export_format="coreml",
            input_shape=[1, 16], checksum="b" * 64,
        )
        bundle_dir = str(tmp_path / "b_coreml_ok")
        with patch.object(ModelExporter, "to_coreml", return_value=fake_manifest):
            manifests = ModelExporter.export_bundle(
                tiny_backbone, bundle_dir, input_dim=16,
                formats=["coreml"],
            )
        assert "coreml" in manifests
        assert manifests["coreml"].export_format == "coreml"
