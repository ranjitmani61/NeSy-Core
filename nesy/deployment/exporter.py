"""
nesy/deployment/exporter.py
============================
Export NeSy neural backbones to ONNX, TFLite, CoreML for edge deployment.

Architecture:
    Only the neural backbone (torch.nn.Module) is exported.
    The symbolic engine, concept graph, and metacognition monitor run
    on CPU independently and are packaged as a reasoning-config JSON
    sidecar via ``ExportManifest``.

Export contract:
    Every export produces:
        1. A model artifact (file on disk).
        2. An ``ExportManifest`` (JSON-serialisable) containing:
           model_name, domain, export_format, input_shape, output_shape,
           quantization_mode, checksum, timestamp.

Dependency handling:
    ONNX   — requires ``torch`` + ``onnx``  (both installed).
    TFLite — requires ``tensorflow`` (NOT installed → staged ONNX + typed error).
    CoreML — requires ``coremltools`` + ``torch``
             (NOT installed → staged ONNX + typed error).

    When a runtime is unavailable the exporter stages an ONNX artifact
    and raises ``NeSyError`` with explicit installation instructions.
    No fake conversion, no placeholder success.

Mathematical basis for checksum:
    SHA-256 over the produced artifact bytes guarantees tamper detection
    with collision probability < 2⁻¹²⁸ (birthday bound on 256-bit hash).

Reference:
    ONNX IR spec — https://onnx.ai/onnx/intro/concepts.html
    torch.onnx — https://pytorch.org/docs/stable/onnx.html
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from nesy.core.exceptions import NeSyError

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Export Manifest
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ExportManifest:
    """JSON-serialisable manifest attached to every model export.

    Contains provenance, shape, and integrity information so that
    the edge runtime can verify the artifact before loading.

    Attributes:
        model_name:        Human-readable name (backbone class name).
        domain:            NeSy domain string (``"medical"``, ``"general"``, …).
        export_format:     One of ``"onnx"``, ``"tflite"``, ``"coreml"``.
        input_shape:       Expected input tensor shape, e.g. ``[1, 128]``.
        output_shape:      Output tensor shape (may be ``None`` if dynamic).
        quantization_mode: ``"none"`` | ``"dynamic"`` | ``"int8"`` | …
        checksum:          SHA-256 hex digest of the artifact bytes.
        timestamp:         ISO-8601 UTC timestamp of export.
        reasoning_config:  Optional NeSy reasoning-config sidecar dict.
        artifact_path:     Filesystem path to the exported artifact.
    """

    model_name: str
    domain: str
    export_format: str
    input_shape: List[int]
    output_shape: Optional[List[int]] = None
    quantization_mode: str = "none"
    checksum: str = ""
    timestamp: str = ""
    reasoning_config: Optional[Dict[str, Any]] = None
    artifact_path: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for ``json.dumps()``."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Write manifest JSON alongside the artifact.

        Args:
            path: Filesystem path for the ``.json`` manifest.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Manifest saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "ExportManifest":
        """Load a manifest from a JSON file.

        Args:
            path: Filesystem path to the manifest JSON.

        Returns:
            Populated ``ExportManifest`` instance.

        Raises:
            NeSyError: if the file cannot be read or parsed.
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise NeSyError(
                f"Cannot load export manifest from '{path}': {exc}",
                context={"path": path},
            )
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════
#  Utility helpers
# ═══════════════════════════════════════════════════════════════════


def _sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest of file at *path*.

    Reads in 64 KiB chunks so large models do not require
    full in-memory buffering.

    Args:
        path: Filesystem path to hash.

    Returns:
        Lowercase hex digest string (64 chars).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65_536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _tflite_encode(backbone: Any, x: Any, tf: Any) -> Any:
    """Run backbone.encode() and coerce the result to a tf.Tensor.

    Separated from the ``_TFWrapper`` inner class so it can be
    unit-tested without a real TensorFlow runtime.

    Args:
        backbone: NeSyBackbone instance.
        x:        TF tensor or mock with ``.numpy()``.
        tf:       The ``tensorflow`` module (or mock).

    Returns:
        A ``tf.Tensor`` (or mock equivalent).
    """
    result = backbone.encode(x.numpy())
    if not isinstance(result, tf.Tensor):
        result = tf.constant(result, dtype=tf.float32)
    return result


def _validate_backbone(backbone: Any) -> None:
    """Ensure *backbone* has an ``encode()`` method.

    Args:
        backbone: Object to validate.

    Raises:
        NeSyError: if the backbone lacks ``encode()``.
    """
    if not hasattr(backbone, "encode"):
        raise NeSyError(
            f"Backbone of type '{type(backbone).__name__}' has no "
            "'encode()' method.  All NeSy backbones must implement "
            "NeSyBackbone.encode().",
            context={"backbone_type": type(backbone).__name__},
        )


def _validate_input_dim(input_dim: int) -> None:
    """Ensure *input_dim* is a positive integer.

    Args:
        input_dim: Feature dimension to validate.

    Raises:
        NeSyError: if input_dim is not a positive integer.
    """
    if not isinstance(input_dim, int) or input_dim <= 0:
        raise NeSyError(
            f"input_dim must be a positive integer, got {input_dim!r}.",
            context={"input_dim": input_dim},
        )


def _validate_output_path(output_path: str) -> None:
    """Ensure *output_path* is a non-empty string.

    Args:
        output_path: Path to validate.

    Raises:
        NeSyError: if output_path is empty.
    """
    if not output_path or not isinstance(output_path, str):
        raise NeSyError(
            "output_path must be a non-empty string.",
            context={"output_path": output_path},
        )


# ═══════════════════════════════════════════════════════════════════
#  ModelExporter
# ═══════════════════════════════════════════════════════════════════


class ModelExporter:
    """Export neural backbone to ONNX / TFLite / CoreML.

    Only the neural backbone is exported.  The symbolic engine,
    concept graph, and metacognition monitor run on CPU independently
    of the exported model.

    Every public method returns an ``ExportManifest`` for provenance
    tracking and integrity verification.

    Usage::

        from nesy.deployment.exporter import ModelExporter

        manifest = ModelExporter.to_onnx(backbone, "out/model.onnx", 128)
        print(manifest.checksum)      # SHA-256 of model.onnx
        print(manifest.to_dict())     # JSON-serialisable
    """

    # ─── ONNX ──────────────────────────────────────────────────────

    @staticmethod
    def to_onnx(
        backbone: Any,
        output_path: str,
        input_dim: int,
        opset_version: int = 13,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        domain: str = "general",
        reasoning_config: Optional[Dict[str, Any]] = None,
    ) -> ExportManifest:
        """Export backbone to ONNX format.

        Requires: ``pip install torch onnx``

        The backbone must expose an ``encode(x) → Tensor`` method.
        A thin ``torch.nn.Module`` wrapper is built automatically so
        that ``torch.onnx.export`` receives a proper module.

        Algorithm:
            1. Validate inputs (backbone, input_dim, output_path).
            2. Build ``_BackboneWrapper(torch.nn.Module)`` delegating
               ``forward()`` → ``backbone.encode()``.
            3. Create dummy input ``torch.randn(1, input_dim)``.
            4. Call ``torch.onnx.export()`` with tracking names.
            5. Compute SHA-256 checksum of produced file.
            6. Build and return ``ExportManifest``.

        Args:
            backbone:      NeSyBackbone instance with ``encode()``.
            output_path:   Destination ``.onnx`` file path.
            input_dim:     Feature dimension of the input tensor.
            opset_version: ONNX opset (default 13, range 7–21).
            dynamic_axes:  Optional dynamic-axis mapping.
            domain:        NeSy domain label for manifest.
            reasoning_config: Optional NeSy reasoning sidecar dict.

        Returns:
            ``ExportManifest`` with SHA-256 checksum and metadata.

        Raises:
            NeSyError: if backbone lacks ``encode()`` or input_dim invalid.
            NeSyError: if torch / onnx are not installed.
        """
        # ── Validation ────────────────────────────────────────────
        _validate_backbone(backbone)
        _validate_input_dim(input_dim)
        _validate_output_path(output_path)

        try:
            import torch
            import torch.onnx
        except ImportError:
            raise NeSyError(
                "ONNX export requires PyTorch and ONNX. Install with: pip install torch onnx",
                context={"missing_dependency": "torch/onnx"},
            )

        # ── Wrapper module ────────────────────────────────────────
        class _BackboneWrapper(torch.nn.Module):
            """Wraps backbone.encode() as a torch.nn.Module."""

            def __init__(self, bb: Any) -> None:
                super().__init__()
                self._bb = bb
                if hasattr(bb, "named_parameters"):
                    for pname, param in bb.named_parameters():
                        self.register_parameter(pname.replace(".", "_"), param)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                return self._bb.encode(x)

        wrapper = _BackboneWrapper(backbone)
        wrapper.eval()

        dummy_input = torch.randn(1, input_dim)

        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*dynamic_axes.*is not recommended.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*from_dynamic_axes_to_dynamic_shapes.*",
                category=DeprecationWarning,
            )
            torch.onnx.export(
                wrapper,
                dummy_input,
                output_path,
                opset_version=opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )

        # ── Detect output shape ───────────────────────────────────
        with torch.no_grad():
            sample_out = wrapper(dummy_input)
        output_shape: Optional[List[int]] = list(sample_out.shape)

        # ── Checksum + manifest ───────────────────────────────────
        checksum = _sha256_file(output_path)

        manifest = ExportManifest(
            model_name=getattr(backbone, "name", type(backbone).__name__),
            domain=domain,
            export_format="onnx",
            input_shape=[1, input_dim],
            output_shape=output_shape,
            quantization_mode="none",
            checksum=checksum,
            reasoning_config=reasoning_config,
            artifact_path=str(Path(output_path).resolve()),
        )

        logger.info("ONNX export complete → %s  (SHA-256: %s)", output_path, checksum)
        return manifest

    # ─── TFLITE ────────────────────────────────────────────────────

    @staticmethod
    def to_tflite(
        backbone: Any,
        output_path: str,
        input_dim: int = 128,
        quantise: bool = False,
        domain: str = "general",
        reasoning_config: Optional[Dict[str, Any]] = None,
    ) -> ExportManifest:
        """Export backbone to TensorFlow Lite format.

        Strategy (best-effort):
            If ``tensorflow`` is importable, build a ``tf.Module``
            wrapping ``backbone.encode()``, convert via
            ``tf.lite.TFLiteConverter``, and optionally apply
            post-training dynamic-range quantisation.

            If ``tensorflow`` is **not** installed, the exporter
            stages an ONNX artifact at the same directory and raises
            ``NeSyError`` with explicit instructions for converting
            ONNX → TFLite using ``onnx-tf`` or TF CLI.

        Args:
            backbone:    NeSyBackbone instance with ``encode()``.
            output_path: Destination ``.tflite`` file path.
            input_dim:   Input feature dimension.
            quantise:    If True, apply dynamic-range quantisation.
            domain:      NeSy domain label for manifest.
            reasoning_config: Optional reasoning sidecar dict.

        Returns:
            ``ExportManifest`` with checksum and metadata.

        Raises:
            NeSyError: if TensorFlow is not available (stages ONNX first).
            NeSyError: if backbone is invalid.
        """
        _validate_backbone(backbone)
        _validate_input_dim(input_dim)
        _validate_output_path(output_path)

        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            # Stage an ONNX artifact, then raise typed error
            onnx_path = str(Path(output_path).with_suffix(".onnx"))
            logger.info(
                "TensorFlow not available — staging ONNX artifact at %s",
                onnx_path,
            )
            manifest = ModelExporter.to_onnx(
                backbone,
                onnx_path,
                input_dim,
                domain=domain,
                reasoning_config=reasoning_config,
            )
            manifest.export_format = "onnx_staged_for_tflite"
            raise NeSyError(
                "TFLite export requires TensorFlow. A staged ONNX "
                f"artifact has been written to '{onnx_path}'. "
                "Convert it with: pip install tensorflow onnx-tf && "
                "onnx-tf convert -i model.onnx -o model_tf && "
                "tflite_convert --saved_model_dir=model_tf "
                "--output_file=model.tflite",
                context={
                    "staged_onnx_path": onnx_path,
                    "staged_manifest": manifest.to_dict(),
                    "missing_dependency": "tensorflow",
                },
            )

        # ── Real TF Lite conversion ───────────────────────────────

        class _TFWrapper(tf.Module):
            def __init__(self, bb: Any) -> None:
                super().__init__()
                self._bb = bb

            @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32)])
            def serve(self, x: Any) -> Any:
                return _tflite_encode(self._bb, x, tf)

        wrapper = _TFWrapper(backbone)
        concrete_fn = wrapper.serve.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])

        quant_mode = "none"
        if quantise:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            quant_mode = "dynamic"

        tflite_bytes = converter.convert()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(tflite_bytes)

        checksum = _sha256_file(output_path)

        manifest = ExportManifest(
            model_name=getattr(backbone, "name", type(backbone).__name__),
            domain=domain,
            export_format="tflite",
            input_shape=[1, input_dim],
            quantization_mode=quant_mode,
            checksum=checksum,
            reasoning_config=reasoning_config,
            artifact_path=str(Path(output_path).resolve()),
        )

        logger.info(
            "TFLite export complete → %s%s  (SHA-256: %s)",
            output_path,
            " (quantised)" if quantise else "",
            checksum,
        )
        return manifest

    # ─── COREML ────────────────────────────────────────────────────

    @staticmethod
    def to_coreml(
        backbone: Any,
        output_path: str,
        input_dim: int = 128,
        model_name: str = "NeSyBackbone",
        domain: str = "general",
        reasoning_config: Optional[Dict[str, Any]] = None,
    ) -> ExportManifest:
        """Export backbone to Apple CoreML format.

        Strategy (best-effort):
            If ``coremltools`` + ``torch`` are importable, trace the
            backbone and convert via ``ct.convert()``.

            Otherwise, stage an ONNX artifact and raise ``NeSyError``
            with instructions for ``coremltools`` conversion from ONNX.

        Args:
            backbone:    NeSyBackbone instance with ``encode()``.
            output_path: Destination ``.mlmodel`` or ``.mlpackage`` path.
            input_dim:   Input feature dimension.
            model_name:  Human-readable label in CoreML metadata.
            domain:      NeSy domain label for manifest.
            reasoning_config: Optional reasoning sidecar dict.

        Returns:
            ``ExportManifest`` with checksum and metadata.

        Raises:
            NeSyError: if coremltools / torch are not available
                      (stages ONNX first).
            NeSyError: if backbone is invalid.
        """
        _validate_backbone(backbone)
        _validate_input_dim(input_dim)
        _validate_output_path(output_path)

        try:
            import coremltools as ct  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            onnx_path = str(Path(output_path).with_suffix(".onnx"))
            logger.info(
                "coremltools not available — staging ONNX artifact at %s",
                onnx_path,
            )
            manifest = ModelExporter.to_onnx(
                backbone,
                onnx_path,
                input_dim,
                domain=domain,
                reasoning_config=reasoning_config,
            )
            manifest.export_format = "onnx_staged_for_coreml"
            raise NeSyError(
                "CoreML export requires coremltools and PyTorch. "
                f"A staged ONNX artifact has been written to '{onnx_path}'. "
                "Convert it with: pip install coremltools && "
                'python -c "import coremltools as ct; '
                "m = ct.converters.onnx.convert(model='model.onnx'); "
                "m.save('model.mlpackage')\"",
                context={
                    "staged_onnx_path": onnx_path,
                    "staged_manifest": manifest.to_dict(),
                    "missing_dependency": "coremltools",
                },
            )

        # ── Real CoreML conversion ────────────────────────────────
        class _TorchWrapper(torch.nn.Module):
            def __init__(self, bb: Any) -> None:
                super().__init__()
                self._bb = bb

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                return self._bb.encode(x)

        wrapper = _TorchWrapper(backbone)
        wrapper.eval()

        dummy_input = torch.randn(1, input_dim)
        traced = torch.jit.trace(wrapper, dummy_input)

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=(1, input_dim))],
        )
        mlmodel.short_description = f"{model_name} — NeSy-Core neural backbone"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(output_path)

        checksum = _sha256_file(output_path)

        manifest = ExportManifest(
            model_name=model_name,
            domain=domain,
            export_format="coreml",
            input_shape=[1, input_dim],
            quantization_mode="none",
            checksum=checksum,
            reasoning_config=reasoning_config,
            artifact_path=str(Path(output_path).resolve()),
        )

        logger.info("CoreML export complete → %s  (SHA-256: %s)", output_path, checksum)
        return manifest

    # ─── BUNDLE EXPORT ─────────────────────────────────────────────

    @staticmethod
    def export_bundle(
        backbone: Any,
        output_dir: str,
        input_dim: int,
        domain: str = "general",
        reasoning_config: Optional[Dict[str, Any]] = None,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, ExportManifest]:
        """Export backbone to multiple formats and write a manifest bundle.

        Produces one directory containing:
            - ``model.onnx``           (always)
            - ``model.tflite``         (if tensorflow available)
            - ``model.mlpackage``      (if coremltools available)
            - ``manifest.json``        (combined manifest)
            - ``reasoning_config.json`` (if provided)

        Each target format that lacks its runtime produces a staged
        ONNX and a warning — no crash, no placeholder success.

        Args:
            backbone:         NeSyBackbone instance.
            output_dir:       Directory for the export bundle.
            input_dim:        Feature dimension.
            domain:           NeSy domain label.
            reasoning_config: Optional reasoning sidecar dict.
            formats:          List of formats to attempt.
                              Default: ``["onnx"]``.

        Returns:
            Dict mapping format name to its ``ExportManifest``.
        """
        _validate_backbone(backbone)
        _validate_input_dim(input_dim)

        if not output_dir or not isinstance(output_dir, str):
            raise NeSyError(
                "output_dir must be a non-empty string.",
                context={"output_dir": output_dir},
            )

        if formats is None:
            formats = ["onnx"]

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        manifests: Dict[str, ExportManifest] = {}

        for fmt in formats:
            fmt_lower = fmt.lower().strip()
            try:
                if fmt_lower == "onnx":
                    m = ModelExporter.to_onnx(
                        backbone,
                        str(Path(output_dir) / "model.onnx"),
                        input_dim,
                        domain=domain,
                        reasoning_config=reasoning_config,
                    )
                    manifests["onnx"] = m
                elif fmt_lower == "tflite":
                    m = ModelExporter.to_tflite(
                        backbone,
                        str(Path(output_dir) / "model.tflite"),
                        input_dim,
                        domain=domain,
                        reasoning_config=reasoning_config,
                    )
                    manifests["tflite"] = m
                elif fmt_lower == "coreml":
                    m = ModelExporter.to_coreml(
                        backbone,
                        str(Path(output_dir) / "model.mlpackage"),
                        input_dim,
                        domain=domain,
                        reasoning_config=reasoning_config,
                    )
                    manifests["coreml"] = m
                else:
                    logger.warning("Unknown export format '%s' — skipping.", fmt)
            except NeSyError as exc:
                logger.warning(
                    "Format '%s' staged only (dependency missing): %s",
                    fmt_lower,
                    exc,
                )

        # Write reasoning config sidecar
        if reasoning_config is not None:
            cfg_path = str(Path(output_dir) / "reasoning_config.json")
            Path(cfg_path).write_text(
                json.dumps(reasoning_config, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("Reasoning config written → %s", cfg_path)

        # Write combined bundle manifest
        bundle_manifest_path = str(Path(output_dir) / "manifest.json")
        bundle_data = {fmt: m.to_dict() for fmt, m in manifests.items()}
        Path(bundle_manifest_path).write_text(
            json.dumps(bundle_data, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(
            "Bundle export complete → %s  (%d format(s))",
            output_dir,
            len(manifests),
        )

        return manifests
