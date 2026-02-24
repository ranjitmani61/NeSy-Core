"""nesy/metacognition — Meta-cognition and self-doubt layer."""

from nesy.metacognition.monitor import MetaCognitionMonitor
from nesy.metacognition.confidence import (
    compute_factual,
    compute_reasoning,
    compute_boundary,
    build_confidence_report,
)
from nesy.metacognition.doubt import SelfDoubtLayer
from nesy.metacognition.trace import TraceBuilder
from nesy.metacognition.calibration import ConfidenceCalibrator
from .shadow import CounterfactualShadowEngine, shadow_flags
from nesy.metacognition.fingerprint import (
    canonicalize_output,
    compute_config_hash,
    compute_reasoning_fingerprint,
)

__all__ = [
    "MetaCognitionMonitor",
    "SelfDoubtLayer",
    "TraceBuilder",
    "ConfidenceCalibrator",
    "compute_factual",
    "compute_reasoning",
    "compute_boundary",
    "build_confidence_report",
    "canonicalize_output",
    "compute_config_hash",
    "compute_reasoning_fingerprint",
    "CounterfactualShadowEngine",
    "shadow_flags",
]
