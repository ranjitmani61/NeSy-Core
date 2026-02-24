"""
nesy/deployment/npu.py
=======================
NPU-optimised inference path for NeSy-Core.

NPU (Neural Processing Unit) chips have specific constraints:
    - Fixed quantization (INT8 or INT4)
    - Limited memory bandwidth
    - Batch size restrictions
    - No dynamic graph support

NeSy-Core's symbolic layer runs on CPU (it's pure Python logic).
The neural backbone is the NPU target.

This module handles the handoff: NPU for embedding, CPU for reasoning.
"""

from __future__ import annotations
import logging
import time
from typing import Any, List, Optional
from nesy.neural.base import NeSyBackbone

logger = logging.getLogger(__name__)


class NPUBackboneWrapper(NeSyBackbone):
    """Wraps any NeSyBackbone with NPU-aware batching and timing.

    In production, replace the _run_on_npu method with
    the vendor-specific NPU SDK call (Qualcomm QNN, Apple ANE, etc.)
    """

    def __init__(self, base_backbone: NeSyBackbone, target_latency_ms: int = 50):
        self._base = base_backbone
        self.target_latency = target_latency_ms
        self._latency_log: List[float] = []

    def encode(self, input_data: Any) -> List[float]:
        t0 = time.perf_counter()
        result = self._run_on_npu(input_data)
        latency_ms = (time.perf_counter() - t0) * 1000
        self._latency_log.append(latency_ms)
        if latency_ms > self.target_latency:
            logger.warning(f"NPU latency {latency_ms:.1f}ms > target {self.target_latency}ms")
        return result

    def _run_on_npu(self, input_data: Any) -> List[float]:
        """NPU inference. Replace with vendor SDK in production.
        Currently delegates to base backbone (CPU fallback)."""
        return self._base.encode(input_data)

    def confidence(self, embedding: List[float]) -> float:
        return self._base.confidence(embedding)

    @property
    def embedding_dim(self) -> int:
        return self._base.embedding_dim

    @property
    def name(self) -> str:
        return f"npu:{self._base.name}"

    @property
    def avg_latency_ms(self) -> Optional[float]:
        if not self._latency_log:
            return None
        return sum(self._latency_log) / len(self._latency_log)

    @property
    def p95_latency_ms(self) -> Optional[float]:
        if not self._latency_log:
            return None
        sorted_log = sorted(self._latency_log)
        idx = int(0.95 * len(sorted_log))
        return sorted_log[idx]
