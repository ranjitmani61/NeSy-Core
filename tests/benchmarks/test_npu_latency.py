"""
tests/benchmarks/test_npu_latency.py
======================================
Latency benchmarks for the NPU inference path.

Target: < 200 ms per inference on NPU hardware.
Uses CPU fallback for CI (NPUBackboneWrapper delegates to base backbone).
"""

import time
import pytest
from nesy.deployment.npu import NPUBackboneWrapper
from nesy.neural.base import PassthroughBackbone


def _emb(dim: int = 128) -> list:
    """Deterministic fake embedding."""
    return [float(i % 7) / 7.0 for i in range(dim)]


@pytest.fixture
def npu_backbone():
    base = PassthroughBackbone(dim=128)
    return NPUBackboneWrapper(base, target_latency_ms=200)


class TestNPULatency:
    def test_single_encode_under_200ms(self, npu_backbone):
        """Single NPU encode must complete within 200 ms target."""
        t0 = time.perf_counter()
        result = npu_backbone.encode(_emb(128))
        latency_ms = (time.perf_counter() - t0) * 1000
        assert latency_ms < 200.0, f"NPU latency {latency_ms:.1f} ms > 200 ms target"
        assert len(result) == 128

    def test_batch_10_under_500ms(self, npu_backbone):
        """10 sequential encodes must complete within 500 ms."""
        t0 = time.perf_counter()
        for _ in range(10):
            npu_backbone.encode(_emb(128))
        total_ms = (time.perf_counter() - t0) * 1000
        assert total_ms < 500.0, f"10 encodes took {total_ms:.1f} ms > 500 ms target"

    def test_avg_latency_tracking(self, npu_backbone):
        """avg_latency_ms property must reflect actual measurements."""
        for _ in range(5):
            npu_backbone.encode(_emb(128))
        avg = npu_backbone.avg_latency_ms
        assert avg is not None
        assert avg >= 0.0

    def test_p95_latency_tracking(self, npu_backbone):
        """p95_latency_ms property must be available after measurements."""
        for _ in range(20):
            npu_backbone.encode(_emb(128))
        p95 = npu_backbone.p95_latency_ms
        assert p95 is not None
        assert p95 >= 0.0

    def test_latency_log_grows(self, npu_backbone):
        """Each encode call should add an entry to the latency log."""
        assert len(npu_backbone._latency_log) == 0
        npu_backbone.encode(_emb(128))
        assert len(npu_backbone._latency_log) == 1
        npu_backbone.encode(_emb(128))
        assert len(npu_backbone._latency_log) == 2


class TestNPUWrapper:
    def test_name_prefix(self, npu_backbone):
        assert npu_backbone.name.startswith("npu:")

    def test_embedding_dim_matches_base(self, npu_backbone):
        assert npu_backbone.embedding_dim == 128

    def test_confidence_delegates(self, npu_backbone):
        result = npu_backbone.encode(_emb(128))
        conf = npu_backbone.confidence(result)
        assert 0.0 <= conf <= 1.0

    def test_custom_target_latency(self):
        base = PassthroughBackbone(dim=64)
        wrapper = NPUBackboneWrapper(base, target_latency_ms=50)
        assert wrapper.target_latency == 50
