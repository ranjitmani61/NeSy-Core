"""
tests/benchmarks/test_inference_speed.py
==========================================
Performance benchmarks for NeSy-Core inference.
"""

import time
import pytest
from nesy.api.nesy_model import NeSyModel
from nesy.core.types import ConceptEdge, Predicate, SymbolicRule


@pytest.fixture
def bench_model():
    model = NeSyModel(domain="medical")
    for i in range(20):
        model.add_rule(
            SymbolicRule(
                id=f"rule_{i}",
                antecedents=[Predicate("A", (f"?x_{i}",))],
                consequents=[Predicate("B", (f"?x_{i}",))],
                weight=0.8,
            )
        )
    for i in range(50):
        model.add_concept_edge(
            ConceptEdge(
                f"concept_{i}",
                f"concept_{i + 1}",
                cooccurrence_prob=0.7,
                causal_strength=0.5,
                temporal_stability=1.0,
            )
        )
    return model


def test_single_inference_under_200ms(bench_model):
    facts = {Predicate("A", ("node_1",))}
    t0 = time.perf_counter()
    bench_model.reason(facts=facts, context_type="general")
    latency_ms = (time.perf_counter() - t0) * 1000
    assert latency_ms < 200.0, f"Inference took {latency_ms:.1f}ms > 200ms target"


def test_batch_10_inferences_under_1000ms(bench_model):
    t0 = time.perf_counter()
    for i in range(10):
        facts = {Predicate("A", (f"node_{i}",))}
        bench_model.reason(facts=facts)
    total_ms = (time.perf_counter() - t0) * 1000
    assert total_ms < 1000.0, f"10 inferences took {total_ms:.1f}ms"


"""
tests/benchmarks/test_memory_usage.py
"""


def test_model_import_footprint():
    """NeSy-Core import should not add > 50MB overhead."""
    import tracemalloc

    tracemalloc.start()
    from nesy.api.nesy_model import NeSyModel

    NeSyModel()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 50.0, f"Peak memory {peak_mb:.1f}MB > 50MB target"


"""
tests/benchmarks/test_npu_latency.py
"""


def test_npu_wrapper_adds_minimal_overhead():
    from nesy.neural.base import PassthroughBackbone
    from nesy.deployment.npu import NPUBackboneWrapper

    base = PassthroughBackbone(dim=64)
    npu = NPUBackboneWrapper(base, target_latency_ms=100)
    emb = [0.1] * 64
    for _ in range(10):
        npu.encode(emb)
    avg = npu.avg_latency_ms
    assert avg is not None
    assert avg < 100.0, f"NPU avg latency {avg:.1f}ms > 100ms target"
