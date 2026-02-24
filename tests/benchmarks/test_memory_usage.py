"""
tests/benchmarks/test_memory_usage.py
=======================================
Memory usage benchmarks for NeSy-Core.
Ensures the framework stays within acceptable memory bounds
for both cloud and edge deployment scenarios.
"""
import sys
import tracemalloc
import pytest
from nesy.core.types import ConceptEdge, Predicate, SymbolicRule


def test_model_import_footprint():
    """NeSy-Core import + model creation should not exceed 50 MB."""
    tracemalloc.start()
    from nesy.api.nesy_model import NeSyModel
    model = NeSyModel()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 50.0, f"Peak memory {peak_mb:.1f} MB > 50 MB target"


def test_100_rules_under_5mb():
    """Loading 100 rules should stay under 5 MB incremental."""
    from nesy.api.nesy_model import NeSyModel
    tracemalloc.start()
    model = NeSyModel(domain="general")
    for i in range(100):
        model.add_rule(SymbolicRule(
            id=f"rule_{i}",
            antecedents=[Predicate("A", (f"?x_{i}",))],
            consequents=[Predicate("B", (f"?x_{i}",))],
            weight=0.8,
        ))
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 5.0, f"100 rules used {peak_mb:.1f} MB > 5 MB target"


def test_500_edges_under_10mb():
    """Loading 500 concept edges should stay under 10 MB."""
    from nesy.api.nesy_model import NeSyModel
    tracemalloc.start()
    model = NeSyModel(domain="general")
    for i in range(500):
        model.add_concept_edge(ConceptEdge(
            f"concept_{i}", f"concept_{i+1}",
            cooccurrence_prob=0.7,
            causal_strength=0.5,
            temporal_stability=1.0,
        ))
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 10.0, f"500 edges used {peak_mb:.1f} MB > 10 MB target"


def test_episodic_buffer_bounded():
    """Memory buffer with max_size=500 should stay under 2 MB."""
    from nesy.continual.memory_buffer import EpisodicMemoryBuffer, MemoryItem
    tracemalloc.start()
    buf = EpisodicMemoryBuffer(max_size=500)
    for i in range(5000):
        buf.add(MemoryItem(data={"x": i, "y": i * 2}, task_id=f"task_{i % 5}"))
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    assert buf.size == 500
    assert peak_mb < 2.0, f"Buffer used {peak_mb:.1f} MB > 2 MB target"


def test_single_inference_memory():
    """A single inference call should not allocate more than 5 MB."""
    from nesy.api.nesy_model import NeSyModel
    model = NeSyModel(domain="general")
    model.add_rule(SymbolicRule(
        id="r1",
        antecedents=[Predicate("A", ("?x",))],
        consequents=[Predicate("B", ("?x",))],
        weight=0.9,
    ))
    tracemalloc.start()
    output = model.reason(
        facts={Predicate("A", ("node",))},
        context_type="general",
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 5.0, f"Inference used {peak_mb:.1f} MB > 5 MB target"
