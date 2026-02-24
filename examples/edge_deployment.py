"""
examples/edge_deployment.py
============================
Demonstrates NeSy-Core graph compression for edge/mobile deployment.

Steps:
    1. Build a full concept graph with domain knowledge
    2. Compress it via NeSyLite (top-K edges, weight threshold)
    3. Show that the compressed graph still catches critical absences
    4. Demonstrate NPU backbone wrapper with latency tracking

Run: python examples/edge_deployment.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nesy.api.nesy_model import NeSyModel
from nesy.core.types import ConceptEdge, Predicate, SymbolicRule
from nesy.deployment.lite import NeSyLite
from nesy.deployment.optimizer import SymbolicGuidedOptimizer
from nesy.neural.base import PassthroughBackbone
from nesy.deployment.npu import NPUBackboneWrapper


def main():
    # ─── 1. Build a full medical concept graph ────────────────────
    model = NeSyModel(domain="medical")

    # Add domain rules
    model.add_rule(SymbolicRule(
        id="fever_infection",
        antecedents=[Predicate("HasSymptom", ("?p", "fever"))],
        consequents=[Predicate("PossiblyHas", ("?p", "infection"))],
        weight=0.80,
        description="Fever suggests infection",
    ))
    model.add_rule(SymbolicRule(
        id="infection_needs_blood",
        antecedents=[Predicate("PossiblyHas", ("?p", "infection"))],
        consequents=[Predicate("Requires", ("?p", "blood_test"))],
        weight=0.90,
        description="Infection requires blood test",
    ))

    # Add concept edges (full graph)
    edges = [
        ConceptEdge("fever", "blood_test", 0.90, 1.0, 1.0),
        ConceptEdge("fever", "temperature_reading", 0.95, 1.0, 1.0),
        ConceptEdge("fever", "hydration_check", 0.60, 0.5, 1.0),
        ConceptEdge("fever", "chest_xray", 0.40, 0.5, 0.5),
        ConceptEdge("cough", "chest_xray", 0.85, 1.0, 1.0),
        ConceptEdge("cough", "sputum_culture", 0.70, 0.5, 1.0),
        ConceptEdge("headache", "ct_scan", 0.30, 0.5, 0.5),
        ConceptEdge("headache", "vision_test", 0.25, 0.1, 0.5),
        ConceptEdge("infection", "antibiotic", 0.75, 1.0, 1.0),
        ConceptEdge("infection", "culture", 0.80, 1.0, 1.0),
    ]
    model.add_concept_edges(edges)

    print("=" * 60)
    print("FULL CONCEPT GRAPH")
    print("=" * 60)
    print(f"Stats: {model.concept_graph_stats}")

    # ─── 2. Reason with full graph ────────────────────────────────
    facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
    output_full = model.reason(facts=facts, context_type="medical")

    print(f"\nFull graph result:")
    print(f"  Answer:     {output_full.answer}")
    print(f"  Status:     {output_full.status.value}")
    print(f"  Confidence: {output_full.confidence.minimum:.3f}")
    print(f"  Null items: {len(output_full.null_set.items)}")

    # ─── 3. Compress graph for edge deployment ────────────────────
    print("\n" + "=" * 60)
    print("COMPRESSED GRAPH (NeSy-Lite)")
    print("=" * 60)

    lite_graph = NeSyLite.compress(
        source=model._cge,
        top_k_edges=3,
        min_weight=0.35,
        preserve_concepts={"blood_test", "temperature_reading"},
    )

    print(f"Lite stats: {lite_graph.stats}")

    # Build a new model with the compressed graph
    lite_model = NeSyModel(domain="medical")
    lite_model.add_rules(model._symbolic.rules)
    lite_model._cge = lite_graph

    output_lite = lite_model.reason(facts=facts, context_type="medical")

    print(f"\nLite graph result:")
    print(f"  Answer:     {output_lite.answer}")
    print(f"  Status:     {output_lite.status.value}")
    print(f"  Confidence: {output_lite.confidence.minimum:.3f}")
    print(f"  Null items: {len(output_lite.null_set.items)}")

    # ─── 4. NPU backbone wrapper demo ─────────────────────────────
    print("\n" + "=" * 60)
    print("NPU BACKBONE WRAPPER")
    print("=" * 60)

    base_backbone = PassthroughBackbone(dim=128)
    npu_backbone = NPUBackboneWrapper(base_backbone, target_latency_ms=50)

    # Simulate encoding
    dummy_embedding = [0.1] * 128
    for i in range(10):
        result = npu_backbone.encode(dummy_embedding)

    print(f"  Backbone: {npu_backbone.name}")
    print(f"  Embedding dim: {npu_backbone.embedding_dim}")
    print(f"  Avg latency: {npu_backbone.avg_latency_ms:.3f} ms")
    print(f"  P95 latency: {npu_backbone.p95_latency_ms:.3f} ms")
    print(f"  Confidence:  {npu_backbone.confidence(result):.3f}")

    # ─── 5. Symbolic-guided optimisation demo ──────────────────────
    print("\n" + "=" * 60)
    print("SYMBOLIC-GUIDED OPTIMISATION")
    print("=" * 60)

    optimizer = SymbolicGuidedOptimizer(
        quantization_bits=8,
        pruning_threshold=0.30,
    )

    # Mock parameter-to-rule mapping
    params = {"layer1.weight": 0.75, "layer1.bias": 0.10,
              "layer2.weight": 0.50, "layer2.bias": 0.05}
    param_rule_map = {
        "layer1.weight": ["fever_infection"],
        "layer1.bias":   ["fever_infection"],
        "layer2.weight": ["infection_needs_blood"],
        "layer2.bias":   [],
    }

    importance = optimizer.compute_importance_scores(
        rules=model._symbolic.rules,
        param_rule_mapping=param_rule_map,
    )
    print(f"  Importance scores: {importance}")

    pruned = optimizer.prune_params(params, importance)
    print(f"  After pruning:     {pruned}")

    quantized = optimizer.quantize(pruned)
    print(f"  After quantization: {quantized}")

    print("\n" + "=" * 60)
    print("Edge deployment demo complete.")


if __name__ == "__main__":
    main()
