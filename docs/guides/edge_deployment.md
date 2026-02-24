# Guide: Edge Deployment

Deploy NeSy-Core on mobile devices, embedded systems, and NPUs.

## Overview

Edge deployment involves two independent steps:

1. **Compress the concept graph** — reduce edges for low-memory devices.
2. **Export the neural backbone** — convert to ONNX / TFLite / CoreML.

The symbolic engine and metacognition monitor run on CPU and are **not**
exported — only the neural backbone is converted.

## Step 1: Compress the Concept Graph

`NeSyLite.compress()` prunes the concept graph to a smaller footprint
(typically 10–100× smaller) while preserving ~85% reasoning accuracy.

```python
from nesy.deployment.lite import NeSyLite
from nesy.nsi.concept_graph import ConceptGraphEngine
from nesy.core.types import ConceptEdge

# Build a full concept graph
full_graph = ConceptGraphEngine(domain="medical")
full_graph.add_edges([
    ConceptEdge(source="fever", target="cough",
                cooccurrence_prob=0.7, causal_strength=0.5,
                temporal_stability=1.0),
    # ... many more edges
])

# Compress: keep top 10 edges per node, min weight 0.30
lite_graph = NeSyLite.compress(
    source=full_graph,
    top_k_edges=10,
    min_weight=0.30,
    preserve_concepts={"fever", "chest_pain"},  # always keep these
)

print(lite_graph.stats)  # {concepts: ..., edges: ..., ...}
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k_edges` | `10` | Maximum outgoing edges per concept |
| `min_weight` | `0.30` | Minimum edge weight to retain |
| `preserve_concepts` | `None` | Concepts whose edges are always kept |

## Step 2: Export the Neural Backbone

```python
from nesy.deployment.exporter import ModelExporter
from nesy.neural.base import PassthroughBackbone

backbone = PassthroughBackbone(dim=128)  # or your trained model
```

### ONNX

```python
ModelExporter.to_onnx(
    backbone=backbone,
    output_path="model.onnx",
    input_dim=128,
    opset_version=13,
)
```

Requires: `pip install torch onnx`

### TFLite

```python
ModelExporter.to_tflite(
    backbone=backbone,
    output_path="model.tflite",
    input_dim=128,
    quantise=True,    # INT8 quantization
)
```

Requires: `pip install tensorflow`

### CoreML

```python
ModelExporter.to_coreml(
    backbone=backbone,
    output_path="model.mlmodel",
    input_dim=128,
)
```

Requires: `pip install coremltools torch`

### CLI Script

```bash
python scripts/export_model.py --format onnx --output model.onnx --input-dim 128
python scripts/export_model.py --format tflite --output model.tflite --quantize
python scripts/export_model.py --format coreml --output model.mlmodel
```

## Step 3: NPU Acceleration (Optional)

The `NPUWrapper` wraps a backbone for NPU execution:

```python
from nesy.deployment.npu import NPUWrapper

npu_backbone = NPUWrapper(backbone=backbone)
embedding = npu_backbone.encode(input_data)
```

If no NPU is detected, it falls back to CPU transparently.

## Step 4: Symbolic-Guided Optimization

```python
from nesy.deployment.optimizer import SymbolicGuidedOptimizer

optimizer = SymbolicGuidedOptimizer(rules=rules)
optimised_graph = optimizer.optimize(concept_graph)
```

The optimizer uses symbolic rule coverage analysis to identify which
concept graph edges are actually reachable by the rule set, and prunes
unreachable edges.

## Complete Example

See `examples/edge_deployment.py` for a full working example that:

1. Builds a medical concept graph
2. Compresses with `NeSyLite`
3. Wraps with `NPUWrapper`
4. Applies `SymbolicGuidedOptimizer`

## Docker Deployment

```bash
docker build -f docker/Dockerfile -t nesy-core .
docker run -p 8000:8000 nesy-core
```

The production Dockerfile produces a slim image with only the runtime
dependencies needed for inference.

## Performance Targets

| Metric | Target |
|--------|--------|
| Compressed graph size | 10–100× smaller than full |
| Accuracy retention | ~85% of full graph |
| Inference latency (NPU) | < 200ms |
| Backbone export | ONNX opset 13+ |
