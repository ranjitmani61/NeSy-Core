# Quickstart

Get NeSy-Core running in under 5 minutes.

## Installation

```bash
# Core framework (no ML dependencies)
pip install nesy-core

# With PyTorch support
pip install nesy-core[torch]

# Everything (torch + server + edge + dev)
pip install nesy-core[all]
```

Or install from source:

```bash
git clone https://github.com/your-org/nesy-core.git
cd nesy-core
pip install -e ".[all]"
```

## Your First Query

```python
from nesy.api.nesy_model import NeSyModel
from nesy.core.types import Predicate, SymbolicRule, ConceptEdge

# 1. Create a model for the medical domain
model = NeSyModel(domain="medical", strict_mode=True)

# 2. Add symbolic rules
model.add_rule(SymbolicRule(
    id="fever_flu",
    antecedents=[Predicate("fever"), Predicate("cough")],
    consequents=[Predicate("flu_likely")],
    weight=0.85,
    domain="medical",
))

model.add_rule(SymbolicRule(
    id="flu_rest",
    antecedents=[Predicate("flu_likely")],
    consequents=[Predicate("recommend_rest")],
    weight=0.90,
    domain="medical",
))

# 3. Add concept edges (NSI world model)
model.add_concept_edges([
    ConceptEdge(source="fever", target="cough",
                cooccurrence_prob=0.7, causal_strength=0.5, temporal_stability=1.0),
    ConceptEdge(source="fever", target="headache",
                cooccurrence_prob=0.6, causal_strength=0.5, temporal_stability=1.0),
    ConceptEdge(source="cough", target="sore_throat",
                cooccurrence_prob=0.5, causal_strength=0.5, temporal_stability=1.0),
])

# 4. Reason over input facts
output = model.reason(
    facts={Predicate("fever"), Predicate("cough")},
    context_type="medical",
)

print(f"Answer : {output.answer}")
print(f"Status : {output.status}")
print(f"Factual: {output.confidence.factual:.3f}")
print(f"Reason : {output.confidence.reasoning:.3f}")
print(f"Boundary: {output.confidence.knowledge_boundary:.3f}")
```

## Understanding the Output

`model.reason()` returns an `NSIOutput` with:

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `str` | Derived conclusions joined by `,` |
| `confidence` | `ConfidenceReport` | Three-axis confidence (factual, reasoning, boundary) |
| `status` | `OutputStatus` | `OK`, `FLAGGED`, `UNCERTAIN`, or `REJECTED` |
| `reasoning_trace` | `ReasoningTrace` | Step-by-step derivation audit trail |
| `null_set` | `NullSet` | Missing concepts (NSI null space) |
| `flags` | `List[str]` | Human-readable warnings |

## Human-Readable Explanations

```python
explanation = model.explain(output)
print(explanation)
```

Produces multi-line text describing which rules fired, which concepts are
missing, and why the confidence scores are what they are.

## Null Set (What's Missing)

```python
for item in output.null_set.items:
    print(f"  Missing: {item.concept} (type={item.null_type.name}, "
          f"weight={item.weight:.2f})")
```

Type 3 (Critical) nulls indicate a significant domain anomaly.
Type 2 (Meaningful) nulls are notable absences worth investigating.

## Adding Anchored Rules (Continual Learning)

Anchored rules survive continual learning — they can never be forgotten:

```python
anchor_rule = SymbolicRule(
    id="critical_safety",
    antecedents=[Predicate("chest_pain"), Predicate("shortness_of_breath")],
    consequents=[Predicate("cardiac_emergency")],
    weight=0.99,
    domain="medical",
    immutable=True,
)
model.anchor(anchor_rule)
```

## Running the Server

```bash
# Start the FastAPI inference server
python scripts/serve.py --host 0.0.0.0 --port 8000

# Or with uvicorn directly
uvicorn nesy.deployment.server.app:app --host 0.0.0.0 --port 8000
```

Then query via REST:

```bash
curl -X POST http://localhost:8000/api/v1/reason \
  -H "Content-Type: application/json" \
  -d '{"facts": ["fever", "cough"], "context_type": "medical"}'
```

## Next Steps

- [Architecture](architecture.md) — Understand the 10-layer stack.
- [Adding Rules](guides/adding_rules.md) — Fluent rule builder API.
- [Custom Backbone](guides/custom_backbone.md) — Plug in your own neural model.
- [Edge Deployment](guides/edge_deployment.md) — Export to ONNX / TFLite for mobile.
- [API Reference: NeSyModel](api_reference/nesy_model.md)
