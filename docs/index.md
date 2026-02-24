# NeSy-Core Documentation

**NeSy-Core** is a unified neuro-symbolic AI framework that fuses neural perception
with symbolic reasoning, metacognition, and Negative Space Intelligence (NSI).

## Key Innovations

| # | Innovation | What it does |
|---|-----------|--------------|
| 1 | **MetaCognition Monitor** | Three-dimensional confidence (Factual, Reasoning, Knowledge-Boundary) with self-doubt and calibration. The system knows *what it knows*. |
| 2 | **Negative Space Intelligence (NSI)** | Detects *what is missing* from an input via concept-graph expected sets: $N(X) = E(X) - P(X)$. |
| 3 | **Continual Symbolic Anchoring** | EWC + immutable symbolic anchors prevent catastrophic forgetting across task boundaries. |

## Architecture Layers

```
 ┌───────────────────────────────────────────┐
 │         integrations (HF · LangChain)     │
 ├───────────────────────────────────────────┤
 │         evaluation (metrics · eval)       │
 ├───────────────────────────────────────────┤
 │         deployment (lite · export · NPU)  │
 ├───────────────────────────────────────────┤
 │         api (NeSyModel · pipeline)        │
 ├───────────────────────────────────────────┤
 │         continual (EWC · anchors · replay)│
 ├───────────────────────────────────────────┤
 │         neural (backbone · bridge · loss) │
 ├───────────────────────────────────────────┤
 │         metacognition (monitor · doubt)   │
 ├───────────────────────────────────────────┤
 │         nsi (concept graph · null sets)   │
 ├───────────────────────────────────────────┤
 │         symbolic (engine · prover · logic)│
 ├───────────────────────────────────────────┤
 │         core (types · config · registry)  │
 └───────────────────────────────────────────┘
```

## Quick Links

- [Quickstart Guide](quickstart.md) — Install and run your first query in 5 minutes.
- [Architecture](architecture.md) — Deep dive into the 10-layer stack.
- **API Reference**
  - [NeSyModel](api_reference/nesy_model.md) — Main developer-facing API.
  - [Symbolic Engine](api_reference/symbolic_engine.md) — Rules, forward chaining, resolution.
  - [MetaCognition](api_reference/metacognition.md) — Confidence, doubt, calibration.
- **Guides**
  - [Adding Rules](guides/adding_rules.md) — Define and load symbolic rules.
  - [Custom Backbone](guides/custom_backbone.md) — Integrate your own neural model.
  - [Edge Deployment](guides/edge_deployment.md) — Compress and export for mobile / NPU.

## Installation

```bash
pip install nesy-core              # core (symbolic + NSI + metacognition)
pip install nesy-core[torch]       # + PyTorch neural backbone support
pip install nesy-core[server]      # + FastAPI server
pip install nesy-core[all]         # everything including dev tools
```

## Minimal Example

```python
from nesy.api.nesy_model import NeSyModel
from nesy.core.types import Predicate, SymbolicRule, ConceptEdge

model = NeSyModel(domain="medical", strict_mode=True)

model.add_rule(
    SymbolicRule(
        id="fever_flu",
        antecedents=[Predicate("fever"), Predicate("cough")],
        consequents=[Predicate("flu_likely")],
        weight=0.85,
        domain="medical",
    )
)

output = model.reason(
    facts={Predicate("fever"), Predicate("cough")},
    context_type="medical",
)

print(output.answer)               # "flu_likely"
print(output.confidence.minimum)   # three-dimensional confidence
print(output.status)               # OK | FLAGGED | UNCERTAIN | REJECTED
```

## License

Apache-2.0 — see [LICENSE](../LICENSE).
