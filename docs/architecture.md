# Architecture

NeSy-Core is organised as a **10-layer stack**. Each layer depends only on
layers below it — no circular imports.

## Layer Diagram

```
Layer 10  integrations   HuggingFace · LangChain · OpenAI · Lightning
Layer  9  evaluation     NeSyEvaluator · Metrics · EvalReport
Layer  8  deployment     ModelExporter · NeSyLite · NPU · Server
Layer  7  api            NeSyModel · Pipeline · Decorators
Layer  6  continual      ContinualLearner · EWC · SymbolicAnchor · Replay
Layer  5  neural         NeSyBackbone · Bridge · SymbolGrounder · Loss
Layer  4  metacognition  MetaCognitionMonitor · SelfDoubt · Calibration
Layer  3  nsi            ConceptGraphEngine · PathFinder · GraphBuilder
Layer  2  symbolic       SymbolicEngine · Prover · Logic · Solver · Betti
Layer  1  core           types · config · registry · validators · exceptions
```

## 1. Core (`nesy/core/`)

Foundation types shared by every module.

| File | Purpose |
|------|---------|
| `types.py` | `Predicate`, `SymbolicRule`, `ConceptEdge`, `NullItem`, `NullSet`, `NSIOutput`, `ConfidenceReport`, `ReasoningTrace`, etc. |
| `config.py` | `NeSyConfig` with domain-specific presets (medical, legal, code, general). |
| `registry.py` | Global registry of backbones, rule loaders, and plugins. |
| `validators.py` | Schema validators for rules, edges, and configs. |
| `exceptions.py` | Typed exceptions: `SymbolicConflict`, `NullSetViolation`, `ContinualLearningConflict`. |

## 2. Symbolic (`nesy/symbolic/`)

Deterministic reasoning using first-order logic.

### Forward Chaining

`SymbolicEngine.reason()` fires rules repeatedly until fixed-point:

```
while new_facts_derived and depth < max_depth:
    for each rule where antecedents ⊆ known_facts:
        add consequents to known_facts
```

Symbolic confidence is the **geometric mean** of activated rule weights,
penalised by the Betti number $\beta_0$ (topological disconnection):

$$
C_{symbolic} = \left(\prod_{r \in R_{fired}} w_r\right)^{1/|R_{fired}|} \times \frac{1}{\beta_0}
$$

### Resolution & Satisfiability

`logic.py` implements Robinson's unification and resolution refutation.
Rules are converted to CNF clauses: $\neg A_1 \lor \neg A_2 \lor C$.

### Backward Chaining (Prover)

`prover.py` provides SLD resolution (Prolog-style goal-directed search)
with depth-bounded proof trees.

### Topology (Betti Numbers)

$\beta_0$ counts connected components among activated predicates.
$\beta_0 = 1$ means a coherent chain; $\beta_0 > 1$ indicates disconnected
sub-proofs and triggers a self-doubt flag.

## 3. NSI — Negative Space Intelligence (`nesy/nsi/`)

The **NSI concept graph** $G = (V, E, W)$ models domain knowledge as a
weighted directed graph where edge weight $w(a, b)$ combines co-occurrence,
causal strength, and temporal stability.

### Null Set Computation

$$
N(X) = E(X) - P(X)
$$

Where $E(X)$ is the **expected set** (all concepts with edge weight above
a context-dependent threshold from present concepts) and $P(X)$ is the
**present set** (observed facts).

Each missing concept is classified:

| Type | Condition | Meaning |
|------|-----------|---------|
| Type 3 (Critical) | weight ≥ 0.60 | Absence is an anomaly |
| Type 2 (Meaningful) | 0.35 ≤ weight < 0.60 | Absence is notable |
| Type 1 (Expected) | threshold ≤ weight < 0.35 | Absence is unremarkable |

## 4. MetaCognition (`nesy/metacognition/`)

**Primary innovation.** Three independent confidence axes:

| Axis | Formula | Source |
|------|---------|--------|
| Factual | $C_f = 1 / (1 + A(X))$ | Null set anomaly score |
| Reasoning | $C_r = \sqrt{C_{sym} \times \text{sat\_ratio}}$ | Symbolic confidence × clause satisfaction |
| Knowledge Boundary | $C_b = C_{neural} \times (1 - 0.3 \times \text{gap})$ | Neural confidence × boundary gap |

### Self-Doubt Layer

Three triggers cause output to be held:

1. **Confidence** — $\min(C_f, C_r, C_b) < \text{doubt\_threshold}$
2. **Critical Nulls** — Any Type 3 null items
3. **Topological Disconnection** — $\beta_0 > \text{betti\_warn}$

### Status Determination

| Status | Condition |
|--------|-----------|
| `REJECTED` | Type 3 critical nulls present |
| `FLAGGED` | $\min < \text{doubt\_threshold}$ |
| `UNCERTAIN` | $\min \in [\text{doubt}, 0.75)$ |
| `OK` | $\min \geq 0.75$ |

### Calibration (Platt Scaling)

`ConfidenceCalibrator` fits logistic parameters on $(predicted, actual)$
pairs and reports Expected Calibration Error (ECE).

## 5. Neural (`nesy/neural/`)

Framework-agnostic abstraction over ML models.

- **`NeSyBackbone`** — abstract base class; implement `encode()` and `confidence()`.
- **`SymbolGrounder`** — maps embeddings to predicates via cosine similarity against registered prototypes.
- **`NeuralSymbolicBridge`** — bidirectional translation: embedding → predicates, constraint violations → loss penalty.
- **`SymbolicHingeLoss` / `KBConstraintLoss`** — differentiable losses shaped by symbolic rules.

## 6. Continual Learning (`nesy/continual/`)

Prevents catastrophic forgetting through two mechanisms:

1. **EWC Regulariser** — $L_{EWC} = \lambda \sum_i F_i (\theta_i - \theta_i^*)^2$
2. **Symbolic Anchors** — Immutable rules that cannot be modified or deleted.

Replay buffers (`RandomReplay`, `PrioritisedReplay`, `SymbolicAnchorReplay`)
provide experience replay strategies for continual training.

## 7. API (`nesy/api/`)

`NeSyModel` is the primary developer interface. It composes all layers:

```python
model = NeSyModel(domain="medical", strict_mode=True)
model.add_rule(rule).add_concept_edges(edges)
output: NSIOutput = model.reason(facts, context_type="medical")
```

## 8. Deployment (`nesy/deployment/`)

- **`NeSyLite.compress()`** — prune concept graph to top-K edges per node.
- **`ModelExporter`** — export backbone to ONNX / TFLite / CoreML.
- **`NPUWrapper`** — NPU acceleration shim.
- **FastAPI Server** — REST endpoints at `/api/v1/reason`, `/health`.

## 9. Evaluation (`nesy/evaluation/`)

`NeSyEvaluator` runs labelled `EvalCase` sets and produces a `NeSyEvalReport`
with symbolic F1, null-set F1, Brier score, ECE, and self-doubt metrics.

## 10. Integrations (`nesy/integrations/`)

Adapters for HuggingFace Transformers, LangChain tools, OpenAI function
calling, and PyTorch Lightning training modules.

## Data Flow

```
Input facts ─► SymbolicEngine.reason()
                  │
                  ├──► derived predicates + reasoning steps
                  │
                  ▼
              ConceptGraphEngine.compute_null_set()
                  │
                  ├──► NullSet (Type 1/2/3 absent concepts)
                  │
                  ▼
              MetaCognitionMonitor.evaluate()
                  │
                  ├──► ConfidenceReport (3-axis)
                  ├──► ReasoningTrace
                  ├──► OutputStatus
                  │
                  ▼
              NSIOutput (final answer + confidence + trace + nulls)
```
