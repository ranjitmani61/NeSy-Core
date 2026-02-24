# API Reference: NeSyModel

`nesy.api.nesy_model.NeSyModel` is the **primary developer-facing API**.
It composes all five layers — SymbolicEngine, ConceptGraphEngine,
MetaCognitionMonitor, ContinualLearner, and SelfDoubtLayer — into a single
"PyTorch-feel" interface.

## Constructor

```python
NeSyModel(
    domain: str = "general",
    doubt_threshold: float = 0.60,
    strict_mode: bool = False,
    lambda_ewc: float = 1000.0,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `domain` | `"general"` | Domain preset (`"medical"`, `"legal"`, `"code"`, `"general"`) |
| `doubt_threshold` | `0.60` | Minimum confidence before self-doubt is triggered |
| `strict_mode` | `False` | In strict mode, `FLAGGED` outputs are upgraded to `REJECTED` |
| `lambda_ewc` | `1000.0` | EWC regularisation strength |

## Knowledge Loading (Chainable)

All loading methods return `self` for fluent API style:

```python
model = (
    NeSyModel(domain="medical")
    .add_rule(rule1)
    .add_rules([rule2, rule3])
    .add_concept_edge(edge1)
    .add_concept_edges([edge2, edge3])
)
```

### `add_rule(rule: SymbolicRule) -> NeSyModel`

Add a single symbolic reasoning rule.

### `add_rules(rules: List[SymbolicRule]) -> NeSyModel`

Add multiple rules at once.

### `add_concept_edge(edge: ConceptEdge) -> NeSyModel`

Add a weighted directed edge to the NSI concept graph.

### `add_concept_edges(edges: List[ConceptEdge]) -> NeSyModel`

Batch-add concept edges.

### `register_critical_concept(concept: str, class_label: str) -> NeSyModel`

Mark a concept as belonging to a critical class (e.g. `"vital_sign"`).
Critical concepts that are absent generate Type 3 (Critical) null items.

### `anchor(rule: SymbolicRule) -> NeSyModel`

Add a rule as an immutable symbolic anchor. Anchored rules survive
continual learning and can never be modified or deleted.

## Reasoning

### `reason(facts, context_type, neural_confidence, raw_input) -> NSIOutput`

```python
def reason(
    self,
    facts: Set[Predicate],
    context_type: str = "general",
    neural_confidence: float = 0.90,
    raw_input: Optional[str] = None,
) -> NSIOutput
```

Full pipeline:

1. **Forward chain** — `SymbolicEngine.reason(facts)` produces derived predicates, reasoning steps, and symbolic confidence.
2. **Null set** — `ConceptGraphEngine.compute_null_set(present_set)` identifies missing concepts.
3. **MetaCognition** — `MetaCognitionMonitor.evaluate(...)` computes 3-axis confidence, builds reasoning trace, determines output status.
4. **Package** — returns `NSIOutput` with answer, confidence, trace, null set, status, flags.

### `explain(output: NSIOutput) -> str`

Generate a human-readable explanation of a reasoning output.

```python
output = model.reason(facts)
print(model.explain(output))
```

## Learning

### `learn(new_rule: SymbolicRule, make_anchor: bool = False) -> NeSyModel`

Add a rule to the symbolic engine. If `make_anchor=True`, the rule is
also registered as an immutable symbolic anchor.

## Persistence

### `save_concept_graph(path: str) -> None`

Serialise the concept graph to JSON.

### `load_concept_graph(path: str) -> NeSyModel`

Load a concept graph from JSON. Returns `self` for chaining.

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `rule_count` | `int` | Total number of loaded rules |
| `concept_graph_stats` | `Dict` | `{concepts, edges, domain, avg_out_degree}` |
| `anchored_rules` | `int` | Number of immutable anchored rules |

## Return Type: `NSIOutput`

```python
@dataclass
class NSIOutput:
    answer: str
    confidence: ConfidenceReport
    reasoning_trace: ReasoningTrace
    null_set: NullSet
    status: OutputStatus        # OK | FLAGGED | UNCERTAIN | REJECTED
    flags: List[str]
    request_id: str             # auto-generated UUID
```

### `NSIOutput.is_trustworthy() -> bool`

Returns `True` if status is `OK` and confidence is reliable.

### `NSIOutput.summary() -> str`

One-line summary string with answer, status, and minimum confidence.

## Example

```python
from nesy.api.nesy_model import NeSyModel
from nesy.core.types import Predicate, SymbolicRule, ConceptEdge

model = NeSyModel(domain="medical", strict_mode=True)

model.add_rule(SymbolicRule(
    id="r1",
    antecedents=[Predicate("fever"), Predicate("cough")],
    consequents=[Predicate("flu_likely")],
    weight=0.85,
))

model.add_concept_edge(ConceptEdge(
    source="fever", target="headache",
    cooccurrence_prob=0.6, causal_strength=0.5, temporal_stability=1.0,
))

output = model.reason(
    facts={Predicate("fever"), Predicate("cough")},
    context_type="medical",
)

assert output.status.name in ("OK", "FLAGGED", "UNCERTAIN", "REJECTED")
print(output.confidence.factual)
print(output.confidence.reasoning)
print(output.confidence.knowledge_boundary)
```
