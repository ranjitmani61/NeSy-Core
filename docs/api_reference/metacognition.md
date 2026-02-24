# API Reference: MetaCognition

The metacognition layer (`nesy/metacognition/`) is the **primary novelty**
of NeSy-Core. It answers: *"Does the system know what it knows?"*

---

## MetaCognitionMonitor (`nesy.metacognition.monitor`)

Central orchestrator that computes three-dimensional confidence, builds
reasoning traces, determines output status, and logs calibration data.

### Constructor

```python
MetaCognitionMonitor(
    doubt_threshold: float = 0.60,
    strict_mode: bool = False,
    trace_all_steps: bool = True,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `doubt_threshold` | `0.60` | Minimum confidence before self-doubt triggers |
| `strict_mode` | `False` | Upgrade `FLAGGED` → `REJECTED` |
| `trace_all_steps` | `True` | Record all reasoning steps in traces |

### Main Method

```python
monitor.evaluate(
    answer: str,
    neural_confidence: float,
    symbolic_confidence: float,
    reasoning_steps: List[ReasoningStep],
    logic_clauses: List[LogicClause],
    null_set: NullSet,
    query: Optional[str] = None,
) -> Tuple[ConfidenceReport, ReasoningTrace, OutputStatus, List[str]]
```

Pipeline:
1. Compute 3D confidence via `confidence.build_confidence_report()`
2. Build `ReasoningTrace` from steps
3. Determine `OutputStatus` (see table below)
4. Apply strict-mode escalation
5. Log `(predicted_confidence, was_correct)` for calibration

### Status Determination

| Status | Condition |
|--------|-----------|
| `REJECTED` | Any Type 3 (Critical) null items present |
| `FLAGGED` | $\min(C_f, C_r, C_b) < \text{doubt\_threshold}$ |
| `UNCERTAIN` | $\min \in [\text{doubt\_threshold}, 0.75)$ |
| `OK` | $\min \geq 0.75$ |

### Properties

- `calibration_data -> List[Tuple[float, bool]]`
- `reset_calibration_log() -> None`

---

## Three-Dimensional Confidence (`nesy.metacognition.confidence`)

Four pure functions — no state, no side effects.

### `compute_factual(null_set: NullSet) -> float`

$$
C_f = \frac{1}{1 + A(X)}
$$

Where $A(X)$ is the total anomaly score from the null set.

### `compute_reasoning(symbolic_confidence: float, clauses: List[LogicClause]) -> float`

$$
C_r = \sqrt{C_{sym} \times \text{satisfied\_ratio}}
$$

Where `satisfied_ratio` = proportion of logic clauses marked as `satisfied`.

### `compute_boundary(neural_confidence: float, null_set: NullSet) -> float`

$$
C_b = C_{neural} \times (1 - 0.3 \times \text{gap})
$$

Where `gap` is the ratio of meaningful/critical nulls to total concepts.

### `build_confidence_report(...) -> ConfidenceReport`

Composes all three axes into a single `ConfidenceReport`:

```python
@dataclass
class ConfidenceReport:
    factual: float
    reasoning: float
    knowledge_boundary: float
    explanation: Dict[ConfidenceType, str]

    @property
    def minimum(self) -> float: ...  # min of all three

    @property
    def is_reliable(self) -> bool: ...  # minimum >= 0.6
```

---

## Self-Doubt Layer (`nesy.metacognition.doubt`)

Three triggers for withholding output:

```python
SelfDoubtLayer(threshold: float = 0.60, betti_warn: int = 2)

doubt.evaluate(
    confidence: ConfidenceReport,
    null_set: NullSet,
    betti_0: int = 1,
) -> Tuple[OutputStatus, List[str]]
```

| Trigger | Check | Result |
|---------|-------|--------|
| Low confidence | $\min(C_f, C_r, C_b) < \text{threshold}$ | `FLAGGED` |
| Critical nulls | Any `NullType.TYPE3_CRITICAL` items | `REJECTED` |
| Topological disconnect | $\beta_0 > \text{betti\_warn}$ | `FLAGGED` |

---

## Confidence Calibration (`nesy.metacognition.calibration`)

Platt scaling to align predicted confidence with empirical accuracy.

### Constructor

```python
ConfidenceCalibrator(n_bins: int = 10)
```

### Fitting

```python
calibrator.fit(data: List[Tuple[float, bool]]) -> ConfidenceCalibrator
```

Requires at least 100 samples for statistical reliability.

### Calibrating

```python
calibrated = calibrator.calibrate(raw_confidence)
```

### Expected Calibration Error

```python
ece = calibrator.ece  # Optional[float] — None if not yet fitted
```

$$
ECE = \sum_{b=1}^{B} \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|
$$

---

## Knowledge Boundary Estimation (`nesy.metacognition.boundary`)

### KNNBoundaryEstimator

Estimates whether a query is in-distribution via k-nearest-neighbour
distance in embedding space.

```python
estimator = KNNBoundaryEstimator(k=5, threshold=0.50)
estimator.fit(training_embeddings)

score = estimator.estimate(query_embedding)   # 0..1 (higher = more in-distribution)
is_known = estimator.is_in_distribution(query_embedding)
```

### DensityBoundaryEstimator

Gaussian kernel density estimation alternative.

```python
estimator = DensityBoundaryEstimator(bandwidth=0.1)
estimator.fit(training_embeddings)
score = estimator.estimate(query_embedding)
```

---

## Reasoning Trace (`nesy.metacognition.trace`)

### TraceBuilder

Fluent builder for auditable reasoning traces:

```python
from nesy.metacognition.trace import TraceBuilder

trace = (
    TraceBuilder()
    .add_initial_facts(facts)
    .add_derived(Predicate("flu_likely"), rule_id="r1", rule_weight=0.85)
    .add_clause(logic_clause)
    .build(neural_confidence=0.9, symbolic_confidence=0.8, null_violations=[])
)
```

Returns a `ReasoningTrace` with:
- `steps: List[ReasoningStep]`
- `rules_activated: List[str]`
- `neural_confidence: float`
- `symbolic_confidence: float`
- `null_violations: List[NullItem]`
- `logic_clauses: List[LogicClause]`
- `overall_confidence -> float` (harmonic mean, penalised by anomaly)
