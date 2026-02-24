# NeSy-Core

**A Unified Neuro-Symbolic AI Framework with Meta-Cognition, Negative Space Intelligence, and Structural Robustness**

[![CI](https://github.com/ranjitmani61/NeSy-Core/actions/workflows/ci.yml/badge.svg)](https://github.com/ranjitmani61/NeSy-Core/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.1-green.svg)](https://pypi.org/project/nesy-core/)

---

## What is NeSy-Core?

NeSy-Core is an open-source framework that bridges neural learning and symbolic reasoning into a single, production-ready system. Every output is logically grounded, epistemically calibrated, absence-aware, and structurally auditable.

---

## Why NeSy-Core Is Different

No existing framework combines all of these capabilities:

| # | Capability | What it does | Module |
|---|-----------|-------------|--------|
| 1 | **3D Meta-Cognition** | Every answer carries three independent confidence scores (factual, reasoning, knowledge boundary). Below threshold, the system holds output instead of hallucinating. | `nesy.metacognition` |
| 2 | **Negative Space Intelligence (NSI)** | Reasons from what *should be present but is absent*. Computes N(X) = E(X) - P(X) and classifies missing evidence by severity. | `nesy.nsi` |
| 3 | **Continual Learning + Symbolic Anchors** | Learns new tasks without forgetting old ones. EWC protects neural weights; symbolic facts are immutable anchors. | `nesy.continual` |
| 4 | **Counterfactual Shadow Reasoning** | Answers "how many facts must be wrong for this conclusion to flip?" Shadow Distance = 1 means one fact removal breaks the answer. | `nesy.metacognition.shadow` |
| 5 | **NSIL (Neural-Symbolic Integrity Link)** | Auditable integrity signal measuring disagreement between neural grounding evidence and symbolic constraint expectations. Deterministic, coupling into meta-cognition boundary. | `nesy.neural.nsil` |
| 6 | **Reasoning Fingerprint** | Stable SHA-256 digest of the semantic reasoning event (rules fired, predicates derived, null items, config). Identical inputs always produce identical fingerprints. | `nesy.metacognition.fingerprint` |
| 7 | **Proof Capsule (PCAP)** | Exportable, self-contained audit artifact containing the full reasoning trace, confidence report, and fingerprint. | `nesy.api.nesy_model` |
| 8 | **Unsat Core Explanation** | When rules contradict, pinpoints the minimal set of conflicting rules and generates human-readable conflict explanations. | `nesy.symbolic.unsat_explanation` |

---

## Quick Start

```bash
pip install nesy-core
```

```python
from nesy import NeSyModel, SymbolicRule, Predicate, ConceptEdge

# 1. Create a model
model = NeSyModel(domain="medical")

# 2. Add rules
model.add_rule(SymbolicRule(
    id="fever_infection",
    antecedents=[
        Predicate("HasSymptom", ("?p", "fever")),
        Predicate("HasLabResult", ("?p", "elevated_wbc")),
    ],
    consequents=[Predicate("PossiblyHas", ("?p", "bacterial_infection"))],
    weight=0.85,
    description="Fever + elevated WBC suggests bacterial infection",
))

# 3. Add concept edges for absence detection
model.add_concept_edge(ConceptEdge(
    "fever", "blood_test",
    cooccurrence_prob=0.90, causal_strength=1.0, temporal_stability=1.0,
))
model.register_critical_concept("blood_test", "diagnostic_test")

# 4. Reason
facts = {
    Predicate("HasSymptom",   ("patient_1", "fever")),
    Predicate("HasLabResult", ("patient_1", "elevated_wbc")),
}

output = model.reason(facts=facts, context_type="medical")
print(output.summary())
print(model.explain(output))
```

Output includes: derived predicates, 3D confidence scores, null set analysis, reasoning fingerprint, and actionable flags.

---

## Architecture

```
Input Facts
    |
    v
[Symbolic Engine]       <-- FOL inference + Robinson resolution + Betti topology
    |  derived facts, reasoning steps, symbolic confidence
    v
[NSI Concept Graph]     <-- N(X) = E(X) - P(X): null set computation
    |  classified null items (Type 1 / 2 / 3)
    v
[NSIL Integrity Check]  <-- Neural-symbolic grounding audit
    |  integrity score, flags, suggestions
    v
[MetaCognition Monitor]  <-- 3D confidence + self-doubt + shadow distance
    |  confidence report, reasoning trace, output status
    v
[Reasoning Fingerprint]  <-- SHA-256 of canonical reasoning event
    |
    v
NSIOutput: answer + confidence + trace + null report + fingerprint + flags
```

---

## Layers

| Layer | Module | Function |
|-------|--------|----------|
| Symbolic Engine | `nesy.symbolic` | FOL inference, contradiction detection, Betti topology, unsat core |
| Concept Graph | `nesy.nsi` | Null set computation N(X) = E(X) - P(X) |
| Neural Bridge | `nesy.neural` | Embedding-to-symbol grounding, NSIL integrity, constraint gradients |
| MetaCognition | `nesy.metacognition` | 3D confidence, self-doubt, shadow reasoning, fingerprint |
| Continual Learning | `nesy.continual` | EWC + symbolic anchors, no catastrophic forgetting |
| Developer API | `nesy.api` | NeSyModel, Pipeline, Proof Capsule, decorators |
| Deployment | `nesy.deployment` | ONNX export, pruning, quantization, NPU offload |

---

## Install Options

```bash
pip install nesy-core              # Core only (no ML dependencies)
pip install nesy-core[torch]       # + PyTorch backbone
pip install nesy-core[server]      # + FastAPI inference server
pip install nesy-core[edge]        # + ONNX/TFLite export
pip install nesy-core[all]         # Everything
```

---

## Examples

Four runnable examples ship in `examples/`:

| Example | What it demonstrates |
|---------|---------------------|
| `basic_reasoning.py` | Syllogism derivation (Socrates is mortal) with full trace |
| `medical_diagnosis.py` | Multi-scenario clinical reasoning with NSI absence detection |
| `continual_learning.py` | Learn new tasks without forgetting (EWC + symbolic anchors) |
| `edge_deployment.py` | Model pruning, quantization, and ONNX export for edge devices |
| `shadow_demo.py` | Counterfactual shadow distances on a medical diagnosis graph |

```bash
make examples        # Run all five examples
make test            # Run full test suite (985+ tests)
make serve           # Start inference server (localhost:8000)
```

---

## Shadow Reasoning (Counterfactual Fragility)

Every derived conclusion now carries **two** dimensions of confidence:

| Dimension | Symbol | Question it answers |
|-----------|--------|---------------------|
| **Probabilistic confidence** | C₁ ∈ [0, 1] | "How likely is this conclusion correct?" |
| **Shadow Distance** | C₂ ∈ [0, ∞) | "How many facts must be wrong for this conclusion to flip?" |

### What is Shadow Distance?

Given facts **F**, rules **R**, and a derived conclusion **C**:

```
ShadowDistance(C, F, R) = min { |S| : S ⊆ F, (F \ S) ⊬ C using R }
```

Shadow Distance is the **minimum number of input facts whose removal makes the conclusion no longer derivable**. It measures structural robustness, not probability.

### Classification

| Shadow Distance | Class | Meaning |
|-----------------|-------|---------|
| ∞ | TAUTOLOGY | Conclusion is always true (no facts needed) |
| ≥ 5 | ROBUST | 5+ independent facts support it — safe to act |
| 3–4 | STABLE | Moderately supported — act with awareness |
| 2 | FRAGILE | Only 2 facts protect it — human review recommended |
| 1 | CRITICAL | **One fact removal flips the answer** — dangerous |

### How Shadow Affects OutputStatus

Shadow enforcement is controlled by `shadow_policy`:

| Policy | Behavior |
|--------|----------|
| `"none"` | Shadow flags are computed and attached but status is **not** changed (default) |
| `"flag"` | If any conclusion has distance ≤ `shadow_critical_distance` **and** domain is in `shadow_apply_domains`, status is downgraded to **FLAGGED** |
| `"reject"` | Same trigger, but status is downgraded to **REJECTED** |

Shadow policy **never upgrades** status — if the output is already REJECTED (e.g. critical null violations), it stays REJECTED.

### Usage

```python
from nesy import NeSy, SymbolicRule, Predicate

# Medical domain auto-configures shadow_policy="flag"
model = NeSy(domain="medical")

model.add_rule(SymbolicRule(
    id="fever_infection",
    antecedents=[Predicate("HasSymptom", ("?p", "fever"))],
    consequents=[Predicate("Diagnosis", ("?p", "infection"))],
    weight=0.85,
))

facts = {Predicate("HasSymptom", ("patient_1", "fever"))}
output = model.reason(facts=facts, context_type="medical")

# Shadow distance = 1 (CRITICAL) → status auto-downgraded to FLAGGED
print(output.status)        # OutputStatus.FLAGGED
print(output.flags)         # ['SHADOW-CRITICAL: Diagnosis(...) distance=1 ...', ...]
```

Adding more independent evidence raises the shadow distance:

```python
model.add_rule(SymbolicRule(
    id="wbc_infection",
    antecedents=[Predicate("HasLabResult", ("?p", "elevated_wbc"))],
    consequents=[Predicate("Diagnosis", ("?p", "infection"))],
    weight=0.80,
))

facts = {
    Predicate("HasSymptom",  ("patient_1", "fever")),
    Predicate("HasLabResult", ("patient_1", "elevated_wbc")),
}
output = model.reason(facts=facts, context_type="medical")
# Shadow distance = 2 (FRAGILE) → still above critical threshold → OK
```

### Shadow Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `shadow_enabled` | `bool` | `True` | Enable/disable shadow computation |
| `shadow_critical_distance` | `int` | `1` | Conclusions with distance ≤ this trigger policy action |
| `shadow_policy` | `str` | `"none"` | `"none"` / `"flag"` / `"reject"` |
| `shadow_apply_domains` | `list[str]` | `["medical", "legal"]` | Domains where policy enforcement applies |

**Domain defaults** (set automatically by `NeSyConfig.for_domain()` and `NeSyModel(domain=...)`):

| Domain | `shadow_policy` | `shadow_critical_distance` |
|--------|----------------|---------------------------|
| `medical` | `"flag"` | `1` |
| `legal` | `"flag"` | `1` |
| all others | `"none"` | `1` |

---

## Mathematical Foundation

**Null Set:** `N(X) = E(X) - P(X)`
where `E(X) = { c | exists e in P(X) : W(e->c) > threshold }`

**Edge Weight:** `W(a->b) = P(b|a) * causal_strength * temporal_stability`

**EWC Loss:** `L_total = L_new + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2`

**3D Confidence:** `min(C_factual, C_reasoning, C_boundary)`

**Shadow Distance:** `SD(C, F, R) = min { |S| : S subset F, (F \ S) does not derive C }`

**NSIL Integrity:** `ISR(s) = need(s) * |membership(s) - evidence(s)|`
`IntegrityScore = clamp(1.0 - (0.7 * avg(ISR) + 0.3 * max(ISR)), 0, 1)`

**Reasoning Fingerprint:** `SHA-256(canonical_json(rules, predicates, nulls, confidence, config))`

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
