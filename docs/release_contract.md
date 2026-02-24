# Release Contract (v0.1.x)

> This document defines the **immutable behavioural guarantees** of
> NeSy-Core v0.1.x. Any change that breaks a contract listed below
> requires a **major version bump** (v0.2.0+).

---

## Section X — Module Contracts

### X.1 Import Contract

```python
from nesy import NeSyModel
```

`NeSyModel` **must** be importable from the top-level `nesy` package.
All public types (`Predicate`, `SymbolicRule`, `ConceptEdge`,
`NSIOutput`, `ConfidenceReport`, etc.) must be importable from
`nesy.core.types`.

### X.2 Deterministic Reasoning

`NeSyModel.reason()` is **deterministic**: given identical `facts`,
`context_type`, `neural_confidence`, and `raw_input`, the returned
`NSIOutput` must satisfy:

| Field | Guarantee |
|-------|-----------|
| `answer` | Identical string (derived facts sorted lexicographically) |
| `status` | Same `OutputStatus` value |
| `confidence.{factual,reasoning,knowledge_boundary}` | Identical floats |
| `flags` | Same list contents (order may vary) |
| `reasoning_trace.steps` | Same step count and descriptions |

### X.3 Proof Capsule (PCAP) Integrity

- `NeSyModel.export_proof_capsule(output)` produces a `ProofCapsule`
  with a SHA-256 checksum.
- `NeSyModel.verify_proof_capsule(capsule)` **must return `False`** if
  any field of the capsule has been modified after export.
- PCAP files use the `.pcap.json` extension and are valid JSON.

### X.4 Validator Wall

Every validation function in `nesy.core.validators` **must raise**
a typed `NeSyError` subclass (never return silently) when input is
invalid. Covered validators:

| Function | Raises on |
|----------|-----------|
| `assert_valid_facts()` | Ungrounded variables, invalid predicate names |
| `assert_valid_rule()` | Empty id, bad weight, missing antecedents/consequents |
| `assert_valid_concept_edge()` | Self-loops, out-of-range probabilities |
| `assert_valid_confidence()` | Scores outside [0, 1] |
| `clamp_probability()` | NaN values (raises); Inf values (clamps) |

### X.5 Counterfactual Fix Generator (CFG)

- `suggest_fixes()` **must never suggest** adding a concept that would
  violate an immutable symbolic anchor.
- Fixes are sorted by predicted uplift (descending).
- Type1 (expected) null items are always excluded from suggestions.

### X.6 Trust Budget / Dual-Channel Verdict / Edge Consistency Seal

These features **must not alter** the `NSIOutput.answer` or
`NSIOutput.status` produced by `reason()`. They add metadata only:

| Feature | Adds |
|---------|------|
| Trust Budget (TB) | `cost`, `remaining_budget`, `budget_exceeded` |
| Dual-Channel Verdict (DCV) | `compliance_grade` (A/B/C/D), `compliance_detail` |
| Edge Consistency Seal (ECS) | `graph_checksum` (SHA-256), `seal_timestamp` |

### X.7 Unsat-Core Explanation

- When the symbolic engine or constraint solver rejects input,
  `explain_contradiction()` **must return** the minimal set of
  conflicting rule IDs.
- Human-readable explanations are generated for every unsat core.
- CFG suggestions are derived from unsat-core analysis when available.

---

## Verification

Run the one-command verification script:

```powershell
.\scripts\verify_release.ps1
```

This executes:
1. Full test suite (`pytest -q`) — must report **0 failures, 0 warnings**
2. Coverage checks for critical modules
3. All example scripts

If this script exits with code 0, the release contract is satisfied.

---

## Contract Change Policy

1. Contracts may be **extended** (new guarantees) in any v0.1.x patch.
2. Contracts may **not** be weakened or removed without a major version.
3. Every new contract must be accompanied by a test in `tests/`.
