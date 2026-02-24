# API Reference: Symbolic Engine

The symbolic layer (`nesy/symbolic/`) provides deterministic first-order
logic reasoning with forward chaining, backward chaining (SLD resolution),
satisfiability checking, constraint solving, and topological analysis.

---

## SymbolicEngine (`nesy.symbolic.engine`)

Orchestrates rule loading, consistency checking, forward chaining, and
topological validation.

### Constructor

```python
SymbolicEngine(domain: Optional[str] = None)
```

### Rule Management

```python
engine.add_rule(rule: SymbolicRule) -> None       # raises SymbolicConflict if immutable
engine.load_rules(rules: List[SymbolicRule]) -> None
engine.remove_rule(rule_id: str) -> None           # raises SymbolicConflict if immutable
```

Properties:
- `engine.rules -> List[SymbolicRule]` — all loaded rules
- `engine.hard_rules -> List[SymbolicRule]` — rules with weight ≥ 0.95

### Consistency Checking

```python
engine.check_consistency(facts: Set[Predicate]) -> bool
```

Raises `SymbolicConflict` on contradiction; returns `True` if consistent.

### Forward Chaining

```python
engine.reason(
    facts: Set[Predicate],
    check_hard_constraints: bool = True,
    max_depth: int = 50,
) -> Tuple[Set[Predicate], List[ReasoningStep], float]
```

Returns `(derived_facts, reasoning_steps, symbolic_confidence)`.

Confidence formula:

$$
C_{sym} = \left(\prod_{r \in R_{fired}} w_r\right)^{1/|R|} \times \frac{1}{\beta_0}
$$

---

## Logic Primitives (`nesy.symbolic.logic`)

### Unification

```python
unify(p1: Predicate, p2: Predicate, theta=None) -> Optional[Substitution]
```

Robinson's unification algorithm. Variables start with `?`.

### Resolution

```python
resolve_clauses(
    clause_a: FrozenSet[Predicate],
    clause_b: FrozenSet[Predicate],
) -> Optional[FrozenSet[Predicate]]
```

Resolution inference rule for refutation-based theorem proving.

### Satisfiability

```python
is_satisfiable(clauses: List[FrozenSet[Predicate]], max_steps=1000) -> bool
```

Resolution refutation: returns `True` if the clause set is satisfiable.

### Forward Chain (Standalone)

```python
forward_chain(
    facts: Set[Predicate],
    rules: List[SymbolicRule],
    max_depth: int = 50,
) -> Tuple[Set[Predicate], List[LogicClause]]
```

### Betti Number

```python
betti_0(predicates: List[Predicate]) -> int
```

Connected components via Union-Find. $\beta_0 = 1$ means coherent;
$\beta_0 > 1$ signals disconnected reasoning.

---

## Backward Chainer / Prover (`nesy.symbolic.prover`)

Goal-directed SLD resolution (Prolog-style).

### Constructor

```python
BackwardChainer(rules: List[SymbolicRule], max_depth: int = 20)
```

### Prove a Goal

```python
prover.prove(
    goal: Predicate,
    facts: Set[Predicate],
    substitution: Optional[Substitution] = None,
) -> ProofResult
```

### ProofResult

```python
@dataclass
class ProofResult:
    goal: Predicate
    proved: bool
    proof_tree: Optional[ProofNode]
    substitution: Substitution
    depth_reached: int
    rules_used: List[str]
    confidence: float
```

`result.explain()` produces a human-readable proof tree string.

---

## Constraint Solver (`nesy.symbolic.solver`)

Wraps Z3 SMT solver (falls back to pure Python if Z3 is unavailable).

```python
solver = ConstraintSolver()
solver.set_value("temperature", 39.5)
solver.add_constraint(ArithmeticConstraint("temperature", ">=", 38.0))
satisfied, violations = solver.check_all()
```

### ArithmeticConstraint

```python
@dataclass
class ArithmeticConstraint:
    variable: str
    operator: str       # >=, <=, >, <, ==, !=, between
    value: object       # float or (low, high) tuple for 'between'
```

---

## CNF Normalizer (`nesy.symbolic.normalizer`)

Converts complex formulas to conjunctive normal form for resolution.

```python
from nesy.symbolic.normalizer import ComplexFormula, CNFNormalizer

f = ComplexFormula.IMPLIES(
    ComplexFormula.AND(ComplexFormula.atom(p1), ComplexFormula.atom(p2)),
    ComplexFormula.atom(p3),
)
clauses = CNFNormalizer().to_cnf(f)
```

---

## Rule Builder (`nesy.symbolic.rules`)

Fluent API for building rules:

```python
from nesy.symbolic.rules import RuleBuilder

rule = (
    RuleBuilder("my_rule")
    .if_fact("fever")
    .if_fact("cough")
    .then("flu_likely")
    .with_weight(0.85)
    .in_domain("medical")
    .build()
)
```

### RuleLoader

```python
RuleLoader.from_json("rules.json") -> List[SymbolicRule]
RuleLoader.to_json(rules, "rules.json") -> None
```

### RuleValidator

```python
errors = RuleValidator().validate(rules)  # returns List[str]
```

Checks: duplicates, circularity, weight range, predicate well-formedness.

---

## Betti Analyser (`nesy.symbolic.betti`)

```python
from nesy.symbolic.betti import BettiAnalyser

beta_0 = BettiAnalyser.compute(predicates)
coherence = BettiAnalyser.coherence_score(predicates)   # 1/β₀
components = BettiAnalyser.components(predicates)
diagnostic = BettiAnalyser.diagnose(predicates)

# From a reasoning trace
beta_0, coherence, diag = BettiAnalyser.from_trace(reasoning_steps)
```
