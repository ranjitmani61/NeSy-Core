# Guide: Adding Rules

Symbolic rules are the backbone of NeSy-Core's reasoning engine. This guide
covers every way to define, load, validate, and manage them.

## Rule Anatomy

A `SymbolicRule` has:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier |
| `antecedents` | `List[Predicate]` | IF conditions |
| `consequents` | `List[Predicate]` | THEN conclusions |
| `weight` | `float` | Confidence weight ∈ (0, 1] |
| `domain` | `Optional[str]` | Domain tag |
| `immutable` | `bool` | If `True`, cannot be deleted (symbolic anchor) |
| `description` | `str` | Human-readable explanation |

A rule with `weight >= 0.95` is treated as a **hard constraint** and is
always checked during consistency validation.

## Method 1: Direct Construction

```python
from nesy.core.types import Predicate, SymbolicRule

rule = SymbolicRule(
    id="fever_flu",
    antecedents=[Predicate("fever"), Predicate("cough")],
    consequents=[Predicate("flu_likely")],
    weight=0.85,
    domain="medical",
    description="Fever + cough suggests flu",
)

model.add_rule(rule)
```

## Method 2: Fluent Builder

```python
from nesy.symbolic.rules import RuleBuilder

rule = (
    RuleBuilder("fever_flu")
    .if_fact("fever")
    .if_fact("cough")
    .then("flu_likely")
    .with_weight(0.85)
    .in_domain("medical")
    .description("Fever + cough suggests flu")
    .build()
)
```

Chain `.as_anchor()` to make the rule immutable.

## Method 3: Load from JSON

```json
[
  {
    "id": "fever_flu",
    "antecedents": [{"name": "fever", "args": []}],
    "consequents": [{"name": "flu_likely", "args": []}],
    "weight": 0.85,
    "domain": "medical"
  }
]
```

```python
from nesy.symbolic.rules import RuleLoader

rules = RuleLoader.from_json("rules.json")
model.add_rules(rules)
```

Save rules back:

```python
RuleLoader.to_json(model._symbolic_engine.rules, "rules_backup.json")
```

## Batch Loading

```python
rules = [
    SymbolicRule(id="r1", antecedents=[Predicate("a")],
                 consequents=[Predicate("b")], weight=0.9),
    SymbolicRule(id="r2", antecedents=[Predicate("b")],
                 consequents=[Predicate("c")], weight=0.8),
]
model.add_rules(rules)
```

## Anchored Rules (Continual Learning)

Anchored rules are **immutable** — they survive continual learning and
cannot be modified or deleted:

```python
# Option 1: set immutable=True in the rule
rule = SymbolicRule(id="safety", ..., immutable=True)
model.add_rule(rule)

# Option 2: use the anchor() method
model.anchor(rule)

# Option 3: via learn() with make_anchor=True
model.learn(rule, make_anchor=True)
```

Attempting to delete an anchored rule raises `SymbolicConflict`.

## Validating Rules

```python
from nesy.symbolic.rules import RuleValidator

errors = RuleValidator().validate(rules)
if errors:
    for err in errors:
        print(f"  ⚠ {err}")
```

Checks performed:
- Duplicate rule IDs
- Circular rule chains
- Weight out of range
- Predicate well-formedness

## Variables in Predicates

Predicates support first-order variables (prefix `?`):

```python
rule = SymbolicRule(
    id="parent_ancestor",
    antecedents=[Predicate("parent", ("?X", "?Y"))],
    consequents=[Predicate("ancestor", ("?X", "?Y"))],
    weight=1.0,
)
```

During forward chaining, variables are unified with concrete values via
Robinson's unification algorithm.

## Tips

- Use descriptive `id` values — they appear in reasoning traces.
- Keep weights honest: 0.6–0.8 for soft heuristics, 0.95+ for hard facts.
- Always validate before loading large rule sets from external sources.
- Anchor domain-critical safety rules to prevent accidental deletion.
