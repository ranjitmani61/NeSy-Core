"""
nesy/symbolic/rules.py
======================
Rule management: definition helpers, YAML loading, validation.

Rules are the domain knowledge of NeSy-Core.
This module provides:
    1. RuleBuilder   — fluent API to construct rules programmatically
    2. RuleLoader    — load rules from YAML/JSON config files
    3. RuleValidator — validate rule sets for logical consistency
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from nesy.core.types import Predicate, SymbolicRule

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  RULE BUILDER  (fluent API)
# ─────────────────────────────────────────────


class RuleBuilder:
    """Fluent builder for SymbolicRule.

    Example:
        rule = (RuleBuilder("fever_infection")
                .if_fact("HasSymptom", "?p", "fever")
                .if_fact("HasLabResult", "?p", "elevated_wbc")
                .then("PossiblyHas", "?p", "bacterial_infection")
                .with_weight(0.85)
                .in_domain("medical")
                .description("Fever + WBC → infection")
                .build())
    """

    def __init__(self, rule_id: str):
        self._id = rule_id
        self._antecedents: List[Predicate] = []
        self._consequents: List[Predicate] = []
        self._weight = 1.0
        self._domain: Optional[str] = None
        self._immutable = False
        self._description = ""

    def if_fact(self, predicate_name: str, *args: str) -> "RuleBuilder":
        self._antecedents.append(Predicate(name=predicate_name, args=tuple(args)))
        return self

    def then(self, predicate_name: str, *args: str) -> "RuleBuilder":
        self._consequents.append(Predicate(name=predicate_name, args=tuple(args)))
        return self

    def with_weight(self, weight: float) -> "RuleBuilder":
        self._weight = weight
        return self

    def in_domain(self, domain: str) -> "RuleBuilder":
        self._domain = domain
        return self

    def as_anchor(self) -> "RuleBuilder":
        """Mark as immutable symbolic anchor."""
        self._immutable = True
        self._weight = 1.0  # anchors are always hard constraints
        return self

    def description(self, text: str) -> "RuleBuilder":
        self._description = text
        return self

    def build(self) -> SymbolicRule:
        if not self._antecedents:
            raise ValueError(f"Rule '{self._id}' has no antecedents.")
        if not self._consequents:
            raise ValueError(f"Rule '{self._id}' has no consequents.")
        return SymbolicRule(
            id=self._id,
            antecedents=self._antecedents,
            consequents=self._consequents,
            weight=self._weight,
            domain=self._domain,
            immutable=self._immutable,
            description=self._description,
        )


# ─────────────────────────────────────────────
#  RULE LOADER
# ─────────────────────────────────────────────


class RuleLoader:
    """Load rules from JSON or dict config.

    JSON format:
    [
      {
        "id": "fever_implies_test",
        "antecedents": [["HasSymptom", "?p", "fever"]],
        "consequents": [["RequiresTest", "?p", "blood_test"]],
        "weight": 0.8,
        "domain": "medical",
        "immutable": false,
        "description": "..."
      }
    ]
    """

    @classmethod
    def from_json(cls, path: str) -> List[SymbolicRule]:
        data = json.loads(Path(path).read_text())
        return cls.from_list(data)

    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> List[SymbolicRule]:
        rules = []
        for item in data:
            rule = SymbolicRule(
                id=item["id"],
                antecedents=[Predicate(name=a[0], args=tuple(a[1:])) for a in item["antecedents"]],
                consequents=[Predicate(name=c[0], args=tuple(c[1:])) for c in item["consequents"]],
                weight=item.get("weight", 1.0),
                domain=item.get("domain"),
                immutable=item.get("immutable", False),
                description=item.get("description", ""),
            )
            rules.append(rule)
        logger.info(f"Loaded {len(rules)} rules.")
        return rules

    @classmethod
    def to_json(cls, rules: List[SymbolicRule], path: str) -> None:
        data = [
            {
                "id": r.id,
                "antecedents": [[p.name] + list(p.args) for p in r.antecedents],
                "consequents": [[p.name] + list(p.args) for p in r.consequents],
                "weight": r.weight,
                "domain": r.domain,
                "immutable": r.immutable,
                "description": r.description,
            }
            for r in rules
        ]
        Path(path).write_text(json.dumps(data, indent=2))


# ─────────────────────────────────────────────
#  RULE VALIDATOR
# ─────────────────────────────────────────────


class RuleValidator:
    """Validate a rule set before loading into the engine.

    Checks:
        1. No duplicate IDs
        2. No circular dependencies (A→B and B→A in hard rules)
        3. All predicates are well-formed
        4. Weight values are in valid range
    """

    @classmethod
    def validate(cls, rules: List[SymbolicRule]) -> List[str]:
        """Returns list of validation error strings. Empty = valid."""
        errors = []
        seen_ids = set()

        for rule in rules:
            # Duplicate ID check
            if rule.id in seen_ids:
                errors.append(f"Duplicate rule id: '{rule.id}'")
            seen_ids.add(rule.id)

            # Weight range
            if not (0.0 < rule.weight <= 1.0):
                errors.append(f"Rule '{rule.id}': weight {rule.weight} not in (0,1]")

            # Predicate well-formedness
            for pred in rule.antecedents + rule.consequents:
                if not pred.name:
                    errors.append(f"Rule '{rule.id}': predicate with empty name")

        # Circular dependency check (hard rules only)
        hard_rules = [r for r in rules if r.is_hard_constraint()]
        errors.extend(cls._check_circular(hard_rules))

        return errors

    @classmethod
    def _check_circular(cls, rules: List[SymbolicRule]) -> List[str]:
        """Detect A→B and B→A loops in hard rules (problematic for resolution)."""
        errors = []
        # Build: consequent_predicate_name → rule_id
        consequent_map: Dict[str, str] = {}
        for rule in rules:
            for cons in rule.consequents:
                consequent_map[cons.name] = rule.id

        for rule in rules:
            for ant in rule.antecedents:
                if ant.name in consequent_map:
                    producer = consequent_map[ant.name]
                    if producer != rule.id:
                        # Check if that producer depends on this rule's consequents
                        producer_rule = next((r for r in rules if r.id == producer), None)
                        if producer_rule:
                            my_consequent_names = {c.name for c in rule.consequents}
                            producer_ant_names = {a.name for a in producer_rule.antecedents}
                            if my_consequent_names & producer_ant_names:
                                errors.append(f"Circular dependency: '{rule.id}' ↔ '{producer}'")
        return errors
