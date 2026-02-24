"""
nesy/core/validators.py
========================
Input validation utilities for NeSy-Core.

Validates:
    - Predicate structure (name, args, variable conventions)
    - Fact set consistency (no contradictions at input time)
    - ConceptEdge parameters (weight ranges, string non-empty)
    - SymbolicRule (weights, non-empty clauses, duplicates, immutables)
    - PresentSet / NullSet structural consistency
    - ConfidenceReport numerical bounds
    - Numerical stability (clamp, division guard)

These validators run at API boundaries, not in hot inference paths.
Call validate_* functions before feeding data to the engine.

All validation failures raise **NeSyError** subclasses already defined.
Silent failures are forbidden — every bad input produces a typed exception
with structured context.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Set, Tuple

from nesy.core.exceptions import NeSyError
from nesy.core.types import (
    ConceptEdge,
    ConfidenceReport,
    NSIOutput,
    NullSet,
    NullType,
    Predicate,
    PresentSet,
    SymbolicRule,
)


# ─── REGEX PATTERNS ───────────────────────────────────────────────

VALID_NAME_RE  = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
VARIABLE_RE    = re.compile(r'^\?[A-Za-z_][A-Za-z0-9_]*$')
GROUND_TERM_RE = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9_.\-]*$')


# ─── PREDICATE VALIDATION ─────────────────────────────────────────

def validate_predicate(p: Predicate) -> List[str]:
    """Validate a single predicate. Returns list of error strings.

    Checks:
        1. Name is non-empty and matches ``[A-Za-z_][A-Za-z0-9_]*``
        2. Every arg is non-empty
        3. Variable args match ``?[A-Za-z_][A-Za-z0-9_]*``
        4. Ground args match ``[A-Za-z0-9_][A-Za-z0-9_.-]*``
    """
    errors: List[str] = []

    if not p.name:
        errors.append("Predicate name is empty")
    elif not VALID_NAME_RE.match(p.name):
        errors.append(
            f"Predicate name '{p.name}' invalid: must be [A-Za-z_][A-Za-z0-9_]*"
        )

    for i, arg in enumerate(p.args):
        if not arg:
            errors.append(f"Predicate '{p.name}' arg[{i}] is empty")
        elif arg.startswith("?"):
            if not VARIABLE_RE.match(arg):
                errors.append(
                    f"Predicate '{p.name}' arg[{i}] '{arg}' looks like variable "
                    "but has invalid format. Expected ?[A-Za-z_][A-Za-z0-9_]*"
                )
        else:
            if not GROUND_TERM_RE.match(arg):
                errors.append(
                    f"Predicate '{p.name}' arg[{i}] '{arg}' has invalid characters. "
                    "Ground terms: [A-Za-z0-9_.-]+"
                )
    return errors


def validate_facts(facts: Set[Predicate]) -> List[str]:
    """Validate a set of facts. Returns list of errors.

    Facts must be fully grounded: no variables (``?foo``) allowed.
    """
    errors: List[str] = []
    for fact in facts:
        errors.extend(validate_predicate(fact))
        # Ground facts must have no variables
        for arg in fact.args:
            if arg.startswith("?"):
                errors.append(
                    f"Fact '{fact}' contains unbound variable '{arg}'. "
                    "Facts must be fully grounded."
                )
    return errors


# ─── RULE VALIDATION ──────────────────────────────────────────────

def validate_rule(rule: SymbolicRule) -> List[str]:
    """Validate a single SymbolicRule. Returns list of errors.

    Checks:
        1. id is non-empty and matches the name regex (hyphens allowed)
        2. weight ∈ (0, 1]
        3. Non-empty antecedents
        4. Non-empty consequents
        5. All antecedent/consequent predicates are individually valid
    """
    errors: List[str] = []

    if not rule.id:
        errors.append("Rule id is empty")
    elif not VALID_NAME_RE.match(rule.id.replace("-", "_")):
        errors.append(f"Rule id '{rule.id}' has invalid characters")

    if not (0.0 < rule.weight <= 1.0):
        errors.append(f"Rule '{rule.id}': weight {rule.weight} not in (0, 1]")

    if not rule.antecedents:
        errors.append(f"Rule '{rule.id}': no antecedents")
    else:
        for pred in rule.antecedents:
            errs = validate_predicate(pred)
            errors.extend([f"Rule '{rule.id}' antecedent: {e}" for e in errs])

    if not rule.consequents:
        errors.append(f"Rule '{rule.id}': no consequents")
    else:
        for pred in rule.consequents:
            errs = validate_predicate(pred)
            errors.extend([f"Rule '{rule.id}' consequent: {e}" for e in errs])

    return errors


def validate_rules_no_duplicates(rules: List[SymbolicRule]) -> List[str]:
    """Check that no two rules share the same id.

    Returns list of duplicate-id error strings.
    """
    errors: List[str] = []
    seen: dict[str, int] = {}
    for rule in rules:
        if rule.id in seen:
            errors.append(
                f"Duplicate rule id '{rule.id}' "
                f"(first at index {seen[rule.id]})"
            )
        else:
            seen[rule.id] = rules.index(rule)
    return errors


def validate_immutable_not_overwritten(
    existing_rules: List[SymbolicRule],
    new_rule: SymbolicRule,
) -> List[str]:
    """Check that an immutable rule is not being overwritten.

    Returns error if an existing rule with the same id is immutable.
    """
    errors: List[str] = []
    for existing in existing_rules:
        if existing.id == new_rule.id and existing.immutable:
            errors.append(
                f"Cannot overwrite immutable rule '{existing.id}' "
                f"(immutable=True, symbolic anchor)"
            )
    return errors


# ─── CONCEPT EDGE VALIDATION ──────────────────────────────────────

def validate_concept_edge(edge: ConceptEdge) -> List[str]:
    """Validate a ConceptEdge. Returns list of errors.

    Checks:
        1. source and target non-empty
        2. No self-loops
        3. cooccurrence_prob ∈ [0, 1]
        4. causal_strength ∈ {0.1, 0.5, 1.0}
        5. temporal_stability ∈ {0.1, 0.5, 1.0}
    """
    errors: List[str] = []

    if not edge.source:
        errors.append("ConceptEdge source is empty")
    if not edge.target:
        errors.append("ConceptEdge target is empty")
    if edge.source == edge.target:
        errors.append(f"ConceptEdge self-loop: {edge.source} → {edge.target}")

    if not (0.0 <= edge.cooccurrence_prob <= 1.0):
        errors.append(
            f"ConceptEdge {edge.source}→{edge.target}: "
            f"cooccurrence_prob {edge.cooccurrence_prob} not in [0,1]"
        )

    if edge.causal_strength not in (0.1, 0.5, 1.0):
        errors.append(
            f"ConceptEdge {edge.source}→{edge.target}: "
            f"causal_strength must be one of {{0.1, 0.5, 1.0}}"
        )

    if edge.temporal_stability not in (0.1, 0.5, 1.0):
        errors.append(
            f"ConceptEdge {edge.source}→{edge.target}: "
            f"temporal_stability must be one of {{0.1, 0.5, 1.0}}"
        )

    return errors


# ─── CONFIDENCE REPORT VALIDATION ────────────────────────────────

def validate_confidence_report(report: ConfidenceReport) -> List[str]:
    """Validate that all confidence scores are in [0, 1]."""
    errors: List[str] = []
    for name, val in [
        ("factual",            report.factual),
        ("reasoning",          report.reasoning),
        ("knowledge_boundary", report.knowledge_boundary),
    ]:
        if not (0.0 <= val <= 1.0):
            errors.append(f"ConfidenceReport.{name} = {val} not in [0,1]")
    return errors


# ─── PRESENT SET / NULL SET VALIDATION ────────────────────────────

def validate_present_set(ps: PresentSet) -> List[str]:
    """Validate a PresentSet.

    Checks:
        1. concepts is a non-empty set
        2. context_type is a non-empty string
    """
    errors: List[str] = []
    if not ps.concepts:
        errors.append("PresentSet concepts is empty")
    if not ps.context_type:
        errors.append("PresentSet context_type is empty")
    return errors


def validate_null_set(ns: NullSet) -> List[str]:
    """Validate a NullSet.

    Checks:
        1. No item in the null set has a concept that is also in the present set.
        2. NullItems are sorted by null_type (Type3 first, then Type2, then Type1).
    """
    errors: List[str] = []
    present_concepts = ns.present_set.concepts
    for item in ns.items:
        if item.concept in present_concepts:
            errors.append(
                f"NullSet contains '{item.concept}' which is also "
                f"in PresentSet — contradictory"
            )
    # Check ordering: Type3 > Type2 > Type1
    type_order = [item.null_type.value for item in ns.items]
    sorted_order = sorted(type_order, reverse=True)
    if type_order != sorted_order:
        errors.append(
            "NullSet items are not sorted by criticality "
            "(expected Type3 first, then Type2, then Type1)"
        )
    return errors


# ─── NUMERICAL STABILITY ─────────────────────────────────────────

def clamp_probability(value: float, label: str = "probability") -> float:
    """Clamp a value to [0, 1] with NaN/Inf guard.

    Args:
        value: Raw float value.
        label: Human-readable label for error messages.

    Returns:
        Clamped float in [0, 1].

    Raises:
        NeSyError: if value is NaN.
    """
    if math.isnan(value):
        raise NeSyError(
            f"{label} is NaN — cannot proceed",
            context={"label": label, "value": "NaN"},
        )
    if math.isinf(value):
        return 1.0 if value > 0 else 0.0
    return max(0.0, min(1.0, value))


def safe_divide(numerator: float, denominator: float, epsilon: float = 1e-8) -> float:
    """Division with zero-guard.

    Returns ``numerator / (denominator + epsilon)`` to prevent
    ``ZeroDivisionError`` in confidence computations.

    Args:
        numerator:   Dividend.
        denominator: Divisor.
        epsilon:     Small constant added to denominator.

    Returns:
        Safe quotient.
    """
    return numerator / (denominator + epsilon)


# ─── CONVENIENCE VALIDATORS ──────────────────────────────────────

def assert_valid_facts(facts: Set[Predicate]) -> None:
    """Validate facts and raise NeSyError on any violation."""
    errors = validate_facts(facts)
    if errors:
        raise NeSyError(
            f"Invalid facts: {'; '.join(errors)}",
            context={"error_count": len(errors)},
        )


def assert_valid_rule(rule: SymbolicRule) -> None:
    """Validate rule and raise NeSyError on any violation."""
    errors = validate_rule(rule)
    if errors:
        raise NeSyError(
            f"Invalid rule '{rule.id}': {'; '.join(errors)}",
            context={"rule_id": rule.id},
        )


def assert_valid_concept_edge(edge: ConceptEdge) -> None:
    """Validate concept edge and raise NeSyError on any violation."""
    errors = validate_concept_edge(edge)
    if errors:
        raise NeSyError(
            f"Invalid concept edge {edge.source}→{edge.target}: "
            f"{'; '.join(errors)}",
            context={"source": edge.source, "target": edge.target},
        )


def assert_valid_confidence(report: ConfidenceReport) -> None:
    """Validate confidence report and raise NeSyError on any violation."""
    errors = validate_confidence_report(report)
    if errors:
        raise NeSyError(
            f"Invalid confidence report: {'; '.join(errors)}",
            context={"error_count": len(errors)},
        )
