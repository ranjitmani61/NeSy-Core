"""
nesy/symbolic/unsat_explanation.py
==================================
Unsat-Core → Human Explanation engine.

Transforms raw unsatisfiability results (Z3 unsat cores, resolution
conflicts) into structured, human-readable explanations with
actionable repair suggestions.

This module is the bridge between the SMT/resolution layer and the
developer-facing API.  It turns NeSy-Core from a "validator" into a
"reasoning auditor + repair agent".

Architecture:
    ┌─────────────┐        ┌──────────────────────┐
    │ Z3 unsat    │───────▸│  explain_unsat_core() │──▸ UnsatCore
    │ core labels │        └──────────────────────┘
    └─────────────┘                    │
                                       ▼
    ┌─────────────┐        ┌──────────────────────┐
    │ Conflicting │───────▸│  propose_repairs()    │──▸ repair_actions
    │ rules list  │        └──────────────────────┘
    └─────────────┘                    │
                                       ▼
    ┌─────────────┐        ┌──────────────────────┐
    │ NSI null    │───────▸│  format_report()      │──▸ str
    │ set items   │        └──────────────────────┘
    └─────────────┘

Mathematical basis:
    Given an unsatisfiable formula Φ = {φ₁, φ₂, …, φₙ}, the unsat
    core C ⊆ Φ is a minimal subset such that ⋀C is UNSAT.  Each φᵢ
    maps to a SymbolicRule or ArithmeticConstraint.  We translate C
    into natural language by instantiating rule descriptions and
    identifying the *missing* predicate that would make ⋀C satisfiable.

    Reference:  Liffiton & Sakallah (2008),
    "Algorithms for Computing Minimal Unsatisfiable Subsets of Constraints".
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from nesy.core.types import (
    NullItem,
    NullSet,
    NullType,
    Predicate,
    SymbolicRule,
    UnsatCore,
)

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────
#  CORE EXPLANATION GENERATOR
# ───────────────────────────────────────────────────────────────────

def explain_unsat_core(
    conflicting_rule_ids: List[str],
    rules_db: Dict[str, SymbolicRule],
    constraint_ids: Optional[List[int]] = None,
    constraint_labels: Optional[List[str]] = None,
    facts: Optional[Set[Predicate]] = None,
) -> UnsatCore:
    """Build a structured ``UnsatCore`` with human-readable explanation.

    This is the main entry point.  Given a set of conflicting rule IDs
    (from the symbolic engine or Z3 unsat core), it:
      1. Looks up rule descriptions in the knowledge base.
      2. Identifies which antecedent predicates create the conflict.
      3. Generates a natural-language explanation.
      4. Proposes minimal additions that could resolve the conflict.

    Args:
        conflicting_rule_ids:
            Rule IDs that form the unsatisfiable core.
        rules_db:
            Knowledge base mapping ``rule.id → SymbolicRule``.
        constraint_ids:
            Numeric constraint indices (from ``ConstraintSolver``).
        constraint_labels:
            String labels from Z3's ``assert_and_track`` (debugging).
        facts:
            The set of ground predicates that triggered the conflict.

    Returns:
        ``UnsatCore`` with explanation, suggested additions, and
        repair actions.
    """
    constraint_ids = constraint_ids or []
    constraint_labels = constraint_labels or []
    facts = facts or set()

    # ── Collect rule descriptions ──────────────────────────────
    conflicting_rules: List[SymbolicRule] = []
    rule_descriptions: List[str] = []

    for rid in conflicting_rule_ids:
        rule = rules_db.get(rid)
        if rule is not None:
            conflicting_rules.append(rule)
            desc = rule.description or f"rule '{rid}'"
            rule_descriptions.append(desc)
        else:
            rule_descriptions.append(f"rule '{rid}' (not in KB)")

    # ── Generate explanation text ──────────────────────────────
    explanation = _generate_explanation_text(
        conflicting_rule_ids, rule_descriptions, conflicting_rules, facts
    )

    # ── Propose repairs ────────────────────────────────────────
    suggested, repair_actions = _propose_repairs(
        conflicting_rules, facts
    )

    core = UnsatCore(
        conflicting_rule_ids=list(conflicting_rule_ids),
        constraint_ids=list(constraint_ids),
        explanation=explanation,
        suggested_additions=suggested,
        repair_actions=repair_actions,
        raw_labels=list(constraint_labels),
    )

    logger.info("UnsatCore generated: %s", core.summary())
    return core


# ───────────────────────────────────────────────────────────────────
#  EXPLANATION TEXT GENERATOR
# ───────────────────────────────────────────────────────────────────

def _generate_explanation_text(
    rule_ids: List[str],
    rule_descriptions: List[str],
    rules: List[SymbolicRule],
    facts: Set[Predicate],
) -> str:
    """Produce natural-language contradiction narrative.

    Pattern:
        "These N rules cannot all be true together: [descriptions].
         The conflict arises because [antecedent analysis].
         Consider [repair suggestion]."
    """
    n = len(rule_ids)

    if n == 0:
        return "No conflicting rules identified."

    # ── Header ─────────────────────────────────────────────────
    if n == 1:
        header = f"Rule '{rule_ids[0]}' is violated by the current facts."
    elif n == 2:
        header = (
            f"These 2 rules cannot be true together: "
            f"'{rule_ids[0]}' and '{rule_ids[1]}'."
        )
    else:
        quoted = ", ".join(f"'{rid}'" for rid in rule_ids[:-1])
        header = (
            f"These {n} rules cannot all be true together: "
            f"{quoted}, and '{rule_ids[-1]}'."
        )

    # ── Rule details ───────────────────────────────────────────
    detail_lines: List[str] = []
    for rid, desc in zip(rule_ids, rule_descriptions):
        detail_lines.append(f"  • {rid}: {desc}")

    # ── Antecedent conflict analysis ───────────────────────────
    conflict_analysis = _analyse_antecedent_conflict(rules, facts)

    # ── Assemble ───────────────────────────────────────────────
    parts = [header]
    if detail_lines:
        parts.append("Rule details:")
        parts.extend(detail_lines)
    if conflict_analysis:
        parts.append(conflict_analysis)

    return "\n".join(parts)


def _analyse_antecedent_conflict(
    rules: List[SymbolicRule],
    facts: Set[Predicate],
) -> str:
    """Identify *why* the antecedents create a contradiction.

    Strategy:
        For each rule, find which antecedents are satisfied by facts.
        If a rule's antecedent is satisfied but its consequent
        contradicts another rule's antecedent, report that.
    """
    if not rules:
        return ""

    fact_names = {str(f) for f in facts}

    satisfied_rules: List[str] = []
    missing_antecedents: Dict[str, List[str]] = {}

    for rule in rules:
        matched = []
        unmatched = []
        for ant in rule.antecedents:
            ant_str = str(ant)
            # Check if any fact matches (including variable-bearing)
            if ant_str in fact_names or any(
                _predicate_matches(ant, f) for f in facts
            ):
                matched.append(ant_str)
            else:
                unmatched.append(ant_str)

        if unmatched:
            missing_antecedents[rule.id] = unmatched
        if matched:
            satisfied_rules.append(rule.id)

    lines: List[str] = []
    if satisfied_rules:
        lines.append(
            f"The conflict arises because the current facts satisfy "
            f"the antecedents of {', '.join(satisfied_rules)}, "
            f"producing contradictory consequents."
        )

    if missing_antecedents:
        for rid, missing in missing_antecedents.items():
            lines.append(
                f"Rule '{rid}' expects: {', '.join(missing)} "
                f"(not currently present)."
            )

    return "\n".join(lines)


def _predicate_matches(pattern: Predicate, fact: Predicate) -> bool:
    """Check if a pattern predicate matches a ground fact.

    Supports simple variable matching: args starting with '?'
    are treated as wildcards.
    """
    if pattern.name != fact.name:
        return False
    if len(pattern.args) != len(fact.args):
        return False
    for p_arg, f_arg in zip(pattern.args, fact.args):
        if p_arg.startswith("?"):
            continue  # variable — matches anything
        if p_arg != f_arg:
            return False
    return True


# ───────────────────────────────────────────────────────────────────
#  REPAIR PROPOSALS
# ───────────────────────────────────────────────────────────────────

def _propose_repairs(
    rules: List[SymbolicRule],
    facts: Set[Predicate],
) -> Tuple[List[str], List[Dict[str, str]]]:
    """Propose minimal additions that could resolve the conflict.

    Strategies:
        1. If a rule has unsatisfied antecedents, suggest adding them
           (so the rule fires and resolves the conflict).
        2. If two rules produce contradictory consequents, suggest
           removing one of the triggering facts.
        3. Identify consequent predicates that could act as guards.

    Returns:
        (suggested_additions, repair_actions)
    """
    suggested: List[str] = []
    repairs: List[Dict[str, str]] = []
    seen_concepts: set = set()

    for rule in rules:
        # Strategy 1: unsatisfied antecedents → suggest adding them
        for ant in rule.antecedents:
            if not any(_predicate_matches(ant, f) for f in facts):
                concept = _predicate_to_concept(ant)
                if concept not in seen_concepts:
                    seen_concepts.add(concept)
                    suggested.append(concept)
                    repairs.append({
                        "action": "add_fact",
                        "concept": concept,
                        "predicate": str(ant),
                        "reason": (
                            f"Adding '{concept}' would satisfy the antecedent "
                            f"of rule '{rule.id}', potentially resolving the "
                            f"conflict."
                        ),
                    })

        # Strategy 2: suggest guard predicates from consequents
        for con in rule.consequents:
            if con.name.startswith("Contraindication") or con.name.startswith("Not"):
                guard_concept = _predicate_to_concept(con)
                if guard_concept not in seen_concepts:
                    seen_concepts.add(guard_concept)
                    repairs.append({
                        "action": "add_guard",
                        "concept": guard_concept,
                        "predicate": str(con),
                        "reason": (
                            f"Explicitly asserting '{guard_concept}' could "
                            f"disambiguate the contradiction from rule "
                            f"'{rule.id}'."
                        ),
                    })

    return suggested, repairs


def _predicate_to_concept(pred: Predicate) -> str:
    """Extract a human-readable concept name from a predicate.

    Examples:
        Predicate("HasSymptom", ("?p", "fever"))  → "fever"
        Predicate("Likely", ("infection",))        → "infection"
        Predicate("DrugInteraction", ())           → "DrugInteraction"
    """
    # Use the last non-variable argument, or the predicate name
    for arg in reversed(pred.args):
        if not arg.startswith("?"):
            return arg
    return pred.name


# ───────────────────────────────────────────────────────────────────
#  UNSAT-CORE + NSI INTEGRATION  (for CFG bridge)
# ───────────────────────────────────────────────────────────────────

def enrich_with_null_set(
    core: UnsatCore,
    null_set: Optional[NullSet],
) -> UnsatCore:
    """Enrich an UnsatCore with NSI null-set insights.

    If the null set contains critical (Type3) items that overlap with
    the unsat core's suggested additions, promote them with higher
    priority.  Also add any critical nulls that aren't already in the
    suggestions.

    Args:
        core:     An existing ``UnsatCore`` (may have suggestions already).
        null_set: The NSI null set from the concept graph engine.

    Returns:
        A new ``UnsatCore`` with enriched suggestions and repairs.
    """
    if null_set is None:
        return core

    suggested = list(core.suggested_additions)
    repairs = list(core.repair_actions)
    seen = set(suggested)

    # Add critical nulls as high-priority repair suggestions
    for item in null_set.critical_items:
        if item.concept not in seen:
            seen.add(item.concept)
            suggested.insert(0, item.concept)  # prepend = higher priority
            repairs.insert(0, {
                "action": "add_critical_concept",
                "concept": item.concept,
                "predicate": item.concept,
                "reason": (
                    f"CRITICAL: '{item.concept}' is absent but causally "
                    f"necessary (expected because of: "
                    f"{', '.join(item.expected_because_of)}). "
                    f"Adding it may resolve the contradiction."
                ),
            })

    # Add meaningful nulls that overlap with suggested additions
    for item in null_set.meaningful_items:
        if item.concept in seen and item.concept not in [
            r.get("concept") for r in repairs
            if r.get("action") == "add_critical_concept"
        ]:
            repairs.append({
                "action": "confirm_meaningful_null",
                "concept": item.concept,
                "predicate": item.concept,
                "reason": (
                    f"'{item.concept}' appears in both the unsat core "
                    f"repair set and the NSI meaningful nulls — strong "
                    f"signal that this concept should be provided."
                ),
            })

    return UnsatCore(
        conflicting_rule_ids=core.conflicting_rule_ids,
        constraint_ids=core.constraint_ids,
        explanation=core.explanation,
        suggested_additions=suggested,
        repair_actions=repairs,
        raw_labels=core.raw_labels,
    )


# ───────────────────────────────────────────────────────────────────
#  FORMATTED REPORT (for explain_contradiction)
# ───────────────────────────────────────────────────────────────────

def format_contradiction_report(
    core: UnsatCore,
    include_repairs: bool = True,
) -> str:
    """Format a full human-readable contradiction report.

    This is the text shown to developers via
    ``NeSyModel.explain_contradiction()``.

    Sections:
        1. CONTRADICTION DETECTED (header)
        2. Explanation narrative
        3. Repair suggestions (optional)
        4. Summary line

    Args:
        core:            The ``UnsatCore`` to report on.
        include_repairs: Whether to include the repair section.

    Returns:
        Multi-line formatted string.
    """
    lines = [
        "=" * 60,
        "  CONTRADICTION DETECTED — Unsat-Core Analysis",
        "=" * 60,
        "",
        core.explanation,
        "",
    ]

    if include_repairs and core.has_repairs:
        lines.append("── Suggested Repairs ─────────────────────────────────────")
        for i, concept in enumerate(core.suggested_additions, 1):
            # Find corresponding repair action
            action_desc = ""
            for ra in core.repair_actions:
                if ra.get("concept") == concept:
                    action_desc = ra["reason"]
                    break
            if action_desc:
                lines.append(f"  {i}. {action_desc}")
            else:
                lines.append(f"  {i}. Consider adding '{concept}'.")
        lines.append("")

    lines.append(f"── {core.summary()}")
    lines.append("=" * 60)

    return "\n".join(lines)


# ───────────────────────────────────────────────────────────────────
#  CONSTRAINT SOLVER → UNSAT CORE BRIDGE
# ───────────────────────────────────────────────────────────────────

def explain_constraint_violations(
    constraint_ids: List[int],
    violations: List[str],
    constraint_labels: Optional[List[str]] = None,
) -> UnsatCore:
    """Build an ``UnsatCore`` from ``ConstraintSolver`` violations.

    Unlike the rule-based ``explain_unsat_core()``, this handles
    arithmetic constraint violations (dosage limits, vital sign
    ranges, temporal ordering).

    Args:
        constraint_ids:   Indices of violated constraints.
        violations:       Human-readable violation strings.
        constraint_labels: Z3 tracked labels (optional).

    Returns:
        ``UnsatCore`` with explanation derived from violation strings.
    """
    if not violations:
        return UnsatCore(explanation="No constraint violations.")

    n = len(violations)
    header = (
        f"{n} arithmetic constraint{'s' if n != 1 else ''} "
        f"{'are' if n != 1 else 'is'} violated:"
    )

    detail_lines = [f"  • {v}" for v in violations]

    explanation = "\n".join([header] + detail_lines)

    return UnsatCore(
        conflicting_rule_ids=[],
        constraint_ids=list(constraint_ids),
        explanation=explanation,
        raw_labels=list(constraint_labels or []),
    )
