"""
nesy/symbolic/logic.py
======================
First-order logic primitives and resolution engine.

Mathematical basis:
  - Implements Robinson's resolution algorithm for clause satisfiability
  - Uses unification for variable binding
  - Betti number computation for topological contradiction detection

Key operations:
  1. unify(p1, p2)       → substitution θ or None if unification fails
  2. resolve(c1, c2)     → resolvent clause or None
  3. is_satisfiable(kb)  → bool via resolution refutation
  4. betti_0(predicates) → number of connected components (contradiction proxy)
"""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from nesy.core.types import (
    LogicClause,
    LogicConnective,
    Predicate,
    SymbolicRule,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  UNIFICATION
# ─────────────────────────────────────────────

# Variable convention: strings starting with '?' are variables
# e.g. "?x", "?patient" — concrete values have no leading '?'

Substitution = Dict[str, str]   # variable → value


def is_variable(term: str) -> bool:
    return term.startswith("?")


def apply_substitution(predicate: Predicate, theta: Substitution) -> Predicate:
    """Apply substitution θ to a predicate, replacing variables with values."""
    new_args = tuple(theta.get(arg, arg) for arg in predicate.args)
    return Predicate(name=predicate.name, args=new_args)


def unify(p1: Predicate, p2: Predicate, theta: Optional[Substitution] = None) -> Optional[Substitution]:
    """Unify two predicates using Robinson's unification algorithm.
    
    Returns substitution θ such that apply(p1, θ) == apply(p2, θ),
    or None if unification is impossible (contradiction).
    
    Time complexity: O(n) where n = total argument count.
    
    Examples:
        unify(HasSymptom(?p, fever), HasSymptom(patient_1, fever))
        → {"?p": "patient_1"}
        
        unify(HasSymptom(?p, fever), HasSymptom(patient_1, cough))
        → None  (fever ≠ cough, both ground terms)
    """
    if theta is None:
        theta = {}

    if p1.name != p2.name or len(p1.args) != len(p2.args):
        return None

    for a1, a2 in zip(p1.args, p2.args):
        # Apply current substitution
        a1 = theta.get(a1, a1)
        a2 = theta.get(a2, a2)

        if a1 == a2:
            continue  # already unified

        if is_variable(a1):
            if _occurs_check(a1, a2, theta):
                return None  # circular reference — unification fails
            theta[a1] = a2

        elif is_variable(a2):
            if _occurs_check(a2, a1, theta):
                return None
            theta[a2] = a1

        else:
            return None   # two different ground terms — cannot unify

    return theta


def _occurs_check(var: str, term: str, theta: Substitution) -> bool:
    """Prevent circular substitutions: ?x → f(?x) is invalid."""
    if var == term:
        return True
    if term in theta:
        return _occurs_check(var, theta[term], theta)
    return False


# ─────────────────────────────────────────────
#  CLAUSE RESOLUTION
# ─────────────────────────────────────────────

def negate_predicate(p: Predicate) -> Predicate:
    """Negation: wrap in NOT or unwrap existing NOT."""
    if p.name.startswith("NOT_"):
        return Predicate(name=p.name[4:], args=p.args)
    return Predicate(name=f"NOT_{p.name}", args=p.args)


def resolve_clauses(
    clause_a: FrozenSet[Predicate],
    clause_b: FrozenSet[Predicate],
) -> Optional[FrozenSet[Predicate]]:
    """Resolution inference rule: given two clauses, produce resolvent.
    
    Two clauses resolve if one contains P and the other ¬P
    (after unification). The resolvent is their union minus {P, ¬P}.
    
    If resolvent is empty → contradiction proved (UNSAT).
    """
    for p_a in clause_a:
        neg_p_a = negate_predicate(p_a)
        for p_b in clause_b:
            theta = unify(neg_p_a, p_b)
            if theta is not None:
                # Apply substitution and form resolvent
                resolved_a = frozenset(
                    apply_substitution(p, theta)
                    for p in clause_a if p != p_a
                )
                resolved_b = frozenset(
                    apply_substitution(p, theta)
                    for p in clause_b if p != p_b
                )
                return resolved_a | resolved_b
    return None


def is_satisfiable(clauses: List[FrozenSet[Predicate]], max_steps: int = 1000) -> bool:
    """Determine satisfiability via resolution refutation.
    
    Algorithm: add negation of goal, then resolve.
    If empty clause found → original is UNSATISFIABLE (contradiction).
    If no new clauses can be derived → SATISFIABLE.
    
    This is a complete decision procedure for propositional logic.
    For FOL it is semi-decidable (will find contradiction if one exists,
    but may not terminate on satisfiable inputs — hence max_steps).
    """
    clause_set = list(clauses)
    
    for _ in range(max_steps):
        new_clauses = []
        pairs = list(itertools.combinations(clause_set, 2))
        
        for c1, c2 in pairs:
            resolvent = resolve_clauses(c1, c2)
            if resolvent is None:
                continue
            if len(resolvent) == 0:
                # Empty clause → contradiction found → UNSATISFIABLE
                return False
            if resolvent not in clause_set:
                new_clauses.append(resolvent)

        if not new_clauses:
            # No new clauses derivable → SATISFIABLE
            return True

        clause_set.extend(new_clauses)

    logger.warning("Resolution reached max_steps without conclusion. Assuming SATISFIABLE.")
    return True


# ─────────────────────────────────────────────
#  RULE → CLAUSE CONVERSION
# ─────────────────────────────────────────────

def rule_to_clause(rule: SymbolicRule) -> FrozenSet[Predicate]:
    """Convert a SymbolicRule to a clause in conjunctive normal form.
    
    Rule:   A ∧ B → C  (if A and B then C)
    CNF:    ¬A ∨ ¬B ∨ C  (equivalent disjunctive clause)
    """
    negated_antecedents = frozenset(
        negate_predicate(p) for p in rule.antecedents
    )
    consequents = frozenset(rule.consequents)
    return negated_antecedents | consequents


# ─────────────────────────────────────────────
#  BETTI NUMBER — TOPOLOGICAL CONTRADICTION CHECK
# ─────────────────────────────────────────────

def betti_0(predicates: List[Predicate]) -> int:
    """Compute Betti number β₀ = number of connected components.
    
    Mathematical basis:
        β₀ = |V| - |E|  for a forest (acyclic graph)
        β₀ = 1 means all predicates are in one connected reasoning chain
        β₀ > 1 means multiple DISCONNECTED reasoning chains exist
    
    In NeSy-Core, β₀ > 1 is a soft warning: the reasoning trace
    has independent chains that do not connect — possible deception
    or hidden information (NSI principle).
    
    Two predicates are connected if they share at least one argument.
    """
    if not predicates:
        return 0

    # Union-Find for connected components
    parent = {p: p for p in predicates}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # Two predicates are connected if they share an argument
    arg_to_predicates: Dict[str, List[Predicate]] = defaultdict(list)
    for p in predicates:
        for arg in p.args:
            if not is_variable(arg):      # only ground terms connect predicates
                arg_to_predicates[arg].append(p)

    for _, sharing_preds in arg_to_predicates.items():
        for i in range(len(sharing_preds) - 1):
            union(sharing_preds[i], sharing_preds[i + 1])

    # Count distinct roots
    roots = {find(p) for p in predicates}
    return len(roots)


# ─────────────────────────────────────────────
#  FORWARD CHAINING
# ─────────────────────────────────────────────

def forward_chain(
    facts: Set[Predicate],
    rules: List[SymbolicRule],
    max_depth: int = 50,
) -> Tuple[Set[Predicate], List[LogicClause]]:
    """Forward chaining inference: derive all provable conclusions.
    
    Algorithm:
        1. Start with known facts
        2. For each rule, check if all antecedents can be unified with facts
        3. If yes, add consequents to fact set
        4. Repeat until fixpoint (no new facts) or max_depth

    Returns:
        derived_facts: all facts provable from input + rules
        audit_trail:   list of LogicClause showing what was derived and why
    """
    derived = set(facts)
    audit_trail: List[LogicClause] = []
    
    for depth in range(max_depth):
        new_facts: Set[Predicate] = set()
        
        for rule in rules:
            # Try to unify all antecedents with current facts
            theta: Substitution = {}
            all_unified = True
            
            for ant in rule.antecedents:
                matched = False
                for fact in derived:
                    result = unify(ant, fact, dict(theta))
                    if result is not None:
                        theta = result
                        matched = True
                        break
                if not matched:
                    all_unified = False
                    break

            if all_unified:
                for consequent in rule.consequents:
                    grounded = apply_substitution(consequent, theta)
                    if grounded not in derived:
                        new_facts.add(grounded)
                        audit_trail.append(LogicClause(
                            predicates=[grounded],
                            connective=LogicConnective.IMPLIES,
                            satisfied=True,
                            weight=rule.weight,
                            source_rule=rule.id,
                        ))

        if not new_facts:
            break  # fixpoint reached
        derived |= new_facts

    return derived, audit_trail
