"""
nesy/symbolic/normalizer.py
============================
CNF (Conjunctive Normal Form) Normalizer for symbolic rules.

Before resolution-based reasoning, all rules must be in CNF:
    CNF: conjunction of disjunctions
    e.g. (A ∨ B) ∧ (¬C ∨ D) ∧ E
    
Conversion steps (Tseitin transformation + simplification):
    1. Eliminate biconditionals (A ↔ B → (A→B) ∧ (B→A))
    2. Eliminate implications (A → B → ¬A ∨ B)
    3. Push negations inward (De Morgan: ¬(A∧B) → ¬A∨¬B)
    4. Distribute OR over AND: A ∨ (B ∧ C) → (A∨B) ∧ (A∨C)
    5. Flatten nested conjunctions and disjunctions

For NeSy-Core:
    Rules are in implication form: A₁ ∧ A₂ → C₁ ∧ C₂
    CNF: ¬A₁ ∨ ¬A₂ ∨ C₁  and  ¬A₁ ∨ ¬A₂ ∨ C₂
    (handled by rule_to_clause in logic.py for simple rules)
    
This module handles COMPLEX rules with disjunctive antecedents,
nested implications, and biconditionals — rare in practice but
needed for legal and scientific knowledge bases.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Set, Tuple

from nesy.core.types import LogicConnective, Predicate, SymbolicRule
from nesy.symbolic.logic import negate_predicate, rule_to_clause

logger = logging.getLogger(__name__)


@dataclass
class ComplexFormula:
    """A formula that may contain nested logical structure.
    
    This goes beyond SymbolicRule which only supports simple implications.
    Used for: disjunctive rules, nested implications, biconditionals.
    
    Representation:
        predicate: leaf node (atomic formula)
        connective: logical operator for this node
        children:  sub-formulas
    """
    predicate:  Optional[Predicate] = None     # leaf
    connective: Optional[LogicConnective] = None  # internal node
    children:   List["ComplexFormula"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    @classmethod
    def atom(cls, p: Predicate) -> "ComplexFormula":
        return cls(predicate=p)

    @classmethod
    def NOT(cls, f: "ComplexFormula") -> "ComplexFormula":
        return cls(connective=LogicConnective.NOT, children=[f])

    @classmethod
    def AND(cls, *formulas: "ComplexFormula") -> "ComplexFormula":
        return cls(connective=LogicConnective.AND, children=list(formulas))

    @classmethod
    def OR(cls, *formulas: "ComplexFormula") -> "ComplexFormula":
        return cls(connective=LogicConnective.OR, children=list(formulas))

    @classmethod
    def IMPLIES(cls, antecedent: "ComplexFormula", consequent: "ComplexFormula") -> "ComplexFormula":
        return cls(connective=LogicConnective.IMPLIES, children=[antecedent, consequent])

    @classmethod
    def IFF(cls, left: "ComplexFormula", right: "ComplexFormula") -> "ComplexFormula":
        return cls(connective=LogicConnective.IFF, children=[left, right])

    def is_atom(self) -> bool:
        return self.predicate is not None


class CNFNormalizer:
    """Convert complex formulas to Conjunctive Normal Form.
    
    Usage:
        norm = CNFNormalizer()
        
        # Complex rule: A → (B ↔ C)
        formula = ComplexFormula.IMPLIES(
            ComplexFormula.atom(A),
            ComplexFormula.IFF(ComplexFormula.atom(B), ComplexFormula.atom(C))
        )
        
        cnf_clauses = norm.to_cnf(formula)
        # Returns list of frozenset[Predicate] — ready for resolution
    """

    def to_cnf(self, formula: ComplexFormula) -> List[FrozenSet[Predicate]]:
        """Convert a ComplexFormula to CNF clauses.
        
        Returns list of clauses. Each clause is a frozenset of Predicates
        (interpreted as disjunction). The full formula is conjunction of clauses.
        """
        # Step 1: Eliminate biconditionals
        f = self._eliminate_iff(formula)
        # Step 2: Eliminate implications
        f = self._eliminate_implies(f)
        # Step 3: Push negations inward (De Morgan)
        f = self._push_negations(f)
        # Step 4: Distribute OR over AND
        f = self._distribute(f)
        # Step 5: Flatten to clauses
        return self._flatten_to_clauses(f)

    def simple_rule_to_clauses(self, rule: SymbolicRule) -> List[FrozenSet[Predicate]]:
        """Convert a standard SymbolicRule to CNF clauses (fast path).
        
        For simple rules (conjunctive antecedent → conjunctive consequent),
        each consequent gets its own clause:
            ¬A₁ ∨ ¬A₂ ∨ ... ∨ Cᵢ
        """
        neg_ants = frozenset(negate_predicate(a) for a in rule.antecedents)
        return [neg_ants | frozenset([c]) for c in rule.consequents]

    # ── Step 1: Eliminate IFF ────────────────────────────────────────

    def _eliminate_iff(self, f: ComplexFormula) -> ComplexFormula:
        """A ↔ B → (A → B) ∧ (B → A)"""
        if f.is_atom():
            return f
        children = [self._eliminate_iff(c) for c in f.children]
        if f.connective == LogicConnective.IFF:
            a, b = children
            return ComplexFormula.AND(
                ComplexFormula.IMPLIES(a, b),
                ComplexFormula.IMPLIES(b, a),
            )
        return ComplexFormula(connective=f.connective, children=children)

    # ── Step 2: Eliminate IMPLIES ────────────────────────────────────

    def _eliminate_implies(self, f: ComplexFormula) -> ComplexFormula:
        """A → B → ¬A ∨ B"""
        if f.is_atom():
            return f
        children = [self._eliminate_implies(c) for c in f.children]
        if f.connective == LogicConnective.IMPLIES:
            a, b = children
            return ComplexFormula.OR(ComplexFormula.NOT(a), b)
        return ComplexFormula(connective=f.connective, children=children)

    # ── Step 3: Push negations inward ────────────────────────────────

    def _push_negations(self, f: ComplexFormula, negated: bool = False) -> ComplexFormula:
        """Apply De Morgan's laws to push NOT to atoms."""
        if f.is_atom():
            if negated:
                return ComplexFormula.atom(negate_predicate(f.predicate))
            return f

        if f.connective == LogicConnective.NOT:
            # Double negation elimination or push inward
            return self._push_negations(f.children[0], not negated)

        if negated:
            # De Morgan
            if f.connective == LogicConnective.AND:
                new_conn = LogicConnective.OR
            elif f.connective == LogicConnective.OR:
                new_conn = LogicConnective.AND
            else:
                new_conn = f.connective
            new_children = [self._push_negations(c, True) for c in f.children]
            return ComplexFormula(connective=new_conn, children=new_children)

        new_children = [self._push_negations(c, False) for c in f.children]
        return ComplexFormula(connective=f.connective, children=new_children)

    # ── Step 4: Distribute OR over AND ───────────────────────────────

    def _distribute(self, f: ComplexFormula) -> ComplexFormula:
        """Distribute OR over AND to get CNF."""
        if f.is_atom():
            return f
        children = [self._distribute(c) for c in f.children]
        if f.connective != LogicConnective.OR:
            return ComplexFormula(connective=f.connective, children=children)

        # We have an OR. If any child is an AND, distribute.
        result = children[0]
        for c in children[1:]:
            result = self._distribute_or_over_and(result, c)
        return result

    def _distribute_or_over_and(self, a: ComplexFormula, b: ComplexFormula) -> ComplexFormula:
        """(A ∨ (B ∧ C)) → (A ∨ B) ∧ (A ∨ C)"""
        if b.connective == LogicConnective.AND:
            return ComplexFormula.AND(*[
                self._distribute_or_over_and(a, bc)
                for bc in b.children
            ])
        if a.connective == LogicConnective.AND:
            return ComplexFormula.AND(*[
                self._distribute_or_over_and(ac, b)
                for ac in a.children
            ])
        return ComplexFormula.OR(a, b)

    # ── Step 5: Flatten to clause list ───────────────────────────────

    def _flatten_to_clauses(self, f: ComplexFormula) -> List[FrozenSet[Predicate]]:
        """Convert CNF formula to list of clauses (frozensets of predicates)."""
        if f.connective == LogicConnective.AND:
            clauses = []
            for child in f.children:
                clauses.extend(self._flatten_to_clauses(child))
            return clauses
        else:
            # Single clause (atom or OR of atoms)
            return [frozenset(self._collect_atoms(f))]

    def _collect_atoms(self, f: ComplexFormula) -> List[Predicate]:
        """Collect all atoms from a (possibly nested) OR formula."""
        if f.is_atom():
            return [f.predicate]
        atoms = []
        for child in f.children:
            atoms.extend(self._collect_atoms(child))
        return atoms
