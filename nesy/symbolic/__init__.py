"""nesy/symbolic — Symbolic reasoning layer."""

from nesy.symbolic.engine import SymbolicEngine
from nesy.symbolic.logic import (
    unify,
    apply_substitution,
    negate_predicate,
    rule_to_clause,
    resolve_clauses,
    is_satisfiable,
    forward_chain,
    betti_0,
)
from nesy.symbolic.rules import RuleBuilder, RuleLoader, RuleValidator
from nesy.symbolic.solver import ConstraintSolver, ArithmeticConstraint
from nesy.symbolic.unsat_explanation import (
    explain_unsat_core,
    explain_constraint_violations,
    enrich_with_null_set,
    format_contradiction_report,
)

__all__ = [
    "SymbolicEngine",
    "RuleBuilder",
    "RuleLoader",
    "RuleValidator",
    "ConstraintSolver",
    "ArithmeticConstraint",
    "unify",
    "apply_substitution",
    "negate_predicate",
    "rule_to_clause",
    "resolve_clauses",
    "is_satisfiable",
    "forward_chain",
    "betti_0",
    "explain_unsat_core",
    "explain_constraint_violations",
    "enrich_with_null_set",
    "format_contradiction_report",
]
