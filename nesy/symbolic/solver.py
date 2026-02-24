"""
nesy/symbolic/solver.py
=======================
Constraint solver layer — wraps Z3 SMT solver for hard constraint checking.

Z3 is used when Robinson's resolution is insufficient:
  - Arithmetic constraints  (age > 18, dosage <= max_safe)
  - Temporal constraints    (event_A before event_B)
  - Cardinality constraints (at_most_one(diagnosis))

Falls back gracefully if Z3 is not installed (pure resolution only).

Mathematical basis:
    SMT (Satisfiability Modulo Theories) extends SAT with:
      - Linear arithmetic over integers/reals
      - Uninterpreted functions
      - Arrays and bit-vectors

    NeSy-Core uses QF_LRA (Quantifier-Free Linear Real Arithmetic)
    for most medical/legal constraint checking.

    Reference: De Moura & Bjørner (2008) "Z3: An Efficient SMT Solver".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nesy.core.exceptions import NeSyError, SymbolicConflict
from nesy.core.types import Predicate, SymbolicRule

logger = logging.getLogger(__name__)

# Z3 is optional — graceful fallback if not installed
try:
    import z3

    Z3_AVAILABLE = True
except ImportError:  # pragma: no cover
    Z3_AVAILABLE = False
    logger.info(
        "Z3 not installed. Arithmetic constraints disabled. "
        "pip install z3-solver"
    )


VALID_OPERATORS = frozenset({">=", "<=", ">", "<", "==", "!=", "between"})


@dataclass
class ArithmeticConstraint:
    """A numeric constraint: variable op value.

    Examples:
        ArithmeticConstraint("age",    ">=", 18)
        ArithmeticConstraint("dosage", "<=", 500.0)
        ArithmeticConstraint("systolic_bp", "between", (90, 140))

    Raises:
        ValueError: if operator is not one of the valid operators.
    """

    variable: str
    operator: str  # ">=", "<=", ">", "<", "==", "!=", "between"
    value: object  # int | float | tuple(lo, hi) for "between"

    def __post_init__(self) -> None:
        if self.operator not in VALID_OPERATORS:
            raise ValueError(
                f"Invalid operator '{self.operator}'. "
                f"Must be one of {sorted(VALID_OPERATORS)}."
            )
        if self.operator == "between":
            if (
                not isinstance(self.value, (tuple, list))
                or len(self.value) != 2
            ):
                raise ValueError(
                    "'between' operator requires a (lo, hi) tuple as value."
                )
        if not self.variable:
            raise ValueError("Constraint variable name must be non-empty.")


@dataclass
class ConstraintResult:
    """Structured result of a constraint check.

    Attributes:
        satisfiable: True if all constraints can be simultaneously satisfied.
        violations:  Human-readable descriptions of violated constraints.
        model:       If satisfiable, a satisfying assignment {var → value}.
                     Empty dict when unsatisfiable or when Z3 is unavailable.
        constraint_ids: Indices of the constraints that contributed to violation
                        (only populated when Z3 is used with tracking).
        unsat_core:  Structured explanation object (populated on UNSAT).
    """

    satisfiable: bool
    violations: List[str] = field(default_factory=list)
    model: Dict[str, float] = field(default_factory=dict)
    constraint_ids: List[int] = field(default_factory=list)
    unsat_core: Optional[object] = None  # UnsatCore (avoid circular import)


class ConstraintSolver:
    """SMT constraint solver for NeSy-Core.

    Handles:
        1. Numeric constraint checking (dosage limits, vital sign ranges)
        2. Temporal ordering (event sequencing)
        3. Cardinality (at-most-one, exactly-one for mutually exclusive facts)
        4. Automatic conversion of immutable SymbolicRules to hard SMT
           constraints (constraint-to-rule bridge)

    If Z3 is unavailable, arithmetic constraints are checked via
    pure-Python fallback (no interaction effects between constraints).

    Mathematical basis (De Moura & Bjørner, 2008):
        Z3 decides the satisfiability of first-order formulas
        modulo background theories (here: QF_LRA — quantifier-free
        linear real arithmetic).

    Usage:
        solver = ConstraintSolver()
        solver.add_constraint(ArithmeticConstraint("dosage", "<=", 500))
        solver.set_value("dosage", 600)
        result = solver.check_all()
        assert not result.satisfiable
    """

    def __init__(self) -> None:
        self._constraints: List[ArithmeticConstraint] = []
        self._variable_values: Dict[str, float] = {}
        self._hard_rules: List[SymbolicRule] = []

    # ─── VARIABLE BINDING ──────────────────────────────────────────

    def set_value(self, variable: str, value: float) -> None:
        """Bind a numeric variable to its observed value.

        Args:
            variable: Name matching an ArithmeticConstraint.variable.
            value:    Observed numeric value for this variable.

        Raises:
            ValueError: if variable name is empty.
        """
        if not variable:
            raise ValueError("Variable name must be non-empty.")
        self._variable_values[variable] = value
        logger.debug("Set %s = %s", variable, value)

    def set_values(self, values: Dict[str, float]) -> None:
        """Bind multiple variables at once.

        Args:
            values: Mapping of variable names to observed values.
        """
        for var, val in values.items():
            self.set_value(var, val)

    # ─── CONSTRAINT REGISTRATION ───────────────────────────────────

    def add_constraint(self, constraint: ArithmeticConstraint) -> None:
        """Register a constraint for checking.

        Args:
            constraint: Arithmetic constraint to add.
        """
        self._constraints.append(constraint)
        logger.debug(
            "Added constraint: %s %s %s",
            constraint.variable,
            constraint.operator,
            constraint.value,
        )

    def add_constraints(self, constraints: List[ArithmeticConstraint]) -> None:
        """Register multiple constraints at once."""
        for c in constraints:
            self.add_constraint(c)

    def add_hard_rule(self, rule: SymbolicRule) -> None:
        """Register an immutable SymbolicRule as a hard SMT constraint.

        Only rules with weight >= 0.95 (hard constraints) are accepted.

        Args:
            rule: SymbolicRule with is_hard_constraint() == True.

        Raises:
            ValueError: if the rule is not a hard constraint.
        """
        if not rule.is_hard_constraint():
            raise ValueError(
                f"Rule '{rule.id}' has weight {rule.weight} < 0.95; "
                "only hard constraints can be added to SMT solver."
            )
        self._hard_rules.append(rule)
        logger.debug("Added hard rule to solver: %s", rule.id)

    # ─── CLEARING ──────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all constraints, values, and hard rules."""
        self._constraints.clear()
        self._variable_values.clear()
        self._hard_rules.clear()
        logger.debug("Solver cleared.")

    def clear_values(self) -> None:
        """Remove all variable bindings but keep constraints."""
        self._variable_values.clear()

    @property
    def constraint_count(self) -> int:
        """Total number of registered arithmetic constraints."""
        return len(self._constraints)

    @property
    def has_values(self) -> bool:
        """Whether any variable values are bound."""
        return len(self._variable_values) > 0

    # ─── MAIN CHECK ────────────────────────────────────────────────

    def check_all(self) -> ConstraintResult:
        """Check all registered constraints against bound values.

        Uses Z3 if available, otherwise falls back to pure-Python
        per-constraint evaluation.

        Returns:
            ConstraintResult with satisfiability, violations, and model.
        """
        if not self._constraints:
            logger.debug("No constraints to check — trivially satisfiable.")
            return ConstraintResult(satisfiable=True)

        if Z3_AVAILABLE:
            return self._check_with_z3()
        else:
            return self._check_pure_python()

    def require_z3(self) -> None:
        """Raise NeSyError if Z3 is not available.

        Call this in code paths that absolutely require SMT solving
        (e.g., complex constraint interactions, satisfiability proofs).

        Raises:
            NeSyError: if Z3 is not installed.
        """
        if not Z3_AVAILABLE:
            raise NeSyError(
                "Z3 SMT solver is required for this operation but is not "
                "installed. Install with: pip install z3-solver",
                context={"dependency": "z3-solver"},
            )

    # ─── PURE PYTHON FALLBACK ──────────────────────────────────────

    def _check_pure_python(self) -> ConstraintResult:
        """Fallback: check constraints one-by-one without Z3.

        Limitation: cannot detect interactions between constraints
        (e.g., x >= 10 AND x <= 5 are each individually plausible
        until a value is bound).  This fallback only evaluates
        constraints that have a bound variable.
        """
        violations: List[str] = []
        violation_ids: List[int] = []

        for idx, c in enumerate(self._constraints):
            val = self._variable_values.get(c.variable)
            if val is None:
                continue  # unbound — cannot evaluate

            ok = self._evaluate(val, c.operator, c.value)
            if not ok:
                violations.append(
                    f"Constraint violated: {c.variable} {c.operator} "
                    f"{c.value} (actual: {val})"
                )
                violation_ids.append(idx)

        satisfiable = len(violations) == 0

        # Build UnsatCore on violation
        unsat_core = None
        if not satisfiable:
            from nesy.symbolic.unsat_explanation import (
                explain_constraint_violations,
            )
            unsat_core = explain_constraint_violations(
                constraint_ids=violation_ids,
                violations=violations,
            )

        return ConstraintResult(
            satisfiable=satisfiable,
            violations=violations,
            model=dict(self._variable_values) if satisfiable else {},
            constraint_ids=violation_ids,
            unsat_core=unsat_core,
        )

    @staticmethod
    def _evaluate(val: float, op: str, bound: object) -> bool:
        """Evaluate a single constraint predicate.

        Args:
            val:   Observed variable value.
            op:    Comparison operator string.
            bound: Reference value (number or (lo, hi) tuple).

        Returns:
            True if constraint is satisfied.
        """
        if op == ">=":
            return val >= bound
        if op == "<=":
            return val <= bound
        if op == ">":
            return val > bound
        if op == "<":
            return val < bound
        if op == "==":
            return abs(val - bound) < 1e-10
        if op == "!=":
            return abs(val - bound) >= 1e-10
        if op == "between":
            lo, hi = bound
            return lo <= val <= hi
        return True  # pragma: no cover — unreachable due to __post_init__

    # ─── Z3 SMT SOLVER ────────────────────────────────────────────

    def _check_with_z3(self) -> ConstraintResult:
        """Full Z3 SMT check — detects constraint interactions.

        Algorithm:
            1. Create a Z3 Real variable for each constraint variable.
            2. Translate each ArithmeticConstraint into a Z3 assertion.
            3. Assert observed variable values as equality constraints.
            4. Call solver.check() for satisfiability.
            5. If SAT: extract satisfying model.
               If UNSAT: extract unsat core for violation explanation.

        Reference: De Moura & Bjørner (2008), "Z3: An Efficient SMT Solver".

        Returns:
            ConstraintResult with full model or violation details.
        """
        solver = z3.Solver()
        z3_vars: Dict[str, z3.ArithRef] = {}
        tracked: Dict[str, Tuple[int, ArithmeticConstraint]] = {}

        # Create Z3 variables and add tracked constraints
        for idx, c in enumerate(self._constraints):
            if c.variable not in z3_vars:
                z3_vars[c.variable] = z3.Real(c.variable)
            v = z3_vars[c.variable]
            label = z3.Bool(f"c_{idx}")
            tracked[f"c_{idx}"] = (idx, c)
            z3_expr = self._to_z3_expr(v, c)
            solver.assert_and_track(z3_expr, label)

        # Add observed variable bindings (not tracked — these are facts)
        for var, val in self._variable_values.items():
            if var in z3_vars:
                solver.add(z3_vars[var] == val)
            else:
                z3_vars[var] = z3.Real(var)
                solver.add(z3_vars[var] == val)

        result = solver.check()

        if result == z3.sat:
            z3_model = solver.model()
            model_dict: Dict[str, float] = {}
            for var_name, z3_var in z3_vars.items():
                val = z3_model.evaluate(z3_var, model_completion=True)
                model_dict[var_name] = self._z3_to_float(val)
            logger.debug("Z3 SAT — model: %s", model_dict)
            return ConstraintResult(satisfiable=True, model=model_dict)

        elif result == z3.unsat:
            core = solver.unsat_core()
            violations: List[str] = []
            constraint_ids: List[int] = []
            for label in core:
                label_name = str(label)
                if label_name in tracked:
                    idx, c = tracked[label_name]
                    constraint_ids.append(idx)
                    violations.append(
                        f"Constraint violated: {c.variable} {c.operator} "
                        f"{c.value}"
                    )
            # If core is empty, fallback to per-constraint check
            if not violations:
                for idx, c in enumerate(self._constraints):
                    val = self._variable_values.get(c.variable)
                    if val is not None:
                        ok = self._evaluate(val, c.operator, c.value)
                        if not ok:
                            violations.append(
                                f"Constraint violated: {c.variable} "
                                f"{c.operator} {c.value} (actual: {val})"
                            )
                            constraint_ids.append(idx)
            logger.info(
                "Z3 UNSAT — %d violation(s): %s", len(violations), violations
            )

            # Build UnsatCore from Z3 unsat core
            from nesy.symbolic.unsat_explanation import (
                explain_constraint_violations,
            )
            core_labels = [str(label) for label in core]
            unsat_core = explain_constraint_violations(
                constraint_ids=constraint_ids,
                violations=violations,
                constraint_labels=core_labels,
            )

            return ConstraintResult(
                satisfiable=False,
                violations=violations,
                constraint_ids=constraint_ids,
                unsat_core=unsat_core,
            )
        else:
            # z3.unknown — solver timeout or incomplete theory
            logger.warning(
                "Z3 returned UNKNOWN — treating as satisfiable."
            )
            return ConstraintResult(satisfiable=True)

    @staticmethod
    def _to_z3_expr(
        v: "z3.ArithRef", c: ArithmeticConstraint
    ) -> "z3.BoolRef":
        """Convert an ArithmeticConstraint to a Z3 boolean expression.

        Args:
            v: Z3 variable corresponding to the constraint's variable.
            c: The constraint to convert.

        Returns:
            Z3 BoolRef expressing the constraint.
        """
        if c.operator == ">=":
            return v >= c.value
        elif c.operator == "<=":
            return v <= c.value
        elif c.operator == ">":
            return v > c.value
        elif c.operator == "<":
            return v < c.value
        elif c.operator == "==":
            return v == c.value
        elif c.operator == "!=":
            return v != c.value
        elif c.operator == "between":
            lo, hi = c.value
            return z3.And(v >= lo, v <= hi)
        return v == v  # pragma: no cover

    @staticmethod
    def _z3_to_float(val: "z3.ExprRef") -> float:
        """Convert a Z3 value (RatNumRef or AlgebraicNumRef) to float.

        Z3 represents rationals as exact fractions. This extracts
        the numerator/denominator and computes a Python float.

        Edge case: if Z3 returns a non-numeric sort, default to 0.0.
        """
        try:
            if z3.is_rational_value(val):
                num = val.numerator_as_long()
                den = val.denominator_as_long()
                if den == 0:
                    return 0.0
                return float(num) / float(den)
            elif z3.is_int_value(val):
                return float(val.as_long())
            else:
                return float(val.as_decimal(10).rstrip("?"))
        except Exception:
            return 0.0

    # ─── CARDINALITY CONSTRAINTS ───────────────────────────────────

    def check_at_most_one(self, predicates: List[Predicate]) -> bool:
        """Verify that at most one predicate in the list is true.

        Used for mutually exclusive diagnoses or classifications.
        The caller passes only the predicates that are TRUE.

        Args:
            predicates: List of true predicates to check mutual exclusivity.

        Returns:
            True if at most one predicate is present (constraint met).
        """
        return len(predicates) <= 1

    def check_exactly_one(self, predicates: List[Predicate]) -> bool:
        """Verify that exactly one predicate in the list is true.

        Used for mandatory, mutually-exclusive selection constraints.

        Args:
            predicates: List of true predicates.

        Returns:
            True if exactly one predicate is present.
        """
        return len(predicates) == 1

    # ─── HARD RULE BRIDGE ──────────────────────────────────────────

    def check_hard_rules(
        self,
        facts: frozenset,
    ) -> Tuple[bool, List[str]]:
        """Check if current facts violate any registered hard rules.

        For each hard rule, if ALL antecedents are satisfied by facts
        but the consequent is a Contraindication predicate, the rule
        is reported as a violation.

        Args:
            facts: Set of ground Predicate instances.

        Returns:
            (all_passed, violation_messages)
        """
        from nesy.symbolic.logic import apply_substitution, unify

        violations: List[str] = []

        for rule in self._hard_rules:
            theta: Dict[str, str] = {}
            all_matched = True

            for ant in rule.antecedents:
                matched = False
                for fact in facts:
                    result = unify(ant, fact, dict(theta))
                    if result is not None:
                        theta = result
                        matched = True
                        break
                if not matched:
                    all_matched = False
                    break

            if all_matched:
                grounded_consequents = [
                    apply_substitution(c, theta) for c in rule.consequents
                ]
                for gc in grounded_consequents:
                    if gc.name.startswith("Contraindication"):
                        violations.append(
                            f"Hard constraint '{rule.id}' violated: "
                            f"{gc} (rule: {rule.description})"
                        )

        return (len(violations) == 0, violations)
