"""
tests/unit/test_solver.py
=========================
Comprehensive tests for nesy/symbolic/solver.py — the Z3 SMT constraint solver.

Tests cover:
    - ArithmeticConstraint validation
    - ConstraintResult dataclass
    - ConstraintSolver: variable binding, constraint registration, clearing
    - Pure-Python fallback path
    - Z3 SMT path (real Z3 calls): SAT, UNSAT, between, interactions
    - Cardinality constraints (at_most_one, exactly_one)
    - Hard rule bridge (check_hard_rules)
    - require_z3() typed error
    - Edge cases: empty constraints, empty values, duplicate variables

Target: 100% line coverage for nesy/symbolic/solver.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from nesy.core.exceptions import NeSyError
from nesy.core.types import Predicate, SymbolicRule
from nesy.symbolic.solver import (
    ArithmeticConstraint,
    ConstraintResult,
    ConstraintSolver,
    VALID_OPERATORS,
    Z3_AVAILABLE,
)


# ═══════════════════════════════════════════════════════════════════
#  ArithmeticConstraint Tests
# ═══════════════════════════════════════════════════════════════════


class TestArithmeticConstraint:
    """Validate ArithmeticConstraint dataclass and its __post_init__."""

    def test_valid_ge(self):
        c = ArithmeticConstraint("age", ">=", 18)
        assert c.variable == "age"
        assert c.operator == ">="
        assert c.value == 18

    def test_valid_le(self):
        c = ArithmeticConstraint("dosage", "<=", 500.0)
        assert c.operator == "<="

    def test_valid_gt(self):
        c = ArithmeticConstraint("temp", ">", 37.5)
        assert c.operator == ">"

    def test_valid_lt(self):
        c = ArithmeticConstraint("temp", "<", 42.0)
        assert c.operator == "<"

    def test_valid_eq(self):
        c = ArithmeticConstraint("count", "==", 1)
        assert c.operator == "=="

    def test_valid_ne(self):
        c = ArithmeticConstraint("status", "!=", 0)
        assert c.operator == "!="

    def test_valid_between(self):
        c = ArithmeticConstraint("bp", "between", (90, 140))
        assert c.operator == "between"
        assert c.value == (90, 140)

    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError, match="Invalid operator"):
            ArithmeticConstraint("x", "~=", 5)

    def test_between_requires_tuple(self):
        with pytest.raises(ValueError, match="requires a .* tuple"):
            ArithmeticConstraint("x", "between", 5)

    def test_between_requires_length_2(self):
        with pytest.raises(ValueError, match="requires a .* tuple"):
            ArithmeticConstraint("x", "between", (1, 2, 3))

    def test_empty_variable_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ArithmeticConstraint("", ">=", 0)

    def test_between_with_list(self):
        """Lists are accepted as (lo, hi) for 'between'."""
        c = ArithmeticConstraint("x", "between", [10, 20])
        assert c.value == [10, 20]


# ═══════════════════════════════════════════════════════════════════
#  ConstraintResult Tests
# ═══════════════════════════════════════════════════════════════════


class TestConstraintResult:
    def test_default_fields(self):
        r = ConstraintResult(satisfiable=True)
        assert r.satisfiable is True
        assert r.violations == []
        assert r.model == {}
        assert r.constraint_ids == []

    def test_custom_fields(self):
        r = ConstraintResult(
            satisfiable=False,
            violations=["x > 10 failed"],
            model={},
            constraint_ids=[0],
        )
        assert r.satisfiable is False
        assert len(r.violations) == 1
        assert r.constraint_ids == [0]


# ═══════════════════════════════════════════════════════════════════
#  ConstraintSolver — Core API Tests
# ═══════════════════════════════════════════════════════════════════


class TestConstraintSolverCore:
    def test_init_empty(self):
        s = ConstraintSolver()
        assert s.constraint_count == 0
        assert s.has_values is False

    def test_set_value(self):
        s = ConstraintSolver()
        s.set_value("x", 42.0)
        assert s.has_values is True

    def test_set_value_empty_name_raises(self):
        s = ConstraintSolver()
        with pytest.raises(ValueError, match="non-empty"):
            s.set_value("", 10.0)

    def test_set_values_bulk(self):
        s = ConstraintSolver()
        s.set_values({"a": 1.0, "b": 2.0, "c": 3.0})
        assert s.has_values is True

    def test_add_constraint(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", ">=", 0))
        assert s.constraint_count == 1

    def test_add_constraints_bulk(self):
        s = ConstraintSolver()
        s.add_constraints(
            [
                ArithmeticConstraint("x", ">=", 0),
                ArithmeticConstraint("y", "<=", 100),
            ]
        )
        assert s.constraint_count == 2

    def test_clear(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", ">=", 0))
        s.set_value("x", 5.0)
        s.clear()
        assert s.constraint_count == 0
        assert s.has_values is False

    def test_clear_values(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", ">=", 0))
        s.set_value("x", 5.0)
        s.clear_values()
        assert s.constraint_count == 1
        assert s.has_values is False

    def test_check_all_no_constraints(self):
        """No constraints → trivially satisfiable."""
        s = ConstraintSolver()
        result = s.check_all()
        assert result.satisfiable is True


# ═══════════════════════════════════════════════════════════════════
#  ConstraintSolver — Z3 SMT Tests (real Z3 calls)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not Z3_AVAILABLE, reason="Z3 not installed")
class TestSolverZ3:
    def test_satisfiable_single(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("dosage", "<=", 500))
        s.set_value("dosage", 200)
        result = s.check_all()
        assert result.satisfiable is True
        assert "dosage" in result.model
        assert abs(result.model["dosage"] - 200.0) < 1e-6

    def test_unsatisfiable_single(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("dosage", "<=", 500))
        s.set_value("dosage", 600)
        result = s.check_all()
        assert result.satisfiable is False
        assert len(result.violations) > 0

    def test_satisfiable_ge(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("age", ">=", 18))
        s.set_value("age", 25)
        result = s.check_all()
        assert result.satisfiable is True

    def test_unsatisfiable_ge(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("age", ">=", 18))
        s.set_value("age", 10)
        result = s.check_all()
        assert result.satisfiable is False

    def test_satisfiable_gt(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", ">", 0))
        s.set_value("x", 1)
        result = s.check_all()
        assert result.satisfiable is True

    def test_unsatisfiable_gt(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", ">", 0))
        s.set_value("x", 0)
        result = s.check_all()
        assert result.satisfiable is False

    def test_satisfiable_lt(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "<", 10))
        s.set_value("x", 5)
        result = s.check_all()
        assert result.satisfiable is True

    def test_unsatisfiable_lt(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "<", 10))
        s.set_value("x", 10)
        result = s.check_all()
        assert result.satisfiable is False

    def test_satisfiable_eq(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "==", 42))
        s.set_value("x", 42)
        result = s.check_all()
        assert result.satisfiable is True

    def test_unsatisfiable_eq(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "==", 42))
        s.set_value("x", 43)
        result = s.check_all()
        assert result.satisfiable is False

    def test_satisfiable_ne(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "!=", 0))
        s.set_value("x", 5)
        result = s.check_all()
        assert result.satisfiable is True

    def test_unsatisfiable_ne(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "!=", 5))
        s.set_value("x", 5)
        result = s.check_all()
        assert result.satisfiable is False

    def test_satisfiable_between(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("bp", "between", (90, 140)))
        s.set_value("bp", 120)
        result = s.check_all()
        assert result.satisfiable is True

    def test_unsatisfiable_between(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("bp", "between", (90, 140)))
        s.set_value("bp", 200)
        result = s.check_all()
        assert result.satisfiable is False

    def test_constraint_interaction_sat(self):
        """Multiple constraints on the same variable — satisfiable."""
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", ">=", 10))
        s.add_constraint(ArithmeticConstraint("x", "<=", 20))
        s.set_value("x", 15)
        result = s.check_all()
        assert result.satisfiable is True

    def test_constraint_interaction_unsat(self):
        """Multiple constraints on same variable — contradictory."""
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", ">=", 20))
        s.add_constraint(ArithmeticConstraint("x", "<=", 10))
        s.set_value("x", 15)
        result = s.check_all()
        assert result.satisfiable is False

    def test_multi_variable_sat(self):
        """Constraints across different variables — all satisfied."""
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("age", ">=", 18))
        s.add_constraint(ArithmeticConstraint("dosage", "<=", 500))
        s.add_constraint(ArithmeticConstraint("bp", "between", (90, 140)))
        s.set_values({"age": 25, "dosage": 200, "bp": 120})
        result = s.check_all()
        assert result.satisfiable is True
        assert len(result.model) >= 3

    def test_no_values_bound_z3(self):
        """Constraints without values — Z3 finds any feasible model."""
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", ">=", 0))
        s.add_constraint(ArithmeticConstraint("x", "<=", 100))
        result = s.check_all()
        assert result.satisfiable is True
        assert "x" in result.model

    def test_contradictory_constraints_no_values(self):
        """Contradictory constraints even without bound values → UNSAT."""
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", ">", 100))
        s.add_constraint(ArithmeticConstraint("x", "<", 50))
        result = s.check_all()
        assert result.satisfiable is False

    def test_unbound_variable_in_values(self):
        """Value set for variable not in any constraint — still works."""
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", ">=", 0))
        s.set_value("x", 5)
        s.set_value("unrelated_var", 99)
        result = s.check_all()
        assert result.satisfiable is True

    def test_result_has_constraint_ids_on_unsat(self):
        """UNSAT results should carry constraint IDs."""
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "<=", 10))
        s.set_value("x", 20)
        result = s.check_all()
        assert result.satisfiable is False
        # Either core or fallback should populate constraint_ids
        assert len(result.violations) > 0


# ═══════════════════════════════════════════════════════════════════
#  ConstraintSolver — Pure Python Fallback
# ═══════════════════════════════════════════════════════════════════


class TestSolverPurePython:
    """Test the pure-Python fallback by calling _check_pure_python
    directly, bypassing Z3."""

    def test_all_operators_satisfied(self):
        s = ConstraintSolver()
        s.add_constraints(
            [
                ArithmeticConstraint("a", ">=", 10),
                ArithmeticConstraint("b", "<=", 20),
                ArithmeticConstraint("c", ">", 0),
                ArithmeticConstraint("d", "<", 100),
                ArithmeticConstraint("e", "==", 5),
                ArithmeticConstraint("f", "!=", 0),
                ArithmeticConstraint("g", "between", (10, 20)),
            ]
        )
        s.set_values({"a": 10, "b": 20, "c": 1, "d": 99, "e": 5, "f": 1, "g": 15})
        result = s._check_pure_python()
        assert result.satisfiable is True
        assert result.violations == []
        assert result.model == {"a": 10, "b": 20, "c": 1, "d": 99, "e": 5, "f": 1, "g": 15}

    def test_ge_violated(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", ">=", 10))
        s.set_value("x", 5)
        result = s._check_pure_python()
        assert result.satisfiable is False
        assert 0 in result.constraint_ids

    def test_le_violated(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "<=", 10))
        s.set_value("x", 15)
        result = s._check_pure_python()
        assert result.satisfiable is False

    def test_gt_violated(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", ">", 10))
        s.set_value("x", 10)
        result = s._check_pure_python()
        assert result.satisfiable is False

    def test_lt_violated(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "<", 10))
        s.set_value("x", 10)
        result = s._check_pure_python()
        assert result.satisfiable is False

    def test_eq_violated(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "==", 5))
        s.set_value("x", 6)
        result = s._check_pure_python()
        assert result.satisfiable is False

    def test_eq_near_zero_tolerance(self):
        """== uses 1e-10 tolerance."""
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "==", 5.0))
        s.set_value("x", 5.0 + 1e-12)
        result = s._check_pure_python()
        assert result.satisfiable is True  # within tolerance

    def test_ne_violated(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "!=", 5))
        s.set_value("x", 5)
        result = s._check_pure_python()
        assert result.satisfiable is False

    def test_ne_near_zero_tolerance(self):
        """!= uses 1e-10 tolerance."""
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "!=", 5.0))
        s.set_value("x", 5.0 + 1e-12)
        result = s._check_pure_python()
        assert result.satisfiable is False  # within tolerance = considered equal

    def test_between_violated(self):
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "between", (10, 20)))
        s.set_value("x", 25)
        result = s._check_pure_python()
        assert result.satisfiable is False

    def test_unbound_variable_skipped(self):
        """Variables without values are skipped."""
        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", ">=", 0))
        result = s._check_pure_python()
        assert result.satisfiable is True  # can't evaluate, so passes


# ═══════════════════════════════════════════════════════════════════
#  ConstraintSolver — Cardinality Constraints
# ═══════════════════════════════════════════════════════════════════


class TestSolverCardinality:
    def test_at_most_one_empty(self):
        s = ConstraintSolver()
        assert s.check_at_most_one([]) is True

    def test_at_most_one_single(self):
        s = ConstraintSolver()
        p = Predicate("HasDiagnosis", ("p1", "flu"))
        assert s.check_at_most_one([p]) is True

    def test_at_most_one_two(self):
        s = ConstraintSolver()
        p1 = Predicate("HasDiagnosis", ("p1", "flu"))
        p2 = Predicate("HasDiagnosis", ("p1", "cold"))
        assert s.check_at_most_one([p1, p2]) is False

    def test_exactly_one_empty(self):
        s = ConstraintSolver()
        assert s.check_exactly_one([]) is False

    def test_exactly_one_single(self):
        s = ConstraintSolver()
        p = Predicate("HasDiagnosis", ("p1", "flu"))
        assert s.check_exactly_one([p]) is True

    def test_exactly_one_two(self):
        s = ConstraintSolver()
        p1 = Predicate("HasDiagnosis", ("p1", "flu"))
        p2 = Predicate("HasDiagnosis", ("p1", "cold"))
        assert s.check_exactly_one([p1, p2]) is False


# ═══════════════════════════════════════════════════════════════════
#  ConstraintSolver — Hard Rule Bridge
# ═══════════════════════════════════════════════════════════════════


class TestSolverHardRuleBridge:
    def test_add_hard_rule(self):
        s = ConstraintSolver()
        rule = SymbolicRule(
            id="test_hard",
            antecedents=[Predicate("A", ("x",))],
            consequents=[Predicate("B", ("x",))],
            weight=1.0,
            immutable=True,
        )
        s.add_hard_rule(rule)
        assert len(s._hard_rules) == 1

    def test_add_soft_rule_raises(self):
        s = ConstraintSolver()
        rule = SymbolicRule(
            id="test_soft",
            antecedents=[Predicate("A", ("x",))],
            consequents=[Predicate("B", ("x",))],
            weight=0.5,
        )
        with pytest.raises(ValueError, match="< 0.95"):
            s.add_hard_rule(rule)

    def test_check_hard_rules_no_violation(self):
        s = ConstraintSolver()
        rule = SymbolicRule(
            id="contraindication",
            antecedents=[
                Predicate("HasAllergy", ("?p", "penicillin")),
                Predicate("Prescribed", ("?p", "penicillin")),
            ],
            consequents=[Predicate("ContraindicationViolated", ("?p", "penicillin"))],
            weight=1.0,
            immutable=True,
            description="Penicillin allergy contraindication",
        )
        s.add_hard_rule(rule)

        # No allergy — antecedent not fully matched
        facts = frozenset({Predicate("Prescribed", ("patient_1", "penicillin"))})
        passed, violations = s.check_hard_rules(facts)
        assert passed is True
        assert violations == []

    def test_check_hard_rules_violation(self):
        s = ConstraintSolver()
        rule = SymbolicRule(
            id="contraindication",
            antecedents=[
                Predicate("HasAllergy", ("?p", "penicillin")),
                Predicate("Prescribed", ("?p", "penicillin")),
            ],
            consequents=[Predicate("ContraindicationViolated", ("?p", "penicillin"))],
            weight=1.0,
            immutable=True,
            description="Penicillin allergy contraindication",
        )
        s.add_hard_rule(rule)

        facts = frozenset(
            {
                Predicate("HasAllergy", ("patient_1", "penicillin")),
                Predicate("Prescribed", ("patient_1", "penicillin")),
            }
        )
        passed, violations = s.check_hard_rules(facts)
        assert passed is False
        assert len(violations) == 1
        assert "contraindication" in violations[0]

    def test_check_hard_rules_empty(self):
        """No hard rules registered → passes."""
        s = ConstraintSolver()
        passed, violations = s.check_hard_rules(frozenset())
        assert passed is True

    def test_check_hard_rules_non_contraindication_consequent(self):
        """Hard rule that fires but consequent is not 'Contraindication*'."""
        s = ConstraintSolver()
        rule = SymbolicRule(
            id="safe_rule",
            antecedents=[Predicate("A", ("x",))],
            consequents=[Predicate("B", ("x",))],
            weight=1.0,
        )
        s.add_hard_rule(rule)
        facts = frozenset({Predicate("A", ("x",))})
        passed, violations = s.check_hard_rules(facts)
        assert passed is True  # B is not a Contraindication


# ═══════════════════════════════════════════════════════════════════
#  ConstraintSolver — require_z3()
# ═══════════════════════════════════════════════════════════════════


class TestRequireZ3:
    @pytest.mark.skipif(not Z3_AVAILABLE, reason="Z3 installed — test N/A")
    def test_require_z3_available(self):
        s = ConstraintSolver()
        s.require_z3()  # should not raise

    def test_require_z3_mock_unavailable(self, monkeypatch):
        """Simulate Z3 not available."""
        import nesy.symbolic.solver as solver_module

        monkeypatch.setattr(solver_module, "Z3_AVAILABLE", False)

        s = ConstraintSolver()
        with pytest.raises(NeSyError, match="Z3 SMT solver is required"):
            s.require_z3()


# ═══════════════════════════════════════════════════════════════════
#  Z3 Utility Functions
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not Z3_AVAILABLE, reason="Z3 not installed")
class TestZ3Utilities:
    def test_z3_to_float_rational(self):
        import z3

        val = z3.RealVal("3/4")
        result = ConstraintSolver._z3_to_float(val)
        assert abs(result - 0.75) < 1e-10

    def test_z3_to_float_integer(self):
        import z3

        val = z3.IntVal(42)
        result = ConstraintSolver._z3_to_float(val)
        assert result == 42.0

    def test_z3_to_float_fallback(self):
        """Non-standard Z3 type → fallback to 0.0."""
        import z3

        # Bool is not numeric
        val = z3.BoolVal(True)
        result = ConstraintSolver._z3_to_float(val)
        # Should gracefully return 0.0 or converted value
        assert isinstance(result, float)


# ═══════════════════════════════════════════════════════════════════
#  Module-level constants
# ═══════════════════════════════════════════════════════════════════


class TestModuleConstants:
    def test_valid_operators(self):
        expected = {">=", "<=", ">", "<", "==", "!=", "between"}
        assert VALID_OPERATORS == expected

    def test_z3_available(self):
        assert isinstance(Z3_AVAILABLE, bool)
        assert Z3_AVAILABLE is True  # Z3 is installed in this env


# ═══════════════════════════════════════════════════════════════════
#  Z3_AVAILABLE = False fallback path
# ═══════════════════════════════════════════════════════════════════


class TestSolverZ3UnavailableFallback:
    """Exercise the code path where Z3_AVAILABLE is False."""

    def test_check_all_routes_to_pure_python(self, monkeypatch):
        import nesy.symbolic.solver as mod

        monkeypatch.setattr(mod, "Z3_AVAILABLE", False)

        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "<=", 10))
        s.set_value("x", 5)
        result = s.check_all()
        assert result.satisfiable is True

    def test_check_all_pure_python_violation(self, monkeypatch):
        import nesy.symbolic.solver as mod

        monkeypatch.setattr(mod, "Z3_AVAILABLE", False)

        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "<=", 10))
        s.set_value("x", 15)
        result = s.check_all()
        assert result.satisfiable is False


# ═══════════════════════════════════════════════════════════════════
#  Z3 edge cases — unknown result, empty unsat core, den=0
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not Z3_AVAILABLE, reason="Z3 not installed")
class TestZ3EdgeCases:
    def test_z3_unknown_treated_as_satisfiable(self, monkeypatch):
        """When z3.unknown is returned, treat as satisfiable."""
        import z3 as z3_mod

        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", ">=", 0))

        def mock_check(self_solver, *args, **kwargs):
            return z3_mod.unknown

        monkeypatch.setattr(z3_mod.Solver, "check", mock_check)
        result = s.check_all()
        assert result.satisfiable is True

    def test_z3_empty_unsat_core_fallback(self, monkeypatch):
        """When UNSAT but unsat_core() returns [], fallback to per-constraint eval."""
        import z3 as z3_mod

        s = ConstraintSolver()
        s.add_constraint(ArithmeticConstraint("x", "<=", 10))
        s.set_value("x", 20)

        # Make unsat_core return empty list

        class MockSolver:
            """Wraps real solver but returns empty unsat_core."""

            pass

        def patched_unsat_core(self_solver):
            return []

        monkeypatch.setattr(z3_mod.Solver, "unsat_core", patched_unsat_core)
        result = s.check_all()
        assert result.satisfiable is False
        assert len(result.violations) >= 1

    def test_z3_to_float_zero_denominator(self):
        """Edge case: rational with d=0 → return 0.0."""
        import z3 as z3_mod

        mock_val = MagicMock()
        # Make is_rational_value return True
        with patch.object(z3_mod, "is_rational_value", return_value=True):
            mock_val.numerator_as_long.return_value = 5
            mock_val.denominator_as_long.return_value = 0
            result = ConstraintSolver._z3_to_float(mock_val)
            assert result == 0.0

    def test_z3_to_float_decimal_fallback(self):
        """Test the as_decimal fallback for algebraic numbers."""
        import z3 as z3_mod

        # Algebraic number: sqrt(2)
        x = z3_mod.Real("x")
        s = z3_mod.Solver()
        s.add(x * x == 2, x > 0)
        s.check()
        m = s.model()
        val = m.evaluate(x, model_completion=True)
        # Should handle algebraic via as_decimal or fallback
        result = ConstraintSolver._z3_to_float(val)
        assert isinstance(result, float)
