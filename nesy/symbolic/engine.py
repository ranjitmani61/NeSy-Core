"""
nesy/symbolic/engine.py
=======================
Main SymbolicEngine: orchestrates rule loading, constraint checking,
forward chaining, and topological validation.

This is the 'brain' of the symbolic layer.
All other modules interact with it through a clean interface.
"""

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from nesy.core.exceptions import SymbolicConflict
from nesy.core.types import (
    LogicClause,
    Predicate,
    ReasoningStep,
    SymbolicRule,
)
from nesy.symbolic.logic import (
    betti_0,
    forward_chain,
    is_satisfiable,
    rule_to_clause,
)
from nesy.symbolic.unsat_explanation import explain_unsat_core

logger = logging.getLogger(__name__)


class SymbolicEngine:
    """Core symbolic reasoning engine for NeSy-Core.

    Responsibilities:
        1. Maintain a knowledge base (KB) of symbolic rules
        2. Perform forward-chaining inference
        3. Detect logical contradictions (hard constraints)
        4. Compute topological consistency (Betti numbers)
        5. Produce fully auditable reasoning traces

    Usage:
        engine = SymbolicEngine()
        engine.load_rules(medical_rules)
        derived, trace = engine.reason(facts)
    """

    def __init__(self, domain: Optional[str] = None):
        self.domain = domain
        self._rules: Dict[str, SymbolicRule] = {}  # id → rule
        self._immutable_rules: Set[str] = set()  # ids of anchor rules

    # ─── RULE MANAGEMENT ───────────────────────────────────────────

    def add_rule(self, rule: SymbolicRule) -> None:
        """Add a single rule to the knowledge base.

        Immutable rules (symbolic anchors) can never be removed
        or overwritten — they are the permanent logical bedrock.
        """
        if rule.id in self._immutable_rules:
            raise SymbolicConflict(
                f"Cannot overwrite immutable rule '{rule.id}'",
                conflicting_rules=[rule.id],
            )
        self._rules[rule.id] = rule
        if rule.immutable:
            self._immutable_rules.add(rule.id)
        logger.debug(f"Rule added: {rule.id} (weight={rule.weight}, immutable={rule.immutable})")

    def load_rules(self, rules: List[SymbolicRule]) -> None:
        for rule in rules:
            self.add_rule(rule)

    def remove_rule(self, rule_id: str) -> None:
        if rule_id in self._immutable_rules:
            raise SymbolicConflict(
                f"Cannot remove immutable rule '{rule_id}'",
                conflicting_rules=[rule_id],
            )
        self._rules.pop(rule_id, None)

    @property
    def rules(self) -> List[SymbolicRule]:
        return list(self._rules.values())

    @property
    def hard_rules(self) -> List[SymbolicRule]:
        return [r for r in self._rules.values() if r.is_hard_constraint()]

    # ─── CONTRADICTION DETECTION ───────────────────────────────────

    def check_consistency(self, facts: Set[Predicate]) -> bool:
        """Check if given facts are consistent with all HARD rules.

        Converts hard rules to CNF clauses, adds negated facts,
        and runs resolution. If empty clause derived → contradiction.

        When a contradiction is detected, builds an ``UnsatCore`` with
        human-readable explanation and repair suggestions, and attaches
        it to the raised ``SymbolicConflict``.

        Raises SymbolicConflict if inconsistency found.
        Returns True if consistent.
        """
        clauses: List[FrozenSet[Predicate]] = []

        # Add negations of all current facts as clauses
        for fact in facts:
            from nesy.symbolic.logic import negate_predicate

            clauses.append(frozenset([negate_predicate(fact)]))

        # Add hard rule clauses
        for rule in self.hard_rules:
            clauses.append(rule_to_clause(rule))

        satisfiable = is_satisfiable(clauses)
        if not satisfiable:
            # Find which rules conflict
            conflicting = [
                rule.id for rule in self.hard_rules if any(ant in facts for ant in rule.antecedents)
            ]

            # Build UnsatCore with explanation + repairs
            unsat_core = explain_unsat_core(
                conflicting_rule_ids=conflicting,
                rules_db=self._rules,
                facts=facts,
            )

            raise SymbolicConflict(
                "Hard constraint violation detected in symbolic reasoning.",
                conflicting_rules=conflicting,
                context={"fact_count": len(facts)},
                unsat_core=unsat_core,
            )
        return True

    # ─── MAIN REASONING ────────────────────────────────────────────

    def reason(
        self,
        facts: Set[Predicate],
        check_hard_constraints: bool = True,
        max_depth: int = 50,
    ) -> Tuple[Set[Predicate], List[ReasoningStep], float]:
        """Main reasoning entry point.

        Steps:
            1. Consistency check (hard constraints only)
            2. Forward chaining to derive all provable conclusions
            3. Topological analysis (Betti number)
            4. Build reasoning steps for audit trail

        Returns:
            derived_facts:  all facts provable from input + rules
            steps:          ordered reasoning steps for trace
            confidence:     symbolic confidence ∈ [0, 1]
        """
        if check_hard_constraints:
            self.check_consistency(facts)

        # Forward chain with ALL rules (hard + soft)
        derived_facts, audit_trail = forward_chain(
            facts=facts,
            rules=self.rules,
            max_depth=max_depth,
        )

        # Build human-readable reasoning steps
        steps = self._build_reasoning_steps(facts, audit_trail)

        # Compute symbolic confidence
        confidence = self._compute_symbolic_confidence(derived_facts, audit_trail)

        # Topological check
        all_predicates = list(derived_facts)
        b0 = betti_0(all_predicates)
        if b0 > 1:
            logger.warning(
                f"Betti β₀ = {b0}: reasoning has {b0} disconnected components. "
                "Possible hidden information or deceptive input."
            )
            # Soft penalty: reduce confidence proportionally
            confidence *= 1.0 / b0

        return derived_facts, steps, confidence

    # ─── PRIVATE HELPERS ───────────────────────────────────────────

    def _build_reasoning_steps(
        self,
        initial_facts: Set[Predicate],
        audit_trail: List[LogicClause],
    ) -> List[ReasoningStep]:
        steps = []

        # Step 0: initial facts
        steps.append(
            ReasoningStep(
                step_number=0,
                description=f"Initial facts: {', '.join(str(p) for p in initial_facts)}",
                rule_applied=None,
                predicates=list(initial_facts),
                confidence=1.0,
            )
        )

        # Subsequent steps from forward chaining
        for i, clause in enumerate(audit_trail, start=1):
            rule_id = clause.source_rule
            rule = self._rules.get(rule_id) if rule_id else None
            steps.append(
                ReasoningStep(
                    step_number=i,
                    description=(
                        f"Applied rule '{rule_id}': "
                        f"{rule.description if rule else 'derived'} → "
                        f"{', '.join(str(p) for p in clause.predicates)}"
                    ),
                    rule_applied=rule_id,
                    predicates=clause.predicates,
                    confidence=clause.weight,
                )
            )

        return steps

    def _compute_symbolic_confidence(
        self,
        derived_facts: Set[Predicate],
        audit_trail: List[LogicClause],
    ) -> float:
        """Symbolic confidence = weighted average of clause weights in the chain.

        A chain where all rules have weight=1.0 gives confidence=1.0.
        Soft rules (weight < 1.0) reduce the overall confidence.
        The minimum weight in the chain is the bottleneck (weakest link).
        """
        if not audit_trail:
            return 1.0  # no inference steps, original facts are ground truth

        weights = [clause.weight for clause in audit_trail]
        # Confidence is the product: each step must be reliable
        # (joint probability of a chain of soft inferences)
        product = 1.0
        for w in weights:
            product *= w
        # Normalise so a single soft step doesn't collapse confidence
        n = len(weights)
        return product ** (1.0 / n) if n > 0 else 1.0  # geometric mean
