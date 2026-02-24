"""
nesy/metacognition/trace.py
============================
ReasoningTrace builder — constructs fully auditable derivation traces.
Every NeSy-Core output carries its complete reasoning history.
"""

from __future__ import annotations
from typing import List, Set
from nesy.core.types import (
    LogicClause,
    NullItem,
    Predicate,
    ReasoningStep,
    ReasoningTrace,
)


class TraceBuilder:
    """Builds a ReasoningTrace incrementally."""

    def __init__(self):
        self._steps: List[ReasoningStep] = []
        self._clauses: List[LogicClause] = []

    def add_initial_facts(self, facts: Set[Predicate]) -> "TraceBuilder":
        self._steps.append(
            ReasoningStep(
                step_number=0,
                description=f"Initial facts ({len(facts)}): "
                + ", ".join(sorted(str(p) for p in facts)),
                rule_applied=None,
                predicates=list(facts),
                confidence=1.0,
            )
        )
        return self

    def add_derived(
        self,
        predicate: Predicate,
        rule_id: str,
        rule_weight: float,
        description: str = "",
    ) -> "TraceBuilder":
        n = len(self._steps)
        self._steps.append(
            ReasoningStep(
                step_number=n,
                description=description or f"Rule '{rule_id}' derived: {predicate}",
                rule_applied=rule_id,
                predicates=[predicate],
                confidence=rule_weight,
            )
        )
        return self

    def add_clause(self, clause: LogicClause) -> "TraceBuilder":
        self._clauses.append(clause)
        return self

    def build(
        self,
        neural_confidence: float,
        symbolic_confidence: float,
        null_violations: List[NullItem],
    ) -> ReasoningTrace:
        return ReasoningTrace(
            steps=self._steps,
            rules_activated=[s.rule_applied for s in self._steps if s.rule_applied],
            neural_confidence=neural_confidence,
            symbolic_confidence=symbolic_confidence,
            null_violations=null_violations,
            logic_clauses=self._clauses,
        )
