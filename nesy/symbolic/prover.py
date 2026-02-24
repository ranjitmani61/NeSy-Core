"""
nesy/symbolic/prover.py
========================
Backward Chaining Prover — goal-directed proof search.

Unlike forward chaining (which derives everything possible),
backward chaining starts from a GOAL and works backward to find
a proof. Much more efficient when you have a specific hypothesis
to verify.

Mathematical basis:
    SLD Resolution (Selected Linear Definite clause resolution)
    — the foundation of Prolog-style reasoning.

    Proof of goal G:
        If G is a known fact → SUCCESS
        If ∃ rule H ← B₁, B₂, ..., Bₙ where unify(G, H) = θ:
            recursively prove B₁θ, B₂θ, ..., Bₙθ
        If no applicable rule → FAILURE

    Proof tree: each node is a subgoal, edges are rule applications.
    This gives us not just "provable or not" but a PROOF TREE
    that explains exactly HOW the conclusion was reached.

Compared to forward chaining:
    Forward: derives ALL provable facts (breadth-first, expensive)
    Backward: proves ONE specific goal (goal-directed, efficient)

    In NeSy-Core:
        Use forward chaining for: "what can we conclude from these facts?"
        Use backward chaining for: "can we prove this specific hypothesis?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from nesy.core.types import Predicate, SymbolicRule
from nesy.symbolic.logic import apply_substitution, unify

logger = logging.getLogger(__name__)


# Type alias
Substitution = Dict[str, str]


@dataclass
class ProofNode:
    """A node in the proof tree.

    goal:        The predicate we are trying to prove at this node.
    rule_used:   The rule that was used to reduce this goal.
    children:    Subgoals that had to be proven to prove this goal.
    substitution: Variable bindings at this step.
    """

    goal: Predicate
    rule_used: Optional[str] = None  # rule.id
    children: List["ProofNode"] = field(default_factory=list)
    substitution: Substitution = field(default_factory=dict)
    is_fact: bool = False  # True if proved directly from KB


@dataclass
class ProofResult:
    """Result of a backward chaining proof attempt."""

    goal: Predicate
    proved: bool
    proof_tree: Optional[ProofNode]
    substitution: Substitution
    depth_reached: int
    rules_used: List[str]  # in order of application
    confidence: float  # product of rule weights in proof

    def explain(self, indent: int = 0) -> str:
        """Generate human-readable proof explanation."""
        prefix = "  " * indent
        if not self.proved:
            return f"{prefix}FAILED: Cannot prove {self.goal}"
        return self._explain_node(self.proof_tree, indent)

    def _explain_node(self, node: Optional[ProofNode], indent: int) -> str:
        if node is None:
            return ""
        prefix = "  " * indent
        if node.is_fact:
            return f"{prefix}✓ {node.goal} [known fact]"
        lines = [f"{prefix}⊢ {node.goal} [via rule '{node.rule_used}']"]
        for child in node.children:
            lines.append(self._explain_node(child, indent + 1))
        return "\n".join(lines)


class BackwardChainer:
    """SLD-resolution backward chaining prover.

    Proves specific hypotheses against a knowledge base of facts + rules.

    Usage:
        prover = BackwardChainer(rules=medical_rules, max_depth=20)
        result = prover.prove(
            goal=Predicate("PossiblyHas", ("patient_1", "bacterial_infection")),
            facts={Predicate("HasSymptom", ("patient_1", "fever")), ...}
        )
        if result.proved:
            print(result.explain())
    """

    def __init__(
        self,
        rules: List[SymbolicRule],
        max_depth: int = 20,
    ):
        self.rules = rules
        self.max_depth = max_depth

    def prove(
        self,
        goal: Predicate,
        facts: Set[Predicate],
        substitution: Optional[Substitution] = None,
    ) -> ProofResult:
        """Attempt to prove a single goal.

        Returns ProofResult with:
            - proved: bool
            - proof_tree: full derivation tree if proved
            - confidence: product of rule weights in the proof
        """
        theta = substitution or {}
        node, final_theta, confidence, rules_used = self._prove_goal(
            goal=goal,
            facts=facts,
            theta=theta,
            depth=0,
        )

        proved = node is not None
        return ProofResult(
            goal=goal,
            proved=proved,
            proof_tree=node,
            substitution=final_theta,
            depth_reached=self.max_depth,
            rules_used=rules_used,
            confidence=confidence,
        )

    def prove_all(
        self,
        goals: List[Predicate],
        facts: Set[Predicate],
    ) -> Dict[str, ProofResult]:
        """Prove multiple goals and return results keyed by goal string."""
        return {str(g): self.prove(g, facts) for g in goals}

    def _prove_goal(
        self,
        goal: Predicate,
        facts: Set[Predicate],
        theta: Substitution,
        depth: int,
    ) -> Tuple[Optional[ProofNode], Substitution, float, List[str]]:
        """Recursive SLD resolution.

        Returns (node, substitution, confidence, rules_used) or (None, {}, 0.0, []) on failure.
        """
        if depth > self.max_depth:
            logger.debug(f"Proof depth {self.max_depth} exceeded for goal: {goal}")
            return None, {}, 0.0, []

        # Apply current substitution to goal
        grounded_goal = apply_substitution(goal, theta)

        # Base case: goal is a known fact
        for fact in facts:
            new_theta = unify(grounded_goal, fact, dict(theta))
            if new_theta is not None:
                node = ProofNode(goal=grounded_goal, is_fact=True, substitution=new_theta)
                return node, new_theta, 1.0, []

        # Recursive case: find a rule whose head unifies with goal
        for rule in self.rules:
            for consequent in rule.consequents:
                head_theta = unify(grounded_goal, consequent, dict(theta))
                if head_theta is None:
                    continue

                # Try to prove all antecedents
                all_proved = True
                child_nodes = []
                confidence = rule.weight
                rules_used = [rule.id]
                current_theta = head_theta

                for antecedent in rule.antecedents:
                    child_node, new_theta, child_conf, child_rules = self._prove_goal(
                        goal=antecedent,
                        facts=facts,
                        theta=current_theta,
                        depth=depth + 1,
                    )
                    if child_node is None:
                        all_proved = False
                        break

                    child_nodes.append(child_node)
                    current_theta = new_theta
                    confidence *= child_conf
                    rules_used.extend(child_rules)

                if all_proved:
                    node = ProofNode(
                        goal=grounded_goal,
                        rule_used=rule.id,
                        children=child_nodes,
                        substitution=current_theta,
                    )
                    return node, current_theta, confidence, rules_used

        # No proof found
        return None, {}, 0.0, []
