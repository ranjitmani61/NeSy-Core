"""
nesy/neural/loss.py
===================
Symbolic-guided loss functions for NeSy-Core training.

These loss functions allow symbolic rules to directly shape
neural network training via differentiable penalties.

Mathematical basis:

1. SymbolicHingeLoss:
   L = Σᵣ wᵣ × max(0, margin - sat(r, ŷ))
   sat(r, ŷ) ∈ [0,1]: how well output ŷ satisfies rule r
   margin = 1.0 (standard hinge)

2. LukasiewiczLoss (t-norm based):
   Encodes soft logic: AND → product t-norm, OR → probabilistic sum
   L_AND(a,b) = max(0, a+b-1)   (Lukasiewicz t-norm)
   L_OR(a,b)  = min(1, a+b)
   
   This gives differentiable logical operators for training.

3. KBConstraintLoss:
   Penalises violation of knowledge base rules in output distributions.
   Particularly useful for sequence models where output must satisfy
   domain constraints at every step.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

from nesy.core.types import SymbolicRule


class SymbolicHingeLoss:
    """Hinge loss over symbolic rule satisfaction.
    
    For each violated rule, penalises proportional to:
        - rule weight (importance)
        - degree of violation (how far from satisfied)
    
    L_symbolic = Σᵣ wᵣ × max(0, 1 - satisfaction(r))
    """

    def __init__(self, rules: List[SymbolicRule], alpha: float = 1.0):
        self.rules = rules
        self.alpha = alpha

    def __call__(
        self,
        satisfaction_scores: Dict[str, float],  # rule_id → satisfaction ∈ [0,1]
    ) -> float:
        """
        Args:
            satisfaction_scores: {rule.id: how satisfied this rule is by current output}
        
        Returns:
            Scalar loss value (add to task loss)
        """
        total = 0.0
        for rule in self.rules:
            sat = satisfaction_scores.get(rule.id, 0.0)
            hinge = max(0.0, 1.0 - sat)
            total += rule.weight * hinge
        return self.alpha * total


class LukasiewiczLogic:
    """Differentiable logical operators using Lukasiewicz t-norms.
    
    Maps logical connectives to continuous [0,1] operations that
    are compatible with gradient descent.
    
    All inputs should be probabilities / confidences ∈ [0,1].
    """

    @staticmethod
    def AND(a: float, b: float) -> float:
        """Lukasiewicz t-norm: max(0, a+b-1)"""
        return max(0.0, a + b - 1.0)

    @staticmethod
    def OR(a: float, b: float) -> float:
        """Lukasiewicz t-conorm: min(1, a+b)"""
        return min(1.0, a + b)

    @staticmethod
    def NOT(a: float) -> float:
        """Logical negation: 1 - a"""
        return 1.0 - a

    @staticmethod
    def IMPLIES(a: float, b: float) -> float:
        """Lukasiewicz implication: min(1, 1-a+b)"""
        return min(1.0, 1.0 - a + b)

    @staticmethod
    def IFF(a: float, b: float) -> float:
        """Biconditional: min(IMPLIES(a,b), IMPLIES(b,a))"""
        return min(LukasiewiczLogic.IMPLIES(a, b), LukasiewiczLogic.IMPLIES(b, a))

    @classmethod
    def rule_satisfaction(
        cls,
        antecedent_scores: List[float],
        consequent_scores: List[float],
    ) -> float:
        """Compute satisfaction score for a single rule.
        
        Rule: A₁ ∧ A₂ ∧ ... → C₁ ∧ C₂ ∧ ...
        
        Satisfaction = IMPLIES(AND(antecedents), AND(consequents))
        """
        if not antecedent_scores:
            return 1.0
        if not consequent_scores:
            return 1.0

        # AND over all antecedents
        ant_score = antecedent_scores[0]
        for s in antecedent_scores[1:]:
            ant_score = cls.AND(ant_score, s)

        # AND over all consequents
        cons_score = consequent_scores[0]
        for s in consequent_scores[1:]:
            cons_score = cls.AND(cons_score, s)

        return cls.IMPLIES(ant_score, cons_score)


class KBConstraintLoss:
    """Knowledge Base constraint loss for sequence generation.
    
    Penalises output tokens/distributions that violate symbolic constraints.
    Useful for ensuring generated text/diagnoses/code obeys domain rules.
    
    L_kb = -log(P(output satisfies all rules))
         = -Σᵣ wᵣ × log(satisfaction(r, output))
    """

    def __init__(self, rules: List[SymbolicRule], epsilon: float = 1e-8):
        self.rules   = rules
        self.epsilon = epsilon   # numerical stability for log

    def __call__(
        self,
        satisfaction_scores: Dict[str, float],
    ) -> float:
        """Negative log-likelihood of rule satisfaction.
        
        Returns scalar. Minimising this encourages rule-consistent outputs.
        """
        total = 0.0
        for rule in self.rules:
            sat = max(self.epsilon, satisfaction_scores.get(rule.id, self.epsilon))
            total += rule.weight * (-math.log(sat))
        return total
