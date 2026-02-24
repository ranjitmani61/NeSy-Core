"""
nesy/continual/ewc.py
======================
Elastic Weight Consolidation — standalone module.

EWC prevents catastrophic forgetting by penalising changes to
parameters that were important for previously learned tasks.

Mathematical basis:
    L_total(θ) = L_B(θ) + Σᵢ (λ/2) Fᵢ (θᵢ - θ*ᵢ)²

    Where:
        L_B(θ):   loss on new task B
        Fᵢ:       Fisher information for parameter i
                  Fᵢ = E[ (∂ log P(y|x,θ) / ∂θᵢ)² ]
                  Measures how much parameter i affects predictions
        θ*ᵢ:      optimal parameter value after task A
        λ:        consolidation strength (higher = protect more)

    Fisher diagonal approximation:
        Fᵢ ≈ (1/n) Σₙ (∂ log P(yₙ|xₙ,θ) / ∂θᵢ)²

    This is computed empirically over the previous task's data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class EWCSnapshot:
    """Saved state after task completion for EWC penalty computation."""

    task_id: str
    theta_star: Dict[str, float]  # θ* per parameter
    fisher: Dict[str, float]  # Fᵢ per parameter


class EWCRegularizer:
    """Pure EWC implementation — framework-agnostic.

    Works with any parameter dict (PyTorch state_dict, custom dicts, etc.)
    Caller is responsible for extracting parameters as Dict[str, float].
    """

    def __init__(self, lambda_ewc: float = 1000.0):
        self.lambda_ewc = lambda_ewc
        self._snapshots: List[EWCSnapshot] = []

    def consolidate(
        self,
        task_id: str,
        params: Dict[str, float],
        fisher: Dict[str, float],
    ) -> None:
        """Record θ* and F after completing a task."""
        import copy

        self._snapshots.append(
            EWCSnapshot(
                task_id=task_id,
                theta_star=copy.deepcopy(params),
                fisher=copy.deepcopy(fisher),
            )
        )
        logger.info(f"EWC: consolidated task '{task_id}' ({len(params)} params)")

    def penalty(self, current_params: Dict[str, float]) -> float:
        """Compute Σ (λ/2) Fᵢ (θᵢ - θ*ᵢ)²"""
        total = 0.0
        for snap in self._snapshots:
            for name, θ_current in current_params.items():
                if name not in snap.theta_star:
                    continue
                F = snap.fisher.get(name, 0.0)
                Δθ = θ_current - snap.theta_star[name]
                total += F * (Δθ**2)
        return (self.lambda_ewc / 2.0) * total

    @staticmethod
    def estimate_fisher(
        params: Dict[str, float],
        grad_fn: Callable[[Dict[str, float]], Dict[str, float]],
        n_samples: int = 200,
    ) -> Dict[str, float]:
        """Estimate diagonal Fisher from gradient samples.

        F_i ≈ (1/n) Σₙ (∂ log P / ∂θᵢ)²

        grad_fn: function that returns gradients dict for one data sample
        """
        fisher_acc: Dict[str, float] = {k: 0.0 for k in params}
        for _ in range(n_samples):
            grads = grad_fn(params)
            for name, g in grads.items():
                if name in fisher_acc:
                    fisher_acc[name] += g**2
        return {k: v / n_samples for k, v in fisher_acc.items()}

    @property
    def consolidated_tasks(self) -> List[str]:
        return [s.task_id for s in self._snapshots]
