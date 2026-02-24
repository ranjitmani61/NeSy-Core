"""
nesy/continual/learner.py
=========================
ContinualLearner — Learns new information without forgetting old.

Mathematical basis (Elastic Weight Consolidation):

    Total loss = L_new(θ) + λ × Σᵢ Fᵢ(θᵢ - θ*ᵢ)²

    Where:
        L_new(θ):  standard loss on new task data
        θ*ᵢ:       parameter value after learning previous task
        Fᵢ:        Fisher information — importance of parameter i
                   High Fᵢ = this parameter is critical for old knowledge
                   Low Fᵢ  = this parameter can be safely modified
        λ:         consolidation strength (higher = more conservative)

    This prevents catastrophic forgetting by penalising changes
    to parameters that are important for previously learned tasks.

    NeSy-Core extension: symbolic facts are stored in SymbolicAnchor
    and are completely immutable — they bypass EWC entirely.
    The neural weights adapt; the symbolic truths never change.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from nesy.core.exceptions import ContinualLearningConflict
from nesy.core.types import SymbolicRule

logger = logging.getLogger(__name__)


@dataclass
class TaskSnapshot:
    """Snapshot of model state after completing a task."""

    task_id: str
    param_means: Dict[str, Any]  # parameter_name → value (θ*)
    fisher_diagonals: Dict[str, Any]  # parameter_name → Fisher information (Fᵢ)
    task_description: str = ""


class ContinualLearner:
    """Manages continual learning with EWC + symbolic anchoring.

    Usage:
        learner = ContinualLearner(lambda_ewc=1000.0)

        # After learning Task A:
        learner.consolidate("task_a", model, dataloader_a)

        # Now learn Task B — Task A knowledge is protected:
        loss = learner.ewc_loss(model) + standard_cross_entropy_loss
    """

    def __init__(self, lambda_ewc: float = 1000.0):
        """
        Args:
            lambda_ewc: EWC regularisation strength.
                        Higher = stronger protection of old knowledge.
                        Typical range: 100 – 10000.
        """
        self.lambda_ewc = lambda_ewc
        self._snapshots: List[TaskSnapshot] = []
        self._anchor: SymbolicAnchor = SymbolicAnchor()

    # ─── SYMBOLIC ANCHORS ──────────────────────────────────────────

    @property
    def anchor(self) -> "SymbolicAnchor":
        return self._anchor

    def add_symbolic_anchor(self, rule: SymbolicRule) -> None:
        """Add an immutable symbolic fact that will never be overwritten."""
        self._anchor.add(rule)

    # ─── EWC CONSOLIDATION ─────────────────────────────────────────

    def consolidate(
        self,
        task_id: str,
        model_params: Dict[str, Any],
        compute_fisher_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        task_description: str = "",
    ) -> None:
        """Consolidate knowledge from a completed task.

        Should be called AFTER a task is learned but BEFORE
        starting the next task.

        Args:
            task_id:           unique identifier for this task
            model_params:      current model parameters (θ*)
            compute_fisher_fn: function that computes diagonal Fisher
                               information for each parameter
            task_description:  human-readable task label
        """
        fisher = compute_fisher_fn(model_params)

        snapshot = TaskSnapshot(
            task_id=task_id,
            param_means=copy.deepcopy(model_params),
            fisher_diagonals=fisher,
            task_description=task_description,
        )
        self._snapshots.append(snapshot)
        logger.info(f"Consolidated task '{task_id}': {len(model_params)} parameters protected.")

    def ewc_penalty(
        self,
        current_params: Dict[str, Any],
    ) -> float:
        """Compute the EWC regularisation penalty.

        L_ewc = λ × Σ_tasks Σ_params Fᵢ × (θᵢ - θ*ᵢ)²

        This is ADDED to the task-specific loss during training.
        High penalty = current parameters have drifted far from
        consolidated task parameters, weighted by their importance.

        Returns scalar penalty (float).
        Framework-agnostic: caller is responsible for using this
        value in their training loop (works with PyTorch, JAX, etc.)
        """
        total_penalty = 0.0

        for snapshot in self._snapshots:
            for name, current_val in current_params.items():
                if name not in snapshot.param_means:
                    continue
                fisher_val = snapshot.fisher_diagonals.get(name, 0.0)
                mean_val = snapshot.param_means[name]

                # Δ² weighted by Fisher information
                if isinstance(current_val, (int, float)):
                    delta_sq = (current_val - mean_val) ** 2
                    total_penalty += fisher_val * delta_sq

        return self.lambda_ewc * total_penalty

    # ─── MEMORY REPLAY ─────────────────────────────────────────────

    def get_replay_rules(self) -> List[SymbolicRule]:
        """Return all consolidated symbolic anchors for replay.

        During new task training, replaying symbolic anchors ensures
        the symbolic reasoning layer never unlearns core knowledge.
        """
        return self._anchor.all_rules()

    # ─── DIAGNOSTICS ───────────────────────────────────────────────

    @property
    def consolidated_tasks(self) -> List[str]:
        return [s.task_id for s in self._snapshots]

    @property
    def anchor_count(self) -> int:
        return len(self._anchor)


# ─────────────────────────────────────────────
#  SYMBOLIC ANCHOR
# ─────────────────────────────────────────────


class SymbolicAnchor:
    """Immutable store for symbolic facts that must never be forgotten.

    This is the 'long-term memory' of NeSy-Core. Once a fact
    enters the SymbolicAnchor, it cannot be modified or removed
    by any learning process.

    Design principle:
        Neural weights can drift.
        Symbolic anchors are permanent.

    Examples of anchor-worthy facts:
        - "A patient cannot be prescribed Drug X if allergic to Drug X"
        - "Division by zero is undefined"
        - "An output claiming certainty > 1.0 is always invalid"
    """

    def __init__(self):
        self._rules: Dict[str, SymbolicRule] = {}
        self._readonly: bool = False

    def add(self, rule: SymbolicRule) -> None:
        """Add a rule as an immutable anchor.

        Once added, this rule will be automatically flagged
        as immutable, regardless of how it was originally created.
        """
        if rule.id in self._rules:
            raise ContinualLearningConflict(
                f"Anchor '{rule.id}' already exists and cannot be overwritten.",
                anchor_id=rule.id,
                attempted_override=str(rule),
            )
        # Force immutable flag
        object.__setattr__(rule, "immutable", True)
        self._rules[rule.id] = rule
        logger.debug(f"Symbolic anchor added: {rule.id}")

    def get(self, rule_id: str) -> Optional[SymbolicRule]:
        return self._rules.get(rule_id)

    def all_rules(self) -> List[SymbolicRule]:
        return list(self._rules.values())

    def contains(self, rule_id: str) -> bool:
        return rule_id in self._rules

    def __len__(self) -> int:
        return len(self._rules)

    def __repr__(self) -> str:
        return f"SymbolicAnchor({len(self._rules)} immutable rules)"
