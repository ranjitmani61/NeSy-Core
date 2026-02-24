"""
nesy/continual/replay.py
=========================
Experience replay strategies for continual learning.

Replay prevents catastrophic forgetting by interleaving stored
examples from previous tasks during new task training.

Three strategies provided:

1. **RandomReplay** — uniformly random samples from episodic buffer.
   Baseline strategy.  O(k) per sampling call.

2. **PrioritisedReplay** — samples proportional to priority weights.
   Uses Schaul et al. (2016) proportional prioritisation:

       P(i) = pᵢᵅ / Σⱼ pⱼᵅ          (sampling probability)
       wᵢ   = (N · P(i))^(−β)         (importance-sampling weight)

   Where:
       pᵢ = priority of item i  (e.g. TD-error or loss magnitude)
       α  ∈ [0, 1] — how much prioritisation is used (0 = uniform)
       β  ∈ [0, 1] — importance-sampling correction (1 = full correction)

   Reference: Schaul, Quan, Antonoglou, Silver (2016)
              "Prioritized Experience Replay", ICLR.

3. **SymbolicAnchorReplay** — always replays symbolic anchors alongside
   sampled episodic memories.  Guarantees that immutable logical
   constraints are never forgotten, regardless of neural weight drift.
"""

from __future__ import annotations

import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from nesy.continual.memory_buffer import EpisodicMemoryBuffer, MemoryItem
from nesy.core.types import SymbolicRule

logger = logging.getLogger(__name__)

__all__ = [
    "ReplayStrategy",
    "RandomReplay",
    "PrioritisedReplay",
    "SymbolicAnchorReplay",
]


# ─────────────────────────────────────────────
#  ABSTRACT BASE
# ─────────────────────────────────────────────


class ReplayStrategy(ABC):
    """Abstract base class for all replay strategies."""

    @abstractmethod
    def sample(self, n: int) -> List[MemoryItem]:
        """Sample *n* items for replay.

        Returns up to *n* items; may return fewer if the buffer
        contains fewer than *n* examples.
        """

    @property
    @abstractmethod
    def buffer_size(self) -> int:
        """Current number of items available for replay."""


# ─────────────────────────────────────────────
#  RANDOM REPLAY (Baseline)
# ─────────────────────────────────────────────


class RandomReplay(ReplayStrategy):
    """Uniformly random replay from an episodic memory buffer.

    Each stored item has equal probability of being selected,
    regardless of age, importance, or task origin.

    The ``replay_ratio`` controls what fraction of requested samples
    is actually returned — e.g. ``replay_ratio=0.2`` with ``n=20``
    returns $\\lfloor 0.2 \\times 20 \\rfloor = 4$ items.

    Usage:
        buffer = EpisodicMemoryBuffer(max_size=500)
        replay = RandomReplay(buffer, replay_ratio=0.5)
        batch  = replay.sample(32)   # returns 16 items
    """

    def __init__(self, buffer: EpisodicMemoryBuffer, replay_ratio: float = 1.0) -> None:
        assert 0.0 < replay_ratio <= 1.0, "replay_ratio must be in (0, 1]"
        self._buffer = buffer
        self._replay_ratio = replay_ratio

    def sample(self, n: int) -> List[MemoryItem]:
        effective_n = max(1, int(n * self._replay_ratio))
        return self._buffer.sample(effective_n)

    @property
    def buffer_size(self) -> int:
        return self._buffer.size


# ─────────────────────────────────────────────
#  PRIORITISED REPLAY
# ─────────────────────────────────────────────


@dataclass
class PrioritisedItem:
    """An item with an associated priority score."""

    item: MemoryItem
    priority: float = 1.0  # |δ| or loss — higher = more important


class PrioritisedReplay(ReplayStrategy):
    """Priority-weighted experience replay.

    Sampling probability:
        P(i) = pᵢᵅ / Σⱼ pⱼᵅ

    Importance-sampling weights (to correct for non-uniform sampling):
        wᵢ = (N · P(i))^(−β)
        Normalised: wᵢ / max(w)

    Parameters:
        alpha:    ∈ [0, 1]  0 → uniform,   1 → greedy priority
        beta:     ∈ [0, 1]  0 → no IS correction,  1 → full correction
        max_size: maximum buffer capacity
        epsilon:  small constant added to priorities to avoid zero probability

    Reference: Schaul et al. (2016) "Prioritized Experience Replay"
    """

    def __init__(
        self,
        max_size: int = 500,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
    ) -> None:
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        assert 0.0 <= beta <= 1.0, "beta must be in [0, 1]"
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self._items: List[PrioritisedItem] = []

    # ─── ADD / UPDATE ──────────────────────────────────────────────

    def add(self, item: MemoryItem, priority: float = 1.0) -> None:
        """Add an item with the given priority.

        If buffer is full, replace the item with the lowest priority.
        """
        p_item = PrioritisedItem(item=item, priority=max(priority, self.epsilon))

        if len(self._items) < self.max_size:
            self._items.append(p_item)
        else:
            # Replace lowest-priority item
            min_idx = min(range(len(self._items)), key=lambda i: self._items[i].priority)
            if p_item.priority > self._items[min_idx].priority:
                self._items[min_idx] = p_item

    def update_priority(self, index: int, new_priority: float) -> None:
        """Update priority of item at *index* (e.g. after computing new TD-error)."""
        if 0 <= index < len(self._items):
            self._items[index].priority = max(new_priority, self.epsilon)

    # ─── SAMPLING ──────────────────────────────────────────────────

    def sample(self, n: int) -> List[MemoryItem]:
        """Sample *n* items with probability proportional to pᵢᵅ.

        Returns plain ``MemoryItem`` objects.
        For importance-sampling weights, use ``sample_with_weights``.
        """
        items, _, _ = self.sample_with_weights(n)
        return items

    def sample_with_weights(self, n: int) -> Tuple[List[MemoryItem], List[float], List[int]]:
        """Sample *n* items and return IS weights and indices.

        Returns:
            items:   selected MemoryItem objects
            weights: normalised importance-sampling weights
            indices: buffer positions (for later priority updates)
        """
        if not self._items:
            return [], [], []

        n = min(n, len(self._items))

        # Compute sampling probabilities: P(i) = pᵢᵅ / Σ pⱼᵅ
        powered = [it.priority**self.alpha for it in self._items]
        total = sum(powered)
        probs = [p / total for p in powered]

        # Weighted sampling without replacement via sequential draws
        indices = self._weighted_sample_indices(probs, n)

        # Importance-sampling weights: wᵢ = (N · P(i))^(-β)
        N = len(self._items)
        raw_weights = [(N * probs[i]) ** (-self.beta) for i in indices]
        max_w = max(raw_weights) if raw_weights else 1.0
        # Normalise so max weight = 1  (stability)
        weights = [w / max_w for w in raw_weights]

        items = [self._items[i].item for i in indices]
        return items, weights, indices

    @staticmethod
    def _weighted_sample_indices(probs: List[float], n: int) -> List[int]:
        """Sample *n* distinct indices proportional to *probs*.

        Falls back to uniform if numerical issues arise.
        """
        if n >= len(probs):
            return list(range(len(probs)))

        chosen: List[int] = []
        remaining = list(range(len(probs)))
        rem_probs = list(probs)

        for _ in range(n):
            total = sum(rem_probs)
            if total <= 0:
                # Degenerate — fall back to uniform
                idx = random.choice(remaining)
            else:
                normalised = [p / total for p in rem_probs]
                r = random.random()
                cumulative = 0.0
                idx = remaining[-1]  # fallback
                for j, (ri, pr) in enumerate(zip(remaining, normalised)):
                    cumulative += pr
                    if r <= cumulative:
                        idx = ri
                        # Remove this index from candidates
                        remaining.pop(j)
                        rem_probs.pop(j)
                        break
                else:
                    # Edge case: floating-point — pick last
                    remaining.pop(-1)
                    rem_probs.pop(-1)
            chosen.append(idx)

        return chosen

    @property
    def buffer_size(self) -> int:
        return len(self._items)

    @property
    def max_priority(self) -> float:
        if not self._items:
            return 1.0
        return max(it.priority for it in self._items)


# ─────────────────────────────────────────────
#  SYMBOLIC ANCHOR REPLAY
# ─────────────────────────────────────────────


class SymbolicAnchorReplay(ReplayStrategy):
    """Replay strategy that always includes symbolic anchors.

    Can operate in two modes:

    1. **Standalone** (anchor only): replays anchor rules directly
       into a ``SymbolicEngine`` via ``replay_to_engine()``.
    2. **Combined** (anchor + buffer): every ``sample`` call returns
       all symbolic anchor rules plus random episodic memories
       filling the remaining budget.

    This guarantees immutable domain constraints are never forgotten,
    regardless of how much the neural backbone drifts.

    Usage (standalone):
        anchor = SymbolicAnchor()
        anchor.add(rule)
        replay = SymbolicAnchorReplay(anchor)
        n = replay.replay_to_engine(engine)   # re-adds anchor rules

    Usage (combined):
        replay = SymbolicAnchorReplay(anchor, buffer=buffer)
        batch  = replay.sample(32)
    """

    def __init__(
        self,
        anchor: Any,  # SymbolicAnchor — typed as Any to avoid circular import
        buffer: Optional[EpisodicMemoryBuffer] = None,
        anchor_task_id: str = "__symbolic_anchor__",
    ) -> None:
        self._anchor = anchor
        self._buffer = buffer
        self._anchor_task_id = anchor_task_id

    def sample(self, n: int) -> List[MemoryItem]:
        """Sample *n* items, always including all anchor rules.

        If there are more anchors than *n*, all anchors are returned
        (exceeding the requested count) to ensure none are lost.
        If no buffer is configured, returns only anchor items.
        """
        anchor_items = self._anchor_to_memory_items()

        if self._buffer is None:
            return anchor_items

        remaining = max(0, n - len(anchor_items))
        episodic_items = self._buffer.sample(remaining) if remaining > 0 else []

        combined = anchor_items + episodic_items
        logger.debug(
            f"SymbolicAnchorReplay: {len(anchor_items)} anchors + "
            f"{len(episodic_items)} episodic = {len(combined)} total"
        )
        return combined

    def replay_to_engine(self, engine: Any) -> int:
        """Replay all anchor rules into a SymbolicEngine.

        Calls ``engine.add_rule(rule)`` for each anchored rule.
        Rules that already exist in the engine (immutable) are
        silently skipped.

        Returns the number of rules successfully replayed.
        """
        count = 0
        for rule in self._anchor.all_rules():
            try:
                engine.add_rule(rule)
                count += 1
            except Exception:
                # Rule may already exist as immutable — skip silently
                pass
        logger.debug(f"Replayed {count} anchor rules to engine")
        return count

    def _anchor_to_memory_items(self) -> List[MemoryItem]:
        """Convert all anchor rules to MemoryItem format."""
        rules: List[SymbolicRule] = self._anchor.all_rules()
        return [
            MemoryItem(
                data=rule,
                task_id=self._anchor_task_id,
                label=rule.id,
                metadata={
                    "type": "symbolic_anchor",
                    "weight": rule.weight,
                    "immutable": True,
                },
            )
            for rule in rules
        ]

    @property
    def buffer_size(self) -> int:
        return self._buffer.size + len(self._anchor.all_rules())
