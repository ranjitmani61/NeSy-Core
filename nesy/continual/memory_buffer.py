"""
nesy/continual/memory_buffer.py
================================
Episodic memory replay buffer for continual learning.

Stores representative examples from past tasks.
During new task training, replaying these examples prevents
the neural backbone from forgetting previous task patterns.

Uses reservoir sampling for bounded, statistically representative storage.

Mathematical basis (Reservoir Sampling):
    Maintain buffer of size k from stream of n items.
    For each new item i (i > k):
        Accept with probability k/i
        If accepted, replace a random existing buffer item.

    Result: Each item has equal probability k/n of being in buffer.
    Unbiased representation of entire data stream.
"""

from __future__ import annotations

import random
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """A single item stored in the episodic buffer."""

    data: Any
    task_id: str
    label: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemoryBuffer:
    """Reservoir-sampled episodic memory buffer.

    Maintains a fixed-size, statistically representative sample
    of all examples seen during training — across all tasks.
    """

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._buffer: List[MemoryItem] = []
        self._total_seen: int = 0

    def add(self, item: MemoryItem) -> None:
        """Add item using reservoir sampling."""
        self._total_seen += 1

        if len(self._buffer) < self.max_size:
            self._buffer.append(item)
        else:
            # Reservoir sampling: accept with probability max_size / total_seen
            j = random.randint(0, self._total_seen - 1)
            if j < self.max_size:
                self._buffer[j] = item

    def add_batch(self, items: List[MemoryItem]) -> None:
        for item in items:
            self.add(item)

    def sample(self, n: int) -> List[MemoryItem]:
        """Sample n items from buffer (without replacement)."""
        n = min(n, len(self._buffer))
        return random.sample(self._buffer, n)

    def get_by_task(self, task_id: str) -> List[MemoryItem]:
        return [item for item in self._buffer if item.task_id == task_id]

    def clear_task(self, task_id: str) -> int:
        before = len(self._buffer)
        self._buffer = [i for i in self._buffer if i.task_id != task_id]
        return before - len(self._buffer)

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def total_seen(self) -> int:
        return self._total_seen

    @property
    def task_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for item in self._buffer:
            dist[item.task_id] = dist.get(item.task_id, 0) + 1
        return dist
