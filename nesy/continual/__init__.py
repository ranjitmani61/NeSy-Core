"""nesy/continual — Continual learning without catastrophic forgetting."""

from nesy.continual.learner import ContinualLearner, SymbolicAnchor, TaskSnapshot
from nesy.continual.ewc import EWCRegularizer, EWCSnapshot
from nesy.continual.memory_buffer import EpisodicMemoryBuffer, MemoryItem
from nesy.continual.scheduler import ConsolidationScheduler
from nesy.continual.replay import (
    ReplayStrategy,
    RandomReplay,
    PrioritisedReplay,
    SymbolicAnchorReplay,
)

__all__ = [
    "ContinualLearner",
    "SymbolicAnchor",
    "TaskSnapshot",
    "EWCRegularizer",
    "EWCSnapshot",
    "EpisodicMemoryBuffer",
    "MemoryItem",
    "ConsolidationScheduler",
    "ReplayStrategy",
    "RandomReplay",
    "PrioritisedReplay",
    "SymbolicAnchorReplay",
]
