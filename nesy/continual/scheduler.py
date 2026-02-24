"""
nesy/continual/scheduler.py
============================
Consolidation scheduler — decides WHEN to consolidate knowledge.

Triggering consolidation too early wastes compute.
Triggering too late risks forgetting.

Strategy: consolidate when any of these conditions are met:
    1. N new samples seen (count-based)
    2. Performance on validation set drops below threshold (quality-based)
    3. Task boundary explicitly signalled by caller (explicit)
"""
from __future__ import annotations
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ConsolidationScheduler:
    """Decides when to trigger EWC consolidation.
    
    Usage:
        scheduler = ConsolidationScheduler(samples_trigger=100)
        for sample in data_stream:
            scheduler.on_sample(sample)
            if scheduler.should_consolidate():
                learner.consolidate(...)
                scheduler.reset()
    """

    def __init__(
        self,
        samples_trigger:      int   = 100,
        quality_threshold:    float = 0.70,
        min_samples_before:   int   = 20,
    ):
        self.samples_trigger    = samples_trigger
        self.quality_threshold  = quality_threshold
        self.min_samples_before = min_samples_before

        self._samples_since_last: int = 0
        self._last_quality: Optional[float] = None
        self._force_flag: bool = False

    def on_sample(self) -> None:
        """Call for each new training sample."""
        self._samples_since_last += 1

    def on_quality_update(self, quality: float) -> None:
        """Call when validation quality is measured."""
        self._last_quality = quality

    def signal_task_boundary(self) -> None:
        """Explicitly signal that a new task is starting."""
        self._force_flag = True

    def should_consolidate(self) -> bool:
        """Returns True if consolidation should happen now."""
        if self._samples_since_last < self.min_samples_before:
            return False

        if self._force_flag:
            return True

        if self._samples_since_last >= self.samples_trigger:
            return True

        if (self._last_quality is not None and
                self._last_quality < self.quality_threshold):
            logger.info(
                f"Quality {self._last_quality:.3f} < {self.quality_threshold}. "
                "Triggering consolidation."
            )
            return True

        return False

    def reset(self) -> None:
        """Reset after consolidation."""
        self._samples_since_last = 0
        self._force_flag = False
        self._last_quality = None
