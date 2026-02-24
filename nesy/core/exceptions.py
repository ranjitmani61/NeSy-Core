"""
nesy/core/exceptions.py
=======================
Custom exception hierarchy for NeSy-Core.

All exceptions carry structured context so callers can
programmatically handle different failure modes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from nesy.core.types import UnsatCore


class NeSyError(Exception):
    """Base exception for all NeSy-Core errors."""

    def __init__(self, message: str, context: Optional[dict] = None):
        super().__init__(message)
        self.context = context or {}


class SymbolicConflict(NeSyError):
    """Raised when symbolic reasoning detects a logical contradiction.

    This is a hard failure: the reasoning chain cannot proceed
    because the input violates a hard constraint (rule.weight >= 0.95).

    Optionally carries an ``UnsatCore`` with the minimal conflict set,
    human-readable explanation, and suggested repair actions.
    """

    def __init__(
        self,
        message: str,
        conflicting_rules: List[str],
        context: dict = None,
        unsat_core: Optional["UnsatCore"] = None,
    ):
        super().__init__(message, context)
        self.conflicting_rules = conflicting_rules
        self.unsat_core = unsat_core


class GroundingFailure(NeSyError):
    """Raised when a neural embedding cannot be grounded to any predicate
    with sufficient confidence (grounding_confidence < threshold)."""

    def __init__(self, message: str, embedding_shape: tuple, threshold: float):
        super().__init__(message)
        self.embedding_shape = embedding_shape
        self.threshold = threshold


class CriticalNullViolation(NeSyError):
    """Raised when Type3 (Critical) null items are detected and
    strict_mode is enabled. The output cannot be trusted."""

    def __init__(self, message: str, critical_items: list):
        super().__init__(message)
        self.critical_items = critical_items


class ContinualLearningConflict(NeSyError):
    """Raised when new learning would overwrite an immutable symbolic anchor."""

    def __init__(self, message: str, anchor_id: str, attempted_override: str):
        super().__init__(message)
        self.anchor_id = anchor_id
        self.attempted_override = attempted_override


class OntologyLoadError(NeSyError):
    """Raised when an ontology file cannot be parsed or is structurally invalid."""

    pass


class CalibrationError(NeSyError):
    """Raised when confidence calibration data is insufficient or corrupted."""

    pass
