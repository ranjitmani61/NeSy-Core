"""
nesy/__init__.py — Public API exports
"""

from nesy.api.nesy_model import (
    ContradictionReport,
    NeSyModel,
    NeSyModel as NeSy,
)
from nesy.core.exceptions import (
    ContinualLearningConflict,
    CriticalNullViolation,
    NeSyError,
    SymbolicConflict,
)
from nesy.core.types import (
    ConceptEdge,
    ConfidenceReport,
    ConfidenceType,
    NSIOutput,
    NullItem,
    NullSet,
    NullType,
    OutputStatus,
    Predicate,
    PresentSet,
    ReasoningTrace,
    SymbolicRule,
    UnsatCore,
)
from nesy.neural.nsil import IntegrityItem, IntegrityReport

__version__ = "0.1.1"

__all__ = [
    "NeSy",
    "NeSyModel",
    "ContradictionReport",
    "IntegrityItem",
    "IntegrityReport",
    "SymbolicRule",
    "Predicate",
    "ConceptEdge",
    "NSIOutput",
    "ConfidenceReport",
    "ConfidenceType",
    "NullItem",
    "NullSet",
    "NullType",
    "OutputStatus",
    "PresentSet",
    "ReasoningTrace",
    "UnsatCore",
    "NeSyError",
    "SymbolicConflict",
    "CriticalNullViolation",
    "ContinualLearningConflict",
]
