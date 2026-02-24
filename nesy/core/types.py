"""
nesy/core/types.py
==================
Foundation type system for NeSy-Core.
Every module imports from here. No circular dependencies.

Mathematical basis:
  - SymbolicRule encodes first-order logic clause: ∀x: P(x) → Q(x)
  - NullItem encodes absence weight: w(a→b) = P(b|a) × causal × temporal
  - NSIOutput encodes three independent epistemic confidence dimensions
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ─────────────────────────────────────────────
#  ENUMERATIONS
# ─────────────────────────────────────────────

class ConfidenceType(Enum):
    """Three orthogonal epistemic confidence dimensions.

    Factual:          Does the claim match known ground truth?
    Reasoning:        Is the logical derivation chain valid?
    KnowledgeBoundary: Is the query within the model's competence?
    """
    FACTUAL            = "factual"
    REASONING          = "reasoning"
    KNOWLEDGE_BOUNDARY = "knowledge_boundary"


class NullType(Enum):
    """Classification of absent concepts (NSI framework).

    Type1: Expected null  — absent because too obvious to state (normal)
    Type2: Meaningful null — absent but should be present (soft flag)
    Type3: Critical null   — absent but causally necessary (hard flag)
    """
    TYPE1_EXPECTED    = 1
    TYPE2_MEANINGFUL  = 2
    TYPE3_CRITICAL    = 3


class OutputStatus(Enum):
    OK        = "ok"
    FLAGGED   = "flagged"
    UNCERTAIN = "uncertain"
    REJECTED  = "rejected"


class LogicConnective(Enum):
    AND     = "∧"
    OR      = "∨"
    NOT     = "¬"
    IMPLIES = "→"
    IFF     = "↔"
    FORALL  = "∀"
    EXISTS  = "∃"


# ─────────────────────────────────────────────
#  SYMBOLIC TYPES
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class Predicate:
    """First-order logic predicate: name(arg1, arg2, ...)
    
    Frozen so predicates can be used in sets and as dict keys.
    Examples:
        Predicate("HasSymptom", ("patient_1", "fever"))
        Predicate("Implies", ("HasSymptom", "RequiresTest"))
    """
    name: str
    args: Tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(self.args)})"

    def __repr__(self) -> str:
        return f"Predicate({self!s})"


@dataclass
class SymbolicRule:
    """A first-order logic rule encoding a constraint or implication.

    Formal structure:
        weight × [ antecedents ] → [ consequents ]
        
    Where weight ∈ (0, 1] encodes the rule's strength:
        1.0 = hard logical constraint (violation raises SymbolicConflict)
        0.5 = soft constraint (violation adds to confidence penalty)
        0.1 = weak heuristic

    Example:
        If patient has fever AND elevated_wbc → infection_likely
        antecedents = [HasSymptom(fever), HasSymptom(elevated_wbc)]
        consequents = [Likely(infection)]
        weight = 0.9
    """
    id:           str
    antecedents:  List[Predicate]
    consequents:  List[Predicate]
    weight:       float = 1.0
    domain:       Optional[str] = None      # "medical", "legal", None = universal
    immutable:    bool  = False             # True = symbolic anchor, never overwritten
    description:  str   = ""

    def __post_init__(self):
        assert 0.0 < self.weight <= 1.0, "Rule weight must be in (0, 1]"
        assert len(self.antecedents) > 0,  "Rule must have at least one antecedent"
        assert len(self.consequents) > 0,  "Rule must have at least one consequent"

    def is_hard_constraint(self) -> bool:
        return self.weight >= 0.95


@dataclass
class LogicClause:
    """A resolved or unresolved logical clause produced during inference.
    
    During symbolic reasoning, rules are chained into clauses.
    satisfied = True  → clause resolved without contradiction
    satisfied = False → clause violated, carries penalty weight
    """
    predicates:  List[Predicate]
    connective:  LogicConnective
    satisfied:   bool
    weight:      float
    source_rule: Optional[str] = None   # rule.id that produced this clause


# ─────────────────────────────────────────────
#  NEURAL TYPES
# ─────────────────────────────────────────────

@dataclass
class GroundedSymbol:
    """A neural embedding grounded to a symbolic predicate.
    
    Grounding is the bridge between continuous neural space
    and discrete symbolic space.
    
    grounding_confidence: How well does this embedding
    represent the symbolic predicate? ∈ [0, 1]
    """
    predicate:            Predicate
    embedding:            List[float]          # raw neural vector
    grounding_confidence: float                # ∈ [0, 1]
    source_layer:         Optional[int] = None # which transformer layer produced this


# ─────────────────────────────────────────────
#  NSI (NEGATIVE SPACE INTELLIGENCE) TYPES
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class ConceptEdge:
    """Weighted directed edge in concept graph: source → target.
    
    Edge weight formula:
        W(a→b) = P(b|a) × causal_strength × temporal_stability
        
    Where:
        P(b|a)             ∈ [0,1] statistical co-occurrence probability
        causal_strength    ∈ {0.1, 0.5, 1.0} logical necessity
        temporal_stability ∈ {0.1, 0.5, 1.0} permanence of relationship
    """
    source:             str
    target:             str
    cooccurrence_prob:  float   # P(b|a)
    causal_strength:    float   # 1.0=necessary, 0.5=associated, 0.1=weak
    temporal_stability: float   # 1.0=permanent, 0.5=stable, 0.1=transient

    @property
    def weight(self) -> float:
        """W(a→b) = P(b|a) × causal × temporal"""
        return self.cooccurrence_prob * self.causal_strength * self.temporal_stability

    def __post_init__(self):
        assert 0.0 <= self.cooccurrence_prob  <= 1.0
        assert self.causal_strength    in (0.1, 0.5, 1.0)
        assert self.temporal_stability in (0.1, 0.5, 1.0)


@dataclass
class NullItem:
    """A single item in the Null Set N(X).
    
    Represents a concept that SHOULD be present in context X
    but is ABSENT. The null_type classifies the significance
    of this absence.
    
    anomaly_contribution = weight × criticality_multiplier(null_type)
    """
    concept:              str
    weight:               float         # edge weight from concept graph
    null_type:            NullType
    expected_because_of:  List[str]     # which present concepts triggered this expectation
    criticality:          float = 1.0   # domain-specific importance override

    @property
    def anomaly_contribution(self) -> float:
        multiplier = {
            NullType.TYPE1_EXPECTED:   0.0,
            NullType.TYPE2_MEANINGFUL: 1.0,
            NullType.TYPE3_CRITICAL:   3.0,
        }[self.null_type]
        return self.weight * self.criticality * multiplier


@dataclass
class PresentSet:
    """Set of concepts explicitly present or implied in input X."""
    concepts:   Set[str]
    context_type: str = "general"   # "medical", "legal", "code", "general"
    raw_input:  Optional[str] = None


@dataclass
class NullSet:
    """N(X) = E(X) − P(X): meaningful absences in context X."""
    items:        List[NullItem]
    present_set:  PresentSet

    @property
    def critical_items(self) -> List[NullItem]:
        return [i for i in self.items if i.null_type == NullType.TYPE3_CRITICAL]

    @property
    def meaningful_items(self) -> List[NullItem]:
        return [i for i in self.items if i.null_type == NullType.TYPE2_MEANINGFUL]

    @property
    def total_anomaly_score(self) -> float:
        """A(X) = Σ anomaly_contribution for all Type2 + Type3 items"""
        return sum(item.anomaly_contribution for item in self.items)


# ─────────────────────────────────────────────
#  METACOGNITION TYPES
# ─────────────────────────────────────────────

@dataclass
class UnsatCore:
    """Minimal set of constraints/rules that cause unsatisfiability.

    When the symbolic solver or engine detects a logical contradiction,
    this dataclass captures the *minimal* conflicting subset plus a
    human-readable explanation and suggested repair actions.

    Mathematical basis:
        An unsatisfiable core (unsat core) is a subset S ⊆ Φ of the
        formula set Φ such that S is unsatisfiable and no proper
        subset of S is unsatisfiable.  Z3 extracts this via
        ``solver.unsat_core()`` using tracked assertions.

    Attributes:
        conflicting_rule_ids:  Rule IDs in the minimal conflict set.
        constraint_ids:        Constraint indices from the SMT solver.
        explanation:           Human-readable contradiction narrative.
        suggested_additions:   Concepts/facts that could resolve the conflict.
        repair_actions:        Structured repair suggestions (concept → action).
        raw_labels:            Z3 tracked assertion labels (for debugging).
    """

    conflicting_rule_ids:  List[str]            = field(default_factory=list)
    constraint_ids:        List[int]            = field(default_factory=list)
    explanation:           str                  = ""
    suggested_additions:   List[str]            = field(default_factory=list)
    repair_actions:        List[Dict[str, str]] = field(default_factory=list)
    raw_labels:            List[str]            = field(default_factory=list)

    @property
    def core_size(self) -> int:
        """Number of conflicting rules/constraints in the minimal core."""
        return max(len(self.conflicting_rule_ids), len(self.constraint_ids))

    @property
    def has_repairs(self) -> bool:
        """Whether any repair suggestions are available."""
        return len(self.suggested_additions) > 0 or len(self.repair_actions) > 0

    def summary(self) -> str:
        """One-line summary for logging / CLI output."""
        n = self.core_size
        fixes = len(self.suggested_additions)
        return (
            f"UnsatCore({n} conflict{'s' if n != 1 else ''}, "
            f"{fixes} suggested fix{'es' if fixes != 1 else ''})"
        )


@dataclass
class ReasoningStep:
    """One step in a symbolic reasoning chain."""
    step_number:  int
    description:  str
    rule_applied: Optional[str]         # rule.id
    predicates:   List[Predicate]
    confidence:   float


@dataclass
class ReasoningTrace:
    """Full auditable trace of how a conclusion was reached.
    
    This is the 'glass box' component of NeSy-Core.
    Every output carries its full derivation.
    """
    steps:              List[ReasoningStep]
    rules_activated:    List[str]               # rule ids
    neural_confidence:  float                   # raw model confidence
    symbolic_confidence: float                  # logic chain confidence
    null_violations:    List[NullItem]          # from NSI layer
    logic_clauses:      List[LogicClause]

    @property
    def overall_confidence(self) -> float:
        """Harmonic mean of neural and symbolic confidence,
        penalised by null set anomaly score."""
        base = 2 * self.neural_confidence * self.symbolic_confidence / (
            self.neural_confidence + self.symbolic_confidence + 1e-8
        )
        # Each meaningful null reduces confidence
        null_penalty = sum(
            item.anomaly_contribution
            for item in self.null_violations
            if item.null_type != NullType.TYPE1_EXPECTED
        )
        return max(0.0, base / (1.0 + null_penalty))


@dataclass
class ConfidenceReport:
    """Three-dimensional epistemic confidence report.
    
    Factual:           Is the claim consistent with known facts?
    Reasoning:         Is the derivation chain logically valid?
    KnowledgeBoundary: Is this query within the model's competence?
    
    These three dimensions are INDEPENDENT.
    A model can be:
      - High factual + Low reasoning  (knows the fact, wrong logic path)
      - Low factual  + High reasoning (correct logic, wrong premise)
      - High both    + Low boundary   (correct now, but outside training)
    """
    factual:            float   # ∈ [0, 1]
    reasoning:          float   # ∈ [0, 1]
    knowledge_boundary: float   # ∈ [0, 1]
    explanation:        Dict[ConfidenceType, str] = field(default_factory=dict)

    def __post_init__(self):
        for score in (self.factual, self.reasoning, self.knowledge_boundary):
            assert 0.0 <= score <= 1.0, f"Confidence score {score} out of [0,1]"

    @property
    def minimum(self) -> float:
        """Overall confidence is gated by the weakest dimension."""
        return min(self.factual, self.reasoning, self.knowledge_boundary)

    @property
    def is_reliable(self) -> bool:
        return self.minimum >= 0.6


# ─────────────────────────────────────────────
#  FINAL OUTPUT TYPE
# ─────────────────────────────────────────────

@dataclass
class NSIOutput:
    """The complete output of the NeSy-Core reasoning pipeline.
    
    This is what nesy.reason() returns.
    Carries: answer + full confidence report + reasoning trace + flags.
    """
    answer:           str
    confidence:       ConfidenceReport
    reasoning_trace:  ReasoningTrace
    null_set:         NullSet
    status:           OutputStatus = OutputStatus.OK
    flags:            List[str]    = field(default_factory=list)
    request_id:       str          = field(default_factory=lambda: str(uuid.uuid4()))
    reasoning_fingerprint: Optional[str] = None

    def is_trustworthy(self) -> bool:
        return (
            self.status == OutputStatus.OK
            and self.confidence.is_reliable
            and len(self.null_set.critical_items) == 0
        )

    def summary(self) -> str:
        return (
            f"Answer: {self.answer}\n"
            f"Status: {self.status.value}\n"
            f"Confidence: factual={self.confidence.factual:.2f} | "
            f"reasoning={self.confidence.reasoning:.2f} | "
            f"boundary={self.confidence.knowledge_boundary:.2f}\n"
            f"Critical nulls: {len(self.null_set.critical_items)}\n"
            f"Reasoning steps: {len(self.reasoning_trace.steps)}"
        )
