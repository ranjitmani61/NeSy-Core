"""
nesy/neural/nsil.py
===================
Neural–Symbolic Integrity Link (NSIL) — Revolution #7.

NSIL makes the neural→symbolic grounding **auditable and self-checking**
by computing a deterministic integrity signal that measures disagreement
between neural grounding evidence and symbolic constraint expectations.

Mathematical basis:
    For each predicate schema s in (R ∪ D):

        e(s) = proto_sim(s)  if grounded evidence exists, else 0.0
        m(s) = 1.0           if s ∈ D (derived), else 0.0
        need(s) = 1.0        if s ∈ R (required by rules), else 0.5

        ISR(s)  = need(s) × |m(s) − e(s)|

    IntegrityScore = clamp01(1.0 − (0.7 × avg(ISR) + 0.3 × max(ISR)))

Behaviour rule:
    If integrity is weak →
        reduce knowledge_boundary deterministically,
        potentially trigger NSIL_LOW_INTEGRITY flag.
    Never fabricate additional facts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Set

from nesy.core.types import GroundedSymbol, NullSet, Predicate, SymbolicRule


# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────


@dataclass
class IntegrityItem:
    """Per-predicate-schema integrity residual.

    Attributes:
        schema:      The predicate schema string (e.g. ``HasSymptom``).
        evidence:    Proto-sim from grounding layer, clamped to [0, 1].
        membership:  1.0 if in derived facts, else 0.0.
        need:        1.0 if required by rules, else 0.5 (derived-only).
        residual:    ISR(s) = need × |membership − evidence|.
    """

    schema: str
    evidence: float
    membership: float
    need: float
    residual: float


@dataclass
class IntegrityReport:
    """Full NSIL integrity report.

    Attributes:
        items:           Per-schema residuals.
        integrity_score: Overall score ∈ [0, 1]. 1.0 = perfect alignment.
        flags:           Integrity-related flags.
        suggestions:     Optional CFG-compatible repair suggestions.
        is_neutral:      True when NSIL is bypassed (symbolic-only mode).
    """

    items: List[IntegrityItem] = field(default_factory=list)
    integrity_score: float = 1.0
    flags: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    is_neutral: bool = False


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────


def _clamp01(value: float) -> float:
    """Clamp *value* to [0, 1].  Maps NaN/Inf to 0.0."""
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _predicate_schema(predicate: Predicate) -> str:
    """Extract the schema name from a predicate (strip variables/args)."""
    return predicate.name


def _predicates_required_by_rules(
    activated_rules: List[SymbolicRule],
) -> Set[str]:
    """Collect predicate schemas from antecedents + consequents of rules."""
    schemas: Set[str] = set()
    for rule in activated_rules:
        for p in rule.antecedents:
            schemas.add(_predicate_schema(p))
        for p in rule.consequents:
            schemas.add(_predicate_schema(p))
    return schemas


def _predicate_schemas_present(predicates) -> Set[str]:
    """Collect predicate schemas from an iterable of predicates."""
    return {_predicate_schema(p) for p in predicates}


def _best_proto_sim(
    schema: str,
    grounded: List[GroundedSymbol],
) -> float:
    """Best grounding_confidence for *schema* among grounded symbols.

    Returns 0.0 if the schema is not present in grounded evidence.
    Clamps values to [0, 1].
    """
    best = 0.0
    for gs in grounded:
        if _predicate_schema(gs.predicate) == schema:
            clamped = _clamp01(gs.grounding_confidence)
            if clamped > best:
                best = clamped
    return best


# ─────────────────────────────────────────────
#  MAIN COMPUTE FUNCTION
# ─────────────────────────────────────────────


def compute_integrity_report(
    grounded: List[GroundedSymbol],
    activated_rules: List[SymbolicRule],
    derived_facts: Set[Predicate],
    null_set: Optional[NullSet] = None,
    is_passthrough: bool = False,
    doubt_threshold: float = 0.6,
) -> IntegrityReport:
    """Compute the full NSIL integrity report.

    Parameters
    ----------
    grounded : list[GroundedSymbol]
        Evidence from the grounding layer.
    activated_rules : list[SymbolicRule]
        Rules that fired during reasoning.
    derived_facts : set[Predicate]
        Facts derived by forward chaining.
    null_set : NullSet, optional
        NSI null set (used for enrichment suggestions).
    is_passthrough : bool
        If ``True``, the backbone is a PassthroughBackbone or no neural
        system is in use — return a **neutral** report (no penalty).
    doubt_threshold : float
        Threshold below which ``NSIL_LOW_INTEGRITY`` flag is added.

    Returns
    -------
    IntegrityReport
        Structured integrity report with per-schema residuals, overall
        score, flags, and optional repair suggestions.
    """

    # ── Symbolic-only / passthrough → neutral ─────────────────
    if is_passthrough:
        return IntegrityReport(
            integrity_score=1.0,
            is_neutral=True,
        )

    # ── Step 1: required predicate set ────────────────────────
    required: Set[str] = _predicates_required_by_rules(activated_rules)
    _predicate_schemas_present(gs.predicate for gs in grounded)
    derived_schemas: Set[str] = _predicate_schemas_present(derived_facts)

    # Evaluate over R ∪ D
    all_schemas: Set[str] = required | derived_schemas

    # If no schemas to evaluate → neutral
    if not all_schemas:
        return IntegrityReport(
            integrity_score=1.0,
            is_neutral=True,
        )

    # ── Step 2: per-schema integrity residual ─────────────────
    items: List[IntegrityItem] = []
    for schema in sorted(all_schemas):
        e = _best_proto_sim(schema, grounded)
        m = 1.0 if schema in derived_schemas else 0.0
        need = 1.0 if schema in required else 0.5
        residual = need * abs(m - e)

        items.append(
            IntegrityItem(
                schema=schema,
                evidence=e,
                membership=m,
                need=need,
                residual=residual,
            )
        )

    # ── Step 3: overall integrity score ───────────────────────
    residuals = [item.residual for item in items]
    avg_r = sum(residuals) / len(residuals)
    max_r = max(residuals)

    integrity_score = _clamp01(1.0 - (0.7 * avg_r + 0.3 * max_r))

    # ── Step 4: flags + suggestions ───────────────────────────
    flags: List[str] = []
    suggestions: List[str] = []

    if integrity_score < doubt_threshold:
        flags.append("NSIL_LOW_INTEGRITY")

    # Items with very high residual (> 0.5) get individual flags
    for item in items:
        if item.residual > 0.5:
            if item.membership == 1.0 and item.evidence < 0.2:
                flags.append(
                    f"NSIL_WEAK_EVIDENCE: '{item.schema}' derived but "
                    f"grounding evidence is {item.evidence:.2f}"
                )
                suggestions.append(
                    f"Provide grounding evidence for '{item.schema}' or verify the derivation."
                )
            elif item.membership == 0.0 and item.evidence > 0.8:
                flags.append(
                    f"NSIL_UNUSED_EVIDENCE: '{item.schema}' has strong "
                    f"grounding ({item.evidence:.2f}) but was not derived."
                )

    # Enrich with null set critical items
    if null_set is not None:
        for ci in null_set.critical_items:
            suggestions.append(f"Critical null '{ci.concept}' may affect integrity.")

    return IntegrityReport(
        items=items,
        integrity_score=integrity_score,
        flags=flags,
        suggestions=suggestions,
        is_neutral=False,
    )
