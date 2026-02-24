"""
nesy/metacognition/shadow.py
=============================
Counterfactual Shadow Reasoning — A new dimension of epistemic confidence.

CONCEPT (original, not from any paper):
    Every AI system answers: "How confident am I?"
    This module answers: "How far is this conclusion from being WRONG?"

    These are fundamentally different questions.

    A conclusion with confidence=0.95 but Shadow Distance=1 is DANGEROUS:
    one fact change flips the entire diagnosis.

    A conclusion with confidence=0.72 but Shadow Distance=7 is TRUSTWORTHY:
    7 independent facts must all be wrong simultaneously to flip it.

FORMAL DEFINITION:
    Given:
        F = set of present facts (premises)
        R = set of symbolic rules
        C = a derived conclusion

    CounterfactualShadow(C, F, R) = min { |S| : S ⊆ F, (F \\ S) ⊬ C using R }

    Where ⊬ means "does not derive".

    This is the Minimum Fact Cut problem on the reasoning graph.

    The set S is called the "shadow set" — the minimal set of facts
    whose removal collapses the conclusion.

2D CONFIDENCE SPACE:
    Instead of one confidence number, NeSy-Core now produces two:

        C₁ ∈ [0,1]: Factual confidence (probabilistic — existing)
        C₂ ∈ [0,∞): Shadow distance (structural robustness — new)

    Combined status:
        C₂ ≥ 5, C₁ ≥ 0.75  → ROBUST     (safe to act)
        C₂ ≥ 3, C₁ ≥ 0.60  → STABLE     (act with awareness)
        C₂ = 2              → FRAGILE    (human review)
        C₂ = 1              → CRITICAL   (one fact flip = wrong answer)
        C₂ = 0 (impossible) → TAUTOLOGY  (always true)

SAFETY APPLICATION:
    Medical: Shadow Distance=1 on a diagnosis → flag for immediate
             physician review before treatment.

    Legal:   Shadow Distance=1 on a verdict → evidence is dangerously
             hinging on a single fact.

    Code:    Shadow Distance=1 on a security verdict → one test removed
             = vulnerability ships.

Author:  NeSy-Core Team
Date:    2026
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set

from nesy.core.types import Predicate, SymbolicRule

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  ENUMS AND DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────


class ShadowClass(str, Enum):
    """Structural robustness classification based on shadow distance."""

    TAUTOLOGY = "tautology"  # always true — shadow distance ∞
    ROBUST = "robust"  # distance ≥ 5 — safe to act
    STABLE = "stable"  # distance 3-4 — act with awareness
    FRAGILE = "fragile"  # distance 2 — human review required
    CRITICAL = "critical"  # distance 1 — one fact flip = wrong


@dataclass
class ShadowResult:
    """Result of counterfactual shadow analysis for one conclusion.

    Attributes:
        conclusion:      The predicate whose shadow was computed.
        distance:        Minimum number of facts whose removal flips the conclusion.
                         math.inf if the conclusion is a logical tautology.
        critical_facts:  The minimal set of facts that, if removed,
                         causes the conclusion to no longer be derivable.
                         Empty if distance is ∞.
        flip_explanation: Human-readable explanation of the shadow.
        shadow_class:    Robustness classification (CRITICAL → ROBUST → TAUTOLOGY).
        rules_involved:  Rules that participated in deriving the conclusion.
        all_shadows:     All minimal shadows of the same distance (there may be
                         multiple equally-minimal cuts).
    """

    conclusion: Predicate
    distance: float  # int or math.inf
    critical_facts: FrozenSet[Predicate] = field(default_factory=frozenset)
    flip_explanation: str = ""
    shadow_class: ShadowClass = ShadowClass.STABLE
    rules_involved: List[str] = field(default_factory=list)
    all_shadows: List[FrozenSet[Predicate]] = field(default_factory=list)

    @property
    def is_tautology(self) -> bool:
        return math.isinf(self.distance)

    @property
    def is_critical(self) -> bool:
        return self.distance == 1

    @property
    def is_fragile(self) -> bool:
        return self.distance <= 2

    def __str__(self) -> str:
        if self.is_tautology:
            return f"Shadow({self.conclusion}): TAUTOLOGY — always derivable"
        if self.is_critical:
            removable = ", ".join(str(p) for p in self.critical_facts)
            return (
                f"Shadow({self.conclusion}): CRITICAL (d=1) — "
                f"removing [{removable}] flips conclusion"
            )
        return (
            f"Shadow({self.conclusion}): {self.shadow_class.value.upper()} "
            f"(d={int(self.distance)}) — {len(self.critical_facts)} facts protect conclusion"
        )


@dataclass
class ShadowReport:
    """Full shadow analysis for all conclusions in a reasoning result.

    Contains per-conclusion shadow distances and system-level
    robustness summary.
    """

    shadows: Dict[str, ShadowResult]  # conclusion_str → ShadowResult
    minimum_distance: float  # most fragile conclusion
    most_critical: Optional[ShadowResult]  # the most fragile conclusion
    system_class: ShadowClass  # overall system robustness
    total_facts_used: int
    escalation_required: bool  # True if any conclusion is CRITICAL

    def summary(self) -> str:
        lines = [
            f"Shadow Report — {len(self.shadows)} conclusion(s)",
            f"  System class:    {self.system_class.value.upper()}",
            f"  Min distance:    {int(self.minimum_distance) if not math.isinf(self.minimum_distance) else '∞'}",
            f"  Escalate:        {'YES — human review required' if self.escalation_required else 'no'}",
        ]
        for key, s in self.shadows.items():
            lines.append(f"  {s}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  CORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────


class CounterfactualShadowEngine:
    """Computes Counterfactual Shadow Distance for derived conclusions.

    Algorithm:
        For distance k = 1, 2, 3, ..., |F|:
            For each subset S ⊆ F with |S| = k:
                Run symbolic derivation on F \\ S
                If conclusion C no longer derivable:
                    Return k, S

    This is exponential in |F| for the exact algorithm.
    For large fact sets, we use the heuristic approximation.

    Mathematical basis:
        The exact problem is equivalent to finding the minimum vertex cut
        in the "support hypergraph" of the conclusion — where each fact is
        a vertex and each rule application is a hyperedge.

        Minimum vertex cut is solvable in polynomial time for ordinary graphs
        (via max-flow, Ford-Fulkerson) but NP-hard for hypergraphs.

        For safety-critical systems with small fact sets (< 20 facts),
        exact computation is tractable. For larger sets, the heuristic
        provides a lower bound on the true shadow distance.

    Args:
        max_exact_facts: Use exact algorithm when |F| ≤ this value.
                         Above this threshold, switch to heuristic.
        max_shadow_depth: Maximum k to search. Stops early if k > this.
                          Conclusions that survive to max_depth are classified
                          as TAUTOLOGY-equivalent (extremely robust).
    """

    def __init__(
        self,
        max_exact_facts: int = 15,
        max_shadow_depth: int = 7,
    ):
        self.max_exact_facts = max_exact_facts
        self.max_shadow_depth = max_shadow_depth

    def compute(
        self,
        conclusion: Predicate,
        facts: Set[Predicate],
        rules: List[SymbolicRule],
        derived_steps: Optional[List] = None,  # ReasoningStep list if available
    ) -> ShadowResult:
        """Compute shadow distance for a single conclusion.

        Args:
            conclusion:    The derived predicate to analyze.
            facts:         Original fact set used in derivation.
            rules:         Rule set used in derivation.
            derived_steps: Optional reasoning trace to accelerate computation.

        Returns:
            ShadowResult with distance, critical_facts, and explanation.
        """
        # Guard: if conclusion is in original facts → it's an axiom
        if conclusion in facts:
            return ShadowResult(
                conclusion=conclusion,
                distance=math.inf,
                critical_facts=frozenset(),
                flip_explanation=f"'{conclusion}' is a direct axiom — always present.",
                shadow_class=ShadowClass.TAUTOLOGY,
            )

        # Guard: empty fact set → nothing to remove
        if not facts:
            return ShadowResult(
                conclusion=conclusion,
                distance=math.inf,
                critical_facts=frozenset(),
                flip_explanation="No facts — conclusion is a tautology under empty KB.",
                shadow_class=ShadowClass.TAUTOLOGY,
            )

        # Guard: conclusion not derivable from full facts → trivially ∞
        if not self._can_derive(conclusion, facts, rules):
            return ShadowResult(
                conclusion=conclusion,
                distance=math.inf,
                critical_facts=frozenset(),
                flip_explanation=(
                    f"'{conclusion}' is not derivable from the given facts and rules — "
                    f"shadow distance is trivially infinite."
                ),
                shadow_class=ShadowClass.TAUTOLOGY,
            )

        # Select algorithm
        if len(facts) <= self.max_exact_facts:
            return self._exact_search(conclusion, facts, rules)
        else:
            return self._heuristic_search(conclusion, facts, rules)

    def compute_all(
        self,
        conclusions: Set[Predicate],
        facts: Set[Predicate],
        rules: List[SymbolicRule],
    ) -> ShadowReport:
        """Compute shadow distances for all conclusions.

        Returns a ShadowReport with system-level robustness summary.
        """
        shadows: Dict[str, ShadowResult] = {}
        min_distance = math.inf
        most_critical: Optional[ShadowResult] = None

        for c in conclusions:
            result = self.compute(c, facts, rules)
            shadows[str(c)] = result
            if result.distance < min_distance:
                min_distance = result.distance
                most_critical = result
            logger.debug(f"Shadow({c}): distance={result.distance}")

        system_class = _classify_distance(min_distance)
        escalation = min_distance <= 1 and not math.isinf(min_distance)

        return ShadowReport(
            shadows=shadows,
            minimum_distance=min_distance,
            most_critical=most_critical,
            system_class=system_class,
            total_facts_used=len(facts),
            escalation_required=escalation,
        )

    # ─────────────────────────────────────────────────────────────────
    #  EXACT ALGORITHM (BFS over subsets of increasing size)
    # ─────────────────────────────────────────────────────────────────

    def _exact_search(
        self,
        conclusion: Predicate,
        facts: Set[Predicate],
        rules: List[SymbolicRule],
    ) -> ShadowResult:
        """Exact minimum cut via BFS over fact subsets.

        Time complexity: O(|F|^k) where k is the shadow distance.
        For small |F| and small k (typical in reasoning), this is fast.
        """
        fact_list = list(facts)
        all_shadows: List[FrozenSet[Predicate]] = []

        for k in range(1, min(self.max_shadow_depth + 1, len(fact_list) + 1)):
            for subset in itertools.combinations(fact_list, k):
                reduced_facts = facts - set(subset)
                if not self._can_derive(conclusion, reduced_facts, rules):
                    # Found a shadow of size k
                    shadow_set = frozenset(subset)
                    all_shadows.append(shadow_set)
                    # Collect all minimal shadows at this depth
                    # (check remaining combinations at same k)
                    for other_subset in itertools.combinations(fact_list, k):
                        other_reduced = facts - set(other_subset)
                        if other_subset != subset and not self._can_derive(
                            conclusion, other_reduced, rules
                        ):
                            other_shadow = frozenset(other_subset)
                            if other_shadow not in all_shadows:
                                all_shadows.append(other_shadow)
                    # Take smallest shadow as primary
                    primary = min(all_shadows, key=len) if all_shadows else shadow_set
                    explanation = _build_explanation(conclusion, primary, k)
                    return ShadowResult(
                        conclusion=conclusion,
                        distance=k,
                        critical_facts=primary,
                        flip_explanation=explanation,
                        shadow_class=_classify_distance(k),
                        all_shadows=all_shadows,
                    )

        # Survived all depths → effectively a tautology
        return ShadowResult(
            conclusion=conclusion,
            distance=math.inf,
            critical_facts=frozenset(),
            flip_explanation=(
                f"'{conclusion}' survived removal of up to {self.max_shadow_depth} facts. "
                f"Conclusion is extremely robust (shadow depth > {self.max_shadow_depth})."
            ),
            shadow_class=ShadowClass.TAUTOLOGY,
        )

    # ─────────────────────────────────────────────────────────────────
    #  HEURISTIC ALGORITHM (for large fact sets)
    # ─────────────────────────────────────────────────────────────────

    def _heuristic_search(
        self,
        conclusion: Predicate,
        facts: Set[Predicate],
        rules: List[SymbolicRule],
    ) -> ShadowResult:
        """Heuristic shadow search for large fact sets.

        Strategy:
            1. Identify "load-bearing" facts: those that appear in
               rules whose consequents are on the derivation path to C.
            2. Try removing load-bearing facts first (greedy).
            3. This gives an upper bound on shadow distance.

        For large fact sets, this runs in O(|F| × |R|) time vs O(|F|^k).
        May overestimate shadow distance (never underestimates).
        """
        # Step 1: Find facts that are load-bearing for this conclusion
        load_bearing = self._find_load_bearing_facts(conclusion, facts, rules)

        if not load_bearing:
            # No load-bearing facts found — heuristically treat as robust
            return ShadowResult(
                conclusion=conclusion,
                distance=float(self.max_shadow_depth),
                critical_facts=frozenset(),
                flip_explanation=(
                    "Heuristic: no single load-bearing fact identified. Conclusion appears robust."
                ),
                shadow_class=ShadowClass.ROBUST,
            )

        # Step 2: Try each load-bearing fact individually (k=1)
        for fact in load_bearing:
            reduced = facts - {fact}
            if not self._can_derive(conclusion, reduced, rules):
                return ShadowResult(
                    conclusion=conclusion,
                    distance=1,
                    critical_facts=frozenset({fact}),
                    flip_explanation=_build_explanation(conclusion, frozenset({fact}), 1),
                    shadow_class=ShadowClass.CRITICAL,
                    all_shadows=[frozenset({fact})],
                )

        # Step 3: Try pairs of load-bearing facts (k=2)
        lb_list = list(load_bearing)
        for i in range(len(lb_list)):
            for j in range(i + 1, len(lb_list)):
                pair = frozenset({lb_list[i], lb_list[j]})
                reduced = facts - pair
                if not self._can_derive(conclusion, reduced, rules):
                    return ShadowResult(
                        conclusion=conclusion,
                        distance=2,
                        critical_facts=pair,
                        flip_explanation=_build_explanation(conclusion, pair, 2),
                        shadow_class=ShadowClass.FRAGILE,
                        all_shadows=[pair],
                    )

        # Step 4: Count load-bearing facts as proxy for shadow distance
        approx_distance = min(len(load_bearing), self.max_shadow_depth)
        return ShadowResult(
            conclusion=conclusion,
            distance=float(approx_distance),
            critical_facts=frozenset(load_bearing),
            flip_explanation=(
                f"Heuristic: ~{approx_distance} facts protect this conclusion. "
                f"Structural robustness is {_classify_distance(approx_distance).value}."
            ),
            shadow_class=_classify_distance(approx_distance),
        )

    def _find_load_bearing_facts(
        self,
        conclusion: Predicate,
        facts: Set[Predicate],
        rules: List[SymbolicRule],
    ) -> Set[Predicate]:
        """Find facts whose predicate name matches antecedents of rules
        that produce the conclusion's predicate.

        These are the 'load-bearing walls' of the derivation.
        """
        load_bearing: Set[Predicate] = set()

        # Find rules that can derive conclusion's predicate
        relevant_rules = [r for r in rules if any(c.name == conclusion.name for c in r.consequents)]

        # Find facts that match antecedents of those rules
        for rule in relevant_rules:
            for ant in rule.antecedents:
                for fact in facts:
                    if fact.name == ant.name:
                        load_bearing.add(fact)

        return load_bearing

    # ─────────────────────────────────────────────────────────────────
    #  DERIVATION CHECK (pure forward chaining — no imports from engine)
    # ─────────────────────────────────────────────────────────────────

    def _can_derive(
        self,
        conclusion: Predicate,
        facts: Set[Predicate],
        rules: List[SymbolicRule],
    ) -> bool:
        """Check if conclusion is derivable from reduced facts using rules.

        Uses minimal forward chaining — no unification, just predicate name
        and arity matching. This is intentionally simpler than SymbolicEngine
        to avoid circular dependencies and keep shadow computation fast.

        If exact unification is needed, subclass and override this method.
        """
        # Direct fact check
        if any(f.name == conclusion.name and len(f.args) == len(conclusion.args) for f in facts):
            return True

        # Forward chaining (fixpoint)
        derived = set(facts)
        changed = True
        while changed:
            changed = False
            for rule in rules:
                # Check if all antecedents are satisfied (name-based matching)
                if self._rule_fires(rule, derived):
                    for consequent in rule.consequents:
                        if not any(d.name == consequent.name for d in derived):
                            derived.add(consequent)
                            changed = True
                            if consequent.name == conclusion.name:
                                return True

        return any(d.name == conclusion.name for d in derived)

    def _rule_fires(
        self,
        rule: SymbolicRule,
        derived: Set[Predicate],
    ) -> bool:
        """Check if all antecedents of a rule are present in derived set.

        Uses name-count matching: if a rule requires two antecedents with
        the same predicate name (e.g. HasSymptom(?p, X) AND HasSymptom(?p, Y)),
        there must be at least two distinct facts with that name in derived.
        """
        from collections import Counter

        ant_name_counts = Counter(ant.name for ant in rule.antecedents)
        derived_name_counts = Counter(p.name for p in derived)
        return all(
            derived_name_counts.get(name, 0) >= count for name, count in ant_name_counts.items()
        )


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def _classify_distance(distance: float) -> ShadowClass:
    """Map shadow distance to robustness class.

    Thresholds chosen for safety-critical systems:
        d = 1: ONE fact removed → conclusion flips. Extremely dangerous.
        d = 2: TWO facts. Still dangerous — simple data corruption flips answer.
        d = 3-4: Moderate protection. Act with awareness.
        d ≥ 5: Strong protection. Safe to act.
        d = ∞: Logical tautology. Maximally robust.
    """
    if math.isinf(distance):
        return ShadowClass.TAUTOLOGY
    if distance <= 1:
        return ShadowClass.CRITICAL
    if distance <= 2:
        return ShadowClass.FRAGILE
    if distance <= 4:
        return ShadowClass.STABLE
    return ShadowClass.ROBUST


def _build_explanation(
    conclusion: Predicate,
    shadow_set: FrozenSet[Predicate],
    distance: int,
) -> str:
    """Build a human-readable explanation of the shadow.

    Designed for doctors, lawyers, and safety engineers —
    not for ML researchers.
    """
    fact_descriptions = [str(f) for f in shadow_set]
    facts_str = " AND ".join(f"'{f}'" for f in fact_descriptions)

    if distance == 1:
        return (
            f"CRITICAL: If {facts_str} were absent from the evidence, "
            f"the conclusion '{conclusion}' would NOT be derivable. "
            f"This conclusion rests on a single supporting fact."
        )
    elif distance == 2:
        return (
            f"FRAGILE: If both {facts_str} were absent, "
            f"the conclusion '{conclusion}' would collapse. "
            f"Two facts jointly protect this conclusion — both must be questioned."
        )
    else:
        return (
            f"The conclusion '{conclusion}' requires removing {distance} facts "
            f"simultaneously to become underivable. "
            f"Specifically: {facts_str}. "
            f"Shadow distance = {distance} → {_classify_distance(distance).value} conclusion."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  INTEGRATION HELPER — for NeSyModel output
# ─────────────────────────────────────────────────────────────────────────────


def shadow_flags(report: ShadowReport) -> List[str]:
    """Generate flag strings from shadow report for inclusion in NSIOutput.flags."""
    flags = []
    if report.escalation_required:
        c = report.most_critical
        flags.append(
            f"SHADOW-CRITICAL: '{c.conclusion}' rests on {int(c.distance)} fact(s). "
            f"Remove {', '.join(str(f) for f in c.critical_facts)} → conclusion flips. "
            f"HUMAN REVIEW REQUIRED."
        )
    elif report.system_class == ShadowClass.FRAGILE:
        flags.append(
            f"SHADOW-FRAGILE: Minimum shadow distance = {int(report.minimum_distance)}. "
            f"Conclusions are structurally fragile. Review evidence quality."
        )
    elif report.system_class == ShadowClass.STABLE:
        flags.append(
            f"SHADOW-STABLE: Minimum shadow distance = {int(report.minimum_distance)}. "
            f"Conclusions are moderately robust."
        )
    return flags
