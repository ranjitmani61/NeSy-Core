"""
nesy/evaluation/metrics.py
===========================
Metrics for evaluating NeSy-Core outputs.

Standard ML metrics (precision, recall, F1) are extended with:
    - NSI null set accuracy: how well the null set predicts actual missing info
    - Confidence calibration: ECE and Brier score
    - Symbolic coverage: proportion of ground truth derivable symbolically
    - Self-doubt accuracy: was the system right to doubt?

Mathematical definitions:

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × P × R / (P + R)

Brier Score = (1/N) Σᵢ (pᵢ - yᵢ)²
    pᵢ = predicted confidence, yᵢ = actual correctness (0 or 1)
    Lower is better. 0 = perfect. 0.25 = random baseline.

ECE = Σₘ (|Bₘ|/N) × |acc(Bₘ) - conf(Bₘ)|
    Bₘ = confidence bins. Lower is better. 0 = perfectly calibrated.

Null Set Precision = |correct_nulls ∩ actual_missing| / |correct_nulls|
    Fraction of flagged absences that were genuinely missing

Null Set Recall = |correct_nulls ∩ actual_missing| / |actual_missing|
    Fraction of genuinely missing info that was caught
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class SymbolicMetrics:
    """Metrics for symbolic reasoning correctness."""
    true_positives:  int = 0    # correct derived facts
    false_positives: int = 0    # wrongly derived facts
    false_negatives: int = 0    # facts not derived but should be

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def exact_match(self) -> bool:
        return self.false_positives == 0 and self.false_negatives == 0

    def update(
        self,
        derived: Set,
        ground_truth: Set,
    ) -> None:
        """Update metrics from one prediction."""
        self.true_positives  += len(derived & ground_truth)
        self.false_positives += len(derived - ground_truth)
        self.false_negatives += len(ground_truth - derived)


@dataclass
class NSIMetrics:
    """Metrics for null set prediction quality."""
    # null_flagged: concepts we flagged as absent but important
    # actual_missing: concepts actually needed but not provided
    flagged_correctly:  int = 0
    flagged_wrongly:    int = 0
    missed_missing:     int = 0

    @property
    def null_precision(self) -> float:
        denom = self.flagged_correctly + self.flagged_wrongly
        return self.flagged_correctly / denom if denom > 0 else 0.0

    @property
    def null_recall(self) -> float:
        denom = self.flagged_correctly + self.missed_missing
        return self.flagged_correctly / denom if denom > 0 else 0.0

    @property
    def null_f1(self) -> float:
        p, r = self.null_precision, self.null_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def update(
        self,
        flagged_absent: Set[str],
        actually_missing: Set[str],
    ) -> None:
        self.flagged_correctly += len(flagged_absent & actually_missing)
        self.flagged_wrongly   += len(flagged_absent - actually_missing)
        self.missed_missing    += len(actually_missing - flagged_absent)


@dataclass
class ConfidenceMetrics:
    """Calibration metrics for confidence scores."""
    n_bins:          int   = 10
    _pairs:          List[Tuple[float, bool]] = field(default_factory=list)

    def add(self, predicted_confidence: float, actual_correct: bool) -> None:
        self._pairs.append((predicted_confidence, actual_correct))

    @property
    def brier_score(self) -> float:
        """Brier Score = (1/N) Σ (pᵢ - yᵢ)²"""
        if not self._pairs:
            return 0.0
        return sum((p - int(y)) ** 2 for p, y in self._pairs) / len(self._pairs)

    @property
    def ece(self) -> float:
        """Expected Calibration Error."""
        if not self._pairs:
            return 0.0
        bins = [[] for _ in range(self.n_bins)]
        for conf, correct in self._pairs:
            idx = min(int(conf * self.n_bins), self.n_bins - 1)
            bins[idx].append((conf, correct))

        n = len(self._pairs)
        ece = 0.0
        for b in bins:
            if not b:
                continue
            avg_conf = sum(x[0] for x in b) / len(b)
            avg_acc  = sum(1 for x in b if x[1]) / len(b)
            ece += (len(b) / n) * abs(avg_acc - avg_conf)
        return ece

    @property
    def accuracy(self) -> float:
        if not self._pairs:
            return 0.0
        return sum(1 for _, y in self._pairs if y) / len(self._pairs)

    @property
    def mean_confidence(self) -> float:
        if not self._pairs:
            return 0.0
        return sum(p for p, _ in self._pairs) / len(self._pairs)

    @property
    def overconfidence(self) -> float:
        """Positive = model is overconfident. Negative = underconfident."""
        return self.mean_confidence - self.accuracy

    @property
    def n_samples(self) -> int:
        return len(self._pairs)


@dataclass
class SelfDoubtMetrics:
    """Evaluate the self-doubt mechanism quality.
    
    Good self-doubt: when system says FLAGGED/REJECTED, it's actually wrong.
    Bad self-doubt: when system says FLAGGED for correct outputs (too conservative).
    """
    true_doubts:   int = 0   # Doubted and was actually wrong
    false_doubts:  int = 0   # Doubted but was actually correct (over-cautious)
    true_accepts:  int = 0   # Accepted and was actually correct
    false_accepts: int = 0   # Accepted but was actually wrong (missed failure)

    @property
    def doubt_precision(self) -> float:
        """When system doubts, how often is it right to doubt?"""
        denom = self.true_doubts + self.false_doubts
        return self.true_doubts / denom if denom > 0 else 0.0

    @property
    def doubt_recall(self) -> float:
        """What fraction of actual failures did the system catch?"""
        denom = self.true_doubts + self.false_accepts
        return self.true_doubts / denom if denom > 0 else 0.0

    @property
    def false_negative_rate(self) -> float:
        """Fraction of actual failures the system MISSED (dangerous!)"""
        denom = self.true_doubts + self.false_accepts
        return self.false_accepts / denom if denom > 0 else 0.0


@dataclass
class NeSyEvalReport:
    """Complete evaluation report for a NeSy model."""
    symbolic:    SymbolicMetrics   = field(default_factory=SymbolicMetrics)
    nsi:         NSIMetrics        = field(default_factory=NSIMetrics)
    confidence:  ConfidenceMetrics = field(default_factory=ConfidenceMetrics)
    self_doubt:  SelfDoubtMetrics  = field(default_factory=SelfDoubtMetrics)
    n_evaluated: int               = 0

    def summary(self) -> str:
        return (
            f"=== NeSy Evaluation Report (n={self.n_evaluated}) ===\n"
            f"Symbolic:   P={self.symbolic.precision:.3f} R={self.symbolic.recall:.3f} F1={self.symbolic.f1:.3f}\n"
            f"NSI Null:   P={self.nsi.null_precision:.3f} R={self.nsi.null_recall:.3f} F1={self.nsi.null_f1:.3f}\n"
            f"Confidence: Brier={self.confidence.brier_score:.4f} ECE={self.confidence.ece:.4f}\n"
            f"Self-Doubt: precision={self.self_doubt.doubt_precision:.3f} "
            f"fnr={self.self_doubt.false_negative_rate:.3f}\n"
        )
