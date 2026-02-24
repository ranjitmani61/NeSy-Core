"""
nesy/metacognition/calibration.py
===================================
Confidence calibration — ensures predicted confidence scores
actually correlate with empirical accuracy.

Mathematical basis (Platt Scaling):
    P_calibrated(y=1|x) = σ(A × f(x) + B)
    Where:
        f(x) = raw confidence score from monitor
        A, B = learned calibration parameters
        σ    = sigmoid function

    Calibration is measured by Expected Calibration Error (ECE):
        ECE = Σₘ (|Bₘ|/n) × |acc(Bₘ) - conf(Bₘ)|

    Where Bₘ are confidence bins, acc = actual accuracy, conf = predicted confidence.
    ECE = 0 is perfectly calibrated.
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """Platt-scaling calibrator for NeSy-Core confidence scores.

    Usage:
        calibrator = ConfidenceCalibrator()
        # After collecting (predicted_confidence, was_correct) pairs:
        calibrator.fit(calibration_data)
        calibrated_score = calibrator.calibrate(raw_score)
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self._A: float = 1.0  # Platt scaling slope
        self._B: float = 0.0  # Platt scaling bias
        self._fitted: bool = False
        self._ece: Optional[float] = None

    def fit(self, data: List[Tuple[float, bool]]) -> "ConfidenceCalibrator":
        """Fit calibration parameters from (predicted_confidence, actual_correct) pairs.

        Minimum 100 samples recommended for reliable calibration.
        """
        if len(data) < 10:
            logger.warning(f"Calibration data too small ({len(data)}). Need ≥100 samples.")
            return self

        predictions = [d[0] for d in data]
        labels = [1.0 if d[1] else 0.0 for d in data]

        # Simple gradient descent for Platt scaling
        self._A, self._B = self._fit_platt(predictions, labels)
        self._fitted = True
        self._ece = self._compute_ece(data)
        logger.info(f"Calibration fitted. ECE={self._ece:.4f}, A={self._A:.3f}, B={self._B:.3f}")
        return self

    def calibrate(self, raw_confidence: float) -> float:
        """Apply calibration to a raw confidence score."""
        if not self._fitted:
            return raw_confidence
        logit = self._A * raw_confidence + self._B
        return self._sigmoid(logit)

    @property
    def ece(self) -> Optional[float]:
        """Expected Calibration Error. Lower is better. 0.0 = perfect."""
        return self._ece

    def _fit_platt(
        self,
        predictions: List[float],
        labels: List[float],
        lr: float = 0.01,
        steps: int = 500,
    ) -> Tuple[float, float]:
        """Gradient descent for Platt scaling parameters."""
        A, B = 1.0, 0.0
        n = len(predictions)

        for _ in range(steps):
            dA = dB = 0.0
            for p, y in zip(predictions, labels):
                prob = self._sigmoid(A * p + B)
                err = prob - y
                dA += err * p
                dB += err
            A -= lr * dA / n
            B -= lr * dB / n

        return A, B

    def _compute_ece(self, data: List[Tuple[float, bool]]) -> float:
        """ECE = Σₘ (|Bₘ|/n) × |acc(Bₘ) - conf(Bₘ)|"""
        bins = [[] for _ in range(self.n_bins)]
        for conf, correct in data:
            idx = min(int(conf * self.n_bins), self.n_bins - 1)
            bins[idx].append((conf, correct))

        ece = 0.0
        n = len(data)
        for b in bins:
            if not b:
                continue
            avg_conf = sum(x[0] for x in b) / len(b)
            avg_acc = sum(1 for x in b if x[1]) / len(b)
            ece += (len(b) / n) * abs(avg_acc - avg_conf)
        return ece

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)
