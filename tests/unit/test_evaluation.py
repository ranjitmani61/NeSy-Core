"""tests/unit/test_evaluator.py — Evaluation metrics"""

import pytest
from nesy.evaluation.metrics import (
    ConfidenceMetrics,
    NSIMetrics,
    NeSyEvalReport,
    SelfDoubtMetrics,
    SymbolicMetrics,
)
from nesy.metacognition.calibration import ConfidenceCalibrator


class TestSymbolicMetrics:
    def test_perfect_precision_recall(self):
        m = SymbolicMetrics()
        m.update({"A", "B"}, {"A", "B"})
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)

    def test_all_wrong(self):
        m = SymbolicMetrics()
        m.update({"C", "D"}, {"A", "B"})
        assert m.precision == pytest.approx(0.0)
        assert m.recall == pytest.approx(0.0)

    def test_partial_match(self):
        m = SymbolicMetrics()
        m.update({"A", "B", "C"}, {"A", "B"})  # 1 false positive
        assert m.precision == pytest.approx(2.0 / 3.0)
        assert m.recall == pytest.approx(1.0)


class TestNSIMetrics:
    def test_perfect_null_detection(self):
        m = NSIMetrics()
        m.update({"missing_a", "missing_b"}, {"missing_a", "missing_b"})
        assert m.null_precision == pytest.approx(1.0)
        assert m.null_recall == pytest.approx(1.0)

    def test_null_recall_partial(self):
        m = NSIMetrics()
        m.update({"missing_a"}, {"missing_a", "missing_b"})
        assert m.null_recall == pytest.approx(0.5)


class TestConfidenceMetrics:
    def test_brier_perfect(self):
        m = ConfidenceMetrics()
        for _ in range(100):
            m.add(1.0, True)
        assert m.brier_score == pytest.approx(0.0, abs=1e-6)

    def test_brier_random(self):
        m = ConfidenceMetrics()
        for _ in range(100):
            m.add(0.5, True)
        assert m.brier_score == pytest.approx(0.25, abs=0.01)

    def test_overconfidence_positive(self):
        m = ConfidenceMetrics()
        m.add(0.9, False)  # confident but wrong
        assert m.overconfidence > 0

    def test_n_samples(self):
        m = ConfidenceMetrics()
        for i in range(10):
            m.add(0.8, True)
        assert m.n_samples == 10


class TestSelfDoubtMetrics:
    def test_perfect_doubt(self):
        m = SelfDoubtMetrics()
        m.true_doubts = 10
        m.false_doubts = 0
        assert m.doubt_precision == pytest.approx(1.0)

    def test_zero_false_negative_rate(self):
        m = SelfDoubtMetrics()
        m.true_doubts = 5
        m.false_accepts = 0
        assert m.false_negative_rate == pytest.approx(0.0)


class TestNeSyEvalReport:
    def test_summary_contains_metrics(self):
        report = NeSyEvalReport(n_evaluated=42)
        summary = report.summary()
        assert "Symbolic" in summary
        assert "NSI" in summary
        assert "Confidence" in summary
        assert "42" in summary


# tests/unit/test_calibration.py — Confidence calibrator


class TestConfidenceCalibrator:
    def test_unfitted_passthrough(self):
        cal = ConfidenceCalibrator()
        result = cal.calibrate(0.75)
        assert result == pytest.approx(0.75)

    def test_fit_returns_self(self):
        cal = ConfidenceCalibrator()
        data = [(0.7, True)] * 50 + [(0.3, False)] * 50
        result = cal.fit(data)
        assert result is cal

    def test_fitted_changes_output(self):
        cal = ConfidenceCalibrator()
        data = [(0.9, True)] * 80 + [(0.9, False)] * 20  # overconfident: 0.9 but 80% accurate
        cal.fit(data)
        calibrated = cal.calibrate(0.9)
        # Calibrated should be closer to actual accuracy 0.8
        assert calibrated != pytest.approx(0.9)

    def test_ece_computed_after_fit(self):
        cal = ConfidenceCalibrator()
        data = [(float(i) / 100, i % 2 == 0) for i in range(100)]
        cal.fit(data)
        assert cal.ece is not None
        assert 0.0 <= cal.ece <= 1.0
