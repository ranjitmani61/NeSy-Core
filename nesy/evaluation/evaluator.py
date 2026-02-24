"""
nesy/evaluation/evaluator.py
=============================
NeSyEvaluator — evaluates model performance on a labelled test set.

Usage:
    evaluator = NeSyEvaluator(model)

    report = evaluator.evaluate(test_cases)
    print(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from nesy.api.nesy_model import NeSyModel
from nesy.core.types import NSIOutput, OutputStatus, Predicate
from nesy.evaluation.metrics import (
    NeSyEvalReport,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    """A single labelled evaluation case.

    Attributes:
        input_facts:          The known facts given to the model
        expected_derivations: The predicates the model SHOULD derive
        actually_missing:     Concepts that are genuinely absent but important
        expected_correct:     Is the expected output actually correct?
        context_type:         Domain context for reasoning
        neural_confidence:    Simulated neural confidence
    """

    input_facts: Set[Predicate]
    expected_derivations: Set[Predicate] = field(default_factory=set)
    actually_missing: Set[str] = field(default_factory=set)
    expected_correct: bool = True
    context_type: str = "general"
    neural_confidence: float = 0.90
    case_id: str = ""


class NeSyEvaluator:
    """Evaluate a NeSyModel on a labelled test set.

    Computes:
        - Symbolic precision/recall/F1 (correct derivations)
        - NSI null set precision/recall/F1 (correct absence detection)
        - Confidence calibration (Brier score, ECE)
        - Self-doubt quality (when system is right to doubt)
    """

    def __init__(self, model: NeSyModel):
        self.model = model

    def evaluate(self, cases: List[EvalCase]) -> NeSyEvalReport:
        """Run evaluation on all cases and return report."""
        report = NeSyEvalReport()

        for case in cases:
            output = self.model.reason(
                facts=case.input_facts,
                context_type=case.context_type,
                neural_confidence=case.neural_confidence,
            )
            self._update_report(report, case, output)
            report.n_evaluated += 1

        logger.info(f"Evaluation complete: {report.n_evaluated} cases")
        return report

    def evaluate_one(self, case: EvalCase) -> Tuple[NSIOutput, Dict]:
        """Evaluate a single case and return output + per-case metrics."""
        output = self.model.reason(
            facts=case.input_facts,
            context_type=case.context_type,
            neural_confidence=case.neural_confidence,
        )
        report = NeSyEvalReport()
        self._update_report(report, case, output)
        return output, {
            "symbolic_f1": report.symbolic.f1,
            "null_f1": report.nsi.null_f1,
            "brier": report.confidence.brier_score,
        }

    def _update_report(
        self,
        report: NeSyEvalReport,
        case: EvalCase,
        output: NSIOutput,
    ) -> None:
        """Update all metric trackers for one case."""
        # 1. Symbolic metrics
        # Parse derived predicates from answer string
        derived_predicates = self._parse_derived(output)
        report.symbolic.update(derived_predicates, case.expected_derivations)

        # 2. NSI null set metrics
        flagged_absent = {item.concept for item in output.null_set.critical_items}
        flagged_absent |= {item.concept for item in output.null_set.meaningful_items}
        report.nsi.update(flagged_absent, case.actually_missing)

        # 3. Confidence calibration
        report.confidence.add(output.confidence.minimum, case.expected_correct)

        # 4. Self-doubt metrics
        doubted = output.status in (OutputStatus.FLAGGED, OutputStatus.REJECTED)
        if doubted and not case.expected_correct:
            report.self_doubt.true_doubts += 1
        elif doubted and case.expected_correct:
            report.self_doubt.false_doubts += 1
        elif not doubted and case.expected_correct:
            report.self_doubt.true_accepts += 1
        else:
            report.self_doubt.false_accepts += 1

    def _parse_derived(self, output: NSIOutput) -> Set[Predicate]:
        """Extract derived predicates from reasoning trace."""
        derived = set()
        for step in output.reasoning_trace.steps:
            for pred in step.predicates:
                derived.add(pred)
        return derived
