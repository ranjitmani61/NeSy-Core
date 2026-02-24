#!/usr/bin/env python3
"""
scripts/evaluate_benchmarks.py
================================
Run NeSy-Core evaluation benchmarks on labelled test cases.

Usage:
    python scripts/evaluate_benchmarks.py --domain medical
    python scripts/evaluate_benchmarks.py --domain general --verbose
"""
import argparse
import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nesy.api.nesy_model import NeSyModel
from nesy.core.types import ConceptEdge, Predicate, SymbolicRule
from nesy.evaluation.evaluator import EvalCase, NeSyEvaluator

logger = logging.getLogger(__name__)


def build_medical_model() -> NeSyModel:
    """Build a medical domain model with sample rules and edges."""
    model = NeSyModel(domain="medical")

    model.add_rule(SymbolicRule(
        id="fever_infection",
        antecedents=[Predicate("HasSymptom", ("?p", "fever"))],
        consequents=[Predicate("PossiblyHas", ("?p", "infection"))],
        weight=0.80,
        description="Fever suggests infection",
    ))
    model.add_rule(SymbolicRule(
        id="infection_blood",
        antecedents=[Predicate("PossiblyHas", ("?p", "infection"))],
        consequents=[Predicate("RequiresTest", ("?p", "blood_test"))],
        weight=0.90,
        description="Infection requires blood test",
    ))
    model.add_rule(SymbolicRule(
        id="cough_xray",
        antecedents=[Predicate("HasSymptom", ("?p", "cough"))],
        consequents=[Predicate("RequiresTest", ("?p", "chest_xray"))],
        weight=0.75,
        description="Cough may require chest X-ray",
    ))

    model.add_concept_edge(ConceptEdge("fever", "blood_test", 0.90, 1.0, 1.0))
    model.add_concept_edge(ConceptEdge("fever", "temperature", 0.95, 1.0, 1.0))
    model.add_concept_edge(ConceptEdge("cough", "chest_xray", 0.85, 1.0, 1.0))

    return model


def build_general_model() -> NeSyModel:
    """Build a general domain model with sample rules."""
    model = NeSyModel(domain="general")

    model.add_rule(SymbolicRule(
        id="human_mortal",
        antecedents=[Predicate("IsHuman", ("?x",))],
        consequents=[Predicate("IsMortal", ("?x",))],
        weight=1.0,
        description="All humans are mortal",
    ))
    model.add_rule(SymbolicRule(
        id="philosopher_human",
        antecedents=[Predicate("IsPhilosopher", ("?x",))],
        consequents=[Predicate("IsHuman", ("?x",))],
        weight=1.0,
        description="All philosophers are human",
    ))

    return model


def get_medical_cases() -> list:
    """Build labelled evaluation cases for the medical domain."""
    return [
        EvalCase(
            input_facts={Predicate("HasSymptom", ("p1", "fever"))},
            expected_derivations={
                Predicate("PossiblyHas", ("p1", "infection")),
                Predicate("RequiresTest", ("p1", "blood_test")),
            },
            actually_missing={"blood_test", "temperature"},
            expected_correct=True,
            context_type="medical",
            case_id="fever_basic",
        ),
        EvalCase(
            input_facts={Predicate("HasSymptom", ("p2", "cough"))},
            expected_derivations={
                Predicate("RequiresTest", ("p2", "chest_xray")),
            },
            actually_missing={"chest_xray"},
            expected_correct=True,
            context_type="medical",
            case_id="cough_basic",
        ),
        EvalCase(
            input_facts={
                Predicate("HasSymptom", ("p3", "fever")),
                Predicate("HasSymptom", ("p3", "cough")),
            },
            expected_derivations={
                Predicate("PossiblyHas", ("p3", "infection")),
                Predicate("RequiresTest", ("p3", "blood_test")),
                Predicate("RequiresTest", ("p3", "chest_xray")),
            },
            actually_missing={"blood_test", "chest_xray"},
            expected_correct=True,
            context_type="medical",
            case_id="fever_and_cough",
        ),
    ]


def get_general_cases() -> list:
    """Build labelled evaluation cases for the general domain."""
    return [
        EvalCase(
            input_facts={Predicate("IsPhilosopher", ("socrates",))},
            expected_derivations={
                Predicate("IsHuman", ("socrates",)),
                Predicate("IsMortal", ("socrates",)),
            },
            expected_correct=True,
            case_id="socrates",
        ),
        EvalCase(
            input_facts={Predicate("IsHuman", ("alice",))},
            expected_derivations={Predicate("IsMortal", ("alice",))},
            expected_correct=True,
            case_id="mortal_alice",
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="NeSy-Core Evaluation Benchmarks")
    parser.add_argument("--domain", choices=["medical", "general"], default="medical")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    print(f"Running NeSy-Core benchmarks for domain: {args.domain}")
    print("=" * 60)

    if args.domain == "medical":
        model = build_medical_model()
        cases = get_medical_cases()
    else:
        model = build_general_model()
        cases = get_general_cases()

    evaluator = NeSyEvaluator(model)
    report = evaluator.evaluate(cases)

    print(report.summary())
    print("=" * 60)
    print(f"Evaluated {report.n_evaluated} cases")
    print(f"Symbolic F1:  {report.symbolic.f1:.3f}")
    print(f"Null Set F1:  {report.nsi.null_f1:.3f}")
    print(f"Brier Score:  {report.confidence.brier_score:.3f}")
    print(f"Self-Doubt Precision: {report.self_doubt.doubt_precision:.3f}")


if __name__ == "__main__":
    main()
