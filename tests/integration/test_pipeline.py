"""
tests/integration/test_pipeline.py
=====================================
End-to-end pipeline integration tests.
"""

import pytest
from nesy.api.nesy_model import NeSyModel
from nesy.core.types import ConceptEdge, NSIOutput, Predicate, SymbolicRule


@pytest.fixture
def full_medical_model():
    model = NeSyModel(domain="medical", doubt_threshold=0.50)
    model.add_rule(
        SymbolicRule(
            id="fever_infection",
            antecedents=[
                Predicate("HasSymptom", ("?p", "fever")),
                Predicate("HasLabResult", ("?p", "elevated_wbc")),
            ],
            consequents=[Predicate("PossiblyHas", ("?p", "bacterial_infection"))],
            weight=0.85,
        )
    )
    model.add_concept_edge(
        ConceptEdge(
            "fever",
            "blood_test",
            cooccurrence_prob=0.90,
            causal_strength=1.0,
            temporal_stability=1.0,
        )
    )
    model.register_critical_concept("blood_test", "diagnostic_test")
    return model


def test_full_pipeline_returns_nsioutput(full_medical_model):
    facts = {
        Predicate("HasSymptom", ("p1", "fever")),
        Predicate("HasLabResult", ("p1", "elevated_wbc")),
    }
    output = full_medical_model.reason(facts=facts, context_type="medical")
    assert isinstance(output, NSIOutput)


def test_pipeline_derives_infection(full_medical_model):
    facts = {
        Predicate("HasSymptom", ("p1", "fever")),
        Predicate("HasLabResult", ("p1", "elevated_wbc")),
    }
    output = full_medical_model.reason(facts=facts, context_type="medical")
    assert "bacterial_infection" in output.answer


def test_pipeline_rejected_on_contraindication():
    model = NeSyModel(domain="medical", strict_mode=False)
    model.add_rule(
        SymbolicRule(
            id="penicillin_contraindication",
            antecedents=[
                Predicate("HasAllergy", ("?p", "penicillin")),
                Predicate("Prescribed", ("?p", "penicillin")),
            ],
            consequents=[Predicate("ContraindicationViolated", ("?p", "penicillin"))],
            weight=1.0,
            immutable=True,
        )
    )
    facts = {
        Predicate("HasAllergy", ("p1", "penicillin")),
        Predicate("Prescribed", ("p1", "penicillin")),
    }
    output = model.reason(facts=facts, context_type="medical")
    # Contraindication should be derived — not rejected at symbolic level
    assert output.answer != ""


def test_continual_learning_adds_rules(full_medical_model):
    initial_count = full_medical_model.rule_count
    full_medical_model.learn(
        SymbolicRule(
            id="new_rule_xyz",
            antecedents=[Predicate("HasLabResult", ("?p", "elevated_crp"))],
            consequents=[Predicate("PossiblyHas", ("?p", "bacterial_infection"))],
            weight=0.75,
        )
    )
    assert full_medical_model.rule_count == initial_count + 1


def test_anchor_rule_count(full_medical_model):
    full_medical_model.learn(
        SymbolicRule(
            id="permanent_rule",
            antecedents=[Predicate("X", ("?a",))],
            consequents=[Predicate("Y", ("?a",))],
            weight=1.0,
        ),
        make_anchor=True,
    )
    assert full_medical_model.anchored_rules >= 1


def test_output_summary_contains_answer(full_medical_model):
    facts = {Predicate("HasSymptom", ("p1", "fever"))}
    output = full_medical_model.reason(facts=facts)
    summary = output.summary()
    assert "Answer:" in summary
    assert "Confidence:" in summary
