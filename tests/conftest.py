"""
tests/conftest.py
==================
Shared pytest fixtures for all NeSy-Core tests.
"""

import pytest
from nesy.api.nesy_model import NeSyModel
from nesy.core.types import ConceptEdge, Predicate, SymbolicRule
from nesy.nsi.concept_graph import ConceptGraphEngine
from nesy.symbolic.engine import SymbolicEngine


# ─── RULES ────────────────────────────────────────────────────────


@pytest.fixture
def fever_infection_rule():
    return SymbolicRule(
        id="fever_infection",
        antecedents=[
            Predicate("HasSymptom", ("?p", "fever")),
            Predicate("HasLabResult", ("?p", "elevated_wbc")),
        ],
        consequents=[Predicate("PossiblyHas", ("?p", "bacterial_infection"))],
        weight=0.85,
        domain="medical",
        description="Fever + WBC → infection",
    )


@pytest.fixture
def hard_contraindication_rule():
    return SymbolicRule(
        id="penicillin_allergy",
        antecedents=[
            Predicate("HasAllergy", ("?p", "penicillin")),
            Predicate("Prescribed", ("?p", "penicillin")),
        ],
        consequents=[Predicate("ContraindicationViolated", ("?p", "penicillin"))],
        weight=1.0,
        immutable=True,
        description="Hard contraindication: penicillin allergy",
    )


@pytest.fixture
def basic_rules(fever_infection_rule, hard_contraindication_rule):
    return [fever_infection_rule, hard_contraindication_rule]


# ─── CONCEPT GRAPH ────────────────────────────────────────────────


@pytest.fixture
def fever_edges():
    return [
        ConceptEdge(
            "fever",
            "blood_test",
            cooccurrence_prob=0.90,
            causal_strength=1.0,
            temporal_stability=1.0,
        ),
        ConceptEdge(
            "fever",
            "temperature_reading",
            cooccurrence_prob=0.95,
            causal_strength=1.0,
            temporal_stability=1.0,
        ),
    ]


@pytest.fixture
def concept_graph(fever_edges):
    cge = ConceptGraphEngine(domain="medical")
    cge.add_edges(fever_edges)
    cge.register_concept_class("temperature_reading", "vital_signs")
    cge.register_concept_class("blood_test", "diagnostic_test")
    return cge


# ─── MODELS ───────────────────────────────────────────────────────


@pytest.fixture
def medical_model(basic_rules, fever_edges):
    model = NeSyModel(domain="medical", doubt_threshold=0.55)
    model.add_rules(basic_rules)
    model.add_concept_edges(fever_edges)
    model.register_critical_concept("temperature_reading", "vital_signs")
    model.register_critical_concept("blood_test", "diagnostic_test")
    return model


@pytest.fixture
def symbolic_engine(basic_rules):
    engine = SymbolicEngine(domain="medical")
    engine.load_rules(basic_rules)
    return engine


# ─── PREDICATES ───────────────────────────────────────────────────


@pytest.fixture
def fever_facts():
    return {
        Predicate("HasSymptom", ("patient_test", "fever")),
        Predicate("HasLabResult", ("patient_test", "elevated_wbc")),
    }


@pytest.fixture
def contraindication_facts():
    return {
        Predicate("HasAllergy", ("patient_test", "penicillin")),
        Predicate("Prescribed", ("patient_test", "penicillin")),
    }
