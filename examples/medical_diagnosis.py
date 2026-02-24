"""
examples/medical_diagnosis.py
==============================
Working end-to-end demo of NeSy-Core on a medical diagnosis use case.

This demonstrates all five layers working together:
  - Symbolic rules (domain knowledge)
  - NSI null set (what symptoms are conspicuously absent)
  - MetaCognition (confidence + self-doubt)
  - ContinualLearner (symbolic anchors)
  - Full reasoning trace

Run: python examples/medical_diagnosis.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nesy.api.nesy_model import NeSyModel
from nesy.core.types import (
    ConceptEdge,
    Predicate,
    SymbolicRule,
)


# ─────────────────────────────────────────────
#  1. DEFINE DOMAIN RULES
# ─────────────────────────────────────────────

MEDICAL_RULES = [

    # Hard rules (weight=1.0) — never violated
    SymbolicRule(
        id="contraindication_penicillin",
        antecedents=[
            Predicate("HasAllergy", ("?p", "penicillin")),
            Predicate("Prescribed",  ("?p", "penicillin")),
        ],
        consequents=[Predicate("ContraindicationViolated", ("?p", "penicillin"))],
        weight=1.0,
        immutable=True,
        description="Prescribing penicillin to a penicillin-allergic patient is a hard violation.",
    ),

    # Soft rules — probabilistic clinical reasoning
    SymbolicRule(
        id="fever_and_high_wbc_implies_infection",
        antecedents=[
            Predicate("HasSymptom", ("?p", "fever")),
            Predicate("HasLabResult", ("?p", "elevated_wbc")),
        ],
        consequents=[Predicate("PossiblyHas", ("?p", "bacterial_infection"))],
        weight=0.85,
        domain="medical",
        description="Fever + elevated WBC is a classic sign of bacterial infection.",
    ),

    SymbolicRule(
        id="chest_pain_and_sweating_implies_cardiac",
        antecedents=[
            Predicate("HasSymptom", ("?p", "chest_pain")),
            Predicate("HasSymptom", ("?p", "sweating")),
        ],
        consequents=[Predicate("RequiresUrgentTest", ("?p", "ecg"))],
        weight=0.92,
        domain="medical",
        description="Chest pain + sweating requires urgent ECG.",
    ),

    SymbolicRule(
        id="infection_implies_culture_needed",
        antecedents=[Predicate("PossiblyHas", ("?p", "bacterial_infection"))],
        consequents=[Predicate("RequiresTest", ("?p", "blood_culture"))],
        weight=0.80,
        domain="medical",
        description="Possible bacterial infection requires blood culture.",
    ),
]


# ─────────────────────────────────────────────
#  2. DEFINE CONCEPT GRAPH EDGES (for NSI)
# ─────────────────────────────────────────────

MEDICAL_EDGES = [

    # fever → expected associated concepts
    ConceptEdge("fever", "blood_test",
                cooccurrence_prob=0.90, causal_strength=1.0,  temporal_stability=1.0),
    ConceptEdge("fever", "temperature_reading",
                cooccurrence_prob=0.95, causal_strength=1.0,  temporal_stability=1.0),
    ConceptEdge("fever", "duration",
                cooccurrence_prob=0.70, causal_strength=0.5,  temporal_stability=1.0),
    ConceptEdge("fever", "recent_travel",
                cooccurrence_prob=0.40, causal_strength=0.5,  temporal_stability=0.5),

    # chest_pain → expected associated concepts
    ConceptEdge("chest_pain", "ecg",
                cooccurrence_prob=0.88, causal_strength=1.0,  temporal_stability=1.0),
    ConceptEdge("chest_pain", "troponin",
                cooccurrence_prob=0.75, causal_strength=1.0,  temporal_stability=1.0),
    ConceptEdge("chest_pain", "blood_pressure",
                cooccurrence_prob=0.80, causal_strength=1.0,  temporal_stability=1.0),
    ConceptEdge("chest_pain", "sweating",
                cooccurrence_prob=0.55, causal_strength=0.5,  temporal_stability=1.0),

    # bacterial_infection → what should follow
    ConceptEdge("bacterial_infection", "antibiotic",
                cooccurrence_prob=0.85, causal_strength=1.0,  temporal_stability=1.0),
    ConceptEdge("bacterial_infection", "blood_culture",
                cooccurrence_prob=0.72, causal_strength=1.0,  temporal_stability=1.0),
    ConceptEdge("bacterial_infection", "source_identification",
                cooccurrence_prob=0.65, causal_strength=1.0,  temporal_stability=1.0),
]


# ─────────────────────────────────────────────
#  3. BUILD MODEL
# ─────────────────────────────────────────────

def build_medical_model() -> NeSyModel:
    model = (
        NeSyModel(domain="medical", doubt_threshold=0.55, strict_mode=False)
        .add_rules(MEDICAL_RULES)
        .add_concept_edges(MEDICAL_EDGES)
        .register_critical_concept("blood_pressure", "vital_signs")
        .register_critical_concept("temperature_reading", "vital_signs")
        .register_critical_concept("troponin", "diagnostic_test")
        .register_critical_concept("ecg", "diagnostic_test")
    )
    return model


# ─────────────────────────────────────────────
#  4. RUN SCENARIOS
# ─────────────────────────────────────────────

def scenario_fever_with_no_lab_work():
    """
    Scenario: Patient has fever, but no lab results ordered.
    Expected: NSI flags missing blood_test, temperature_reading (critical).
    """
    print("\n" + "═" * 60)
    print("SCENARIO 1: Fever with no lab work")
    print("═" * 60)

    model = build_medical_model()
    facts = {
        Predicate("HasSymptom", ("patient_001", "fever")),
    }

    output = model.reason(
        facts=facts,
        context_type="medical",
        neural_confidence=0.80,
        raw_input="Patient presents with fever.",
    )

    print(model.explain(output))
    return output


def scenario_chest_pain_complete():
    """
    Scenario: Patient has chest pain AND sweating AND ECG ordered.
    Expected: High confidence, urgent test derived, no critical nulls.
    """
    print("\n" + "═" * 60)
    print("SCENARIO 2: Chest pain — complete workup")
    print("═" * 60)

    model = build_medical_model()
    facts = {
        Predicate("HasSymptom",   ("patient_002", "chest_pain")),
        Predicate("HasSymptom",   ("patient_002", "sweating")),
        Predicate("HasLabResult", ("patient_002", "ecg_ordered")),
        Predicate("HasLabResult", ("patient_002", "troponin_ordered")),
    }

    output = model.reason(
        facts=facts,
        context_type="medical",
        neural_confidence=0.92,
        raw_input="Patient with chest pain and sweating. ECG and troponin ordered.",
    )

    print(model.explain(output))
    return output


def scenario_contraindication_violation():
    """
    Scenario: Patient is allergic to penicillin but it is prescribed.
    Expected: REJECTED — hard constraint violated.
    """
    print("\n" + "═" * 60)
    print("SCENARIO 3: Contraindication violation (hard constraint)")
    print("═" * 60)

    model = build_medical_model()
    facts = {
        Predicate("HasAllergy",  ("patient_003", "penicillin")),
        Predicate("Prescribed",  ("patient_003", "penicillin")),
    }

    output = model.reason(
        facts=facts,
        context_type="medical",
        neural_confidence=0.95,
        raw_input="Patient with penicillin allergy prescribed penicillin.",
    )

    print(model.explain(output))
    print(f"\nTrustworthy: {output.is_trustworthy()}")
    return output


def scenario_continual_learning():
    """
    Scenario: Learn a new rule. Anchor an immutable fact.
    Verify old knowledge is intact.
    """
    print("\n" + "═" * 60)
    print("SCENARIO 4: Continual learning")
    print("═" * 60)

    model = build_medical_model()

    # Learn new soft rule
    model.learn(SymbolicRule(
        id="high_crp_confirms_infection",
        antecedents=[Predicate("HasLabResult", ("?p", "elevated_crp"))],
        consequents=[Predicate("PossiblyHas", ("?p", "bacterial_infection"))],
        weight=0.75,
        description="Elevated CRP supports infection diagnosis.",
    ))

    # Anchor an immutable safety rule
    model.learn(
        SymbolicRule(
            id="never_prescribe_to_dead_patient",
            antecedents=[
                Predicate("PatientStatus", ("?p", "deceased")),
                Predicate("Prescribed",    ("?p", "?drug")),
            ],
            consequents=[Predicate("CriticalError", ("?p", "?drug"))],
            weight=1.0,
            description="Cannot prescribe to a deceased patient.",
        ),
        make_anchor=True,
    )

    print(f"Total rules:   {model.rule_count}")
    print(f"Anchored rules: {model.anchored_rules}")

    # Verify new rule works
    facts = {Predicate("HasLabResult", ("patient_004", "elevated_crp"))}
    output = model.reason(facts=facts, context_type="medical")
    print(f"\nNew rule result: {output.answer}")
    print(f"Status: {output.status.value}")
    return output


# ─────────────────────────────────────────────
#  5. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("NeSy-Core — Medical Diagnosis Demo")
    print("NSI + Symbolic Reasoning + MetaCognition")
    print()

    o1 = scenario_fever_with_no_lab_work()
    o2 = scenario_chest_pain_complete()
    o3 = scenario_contraindication_violation()
    o4 = scenario_continual_learning()

    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    print(f"Scenario 1 — Fever (incomplete):      {o1.status.value:10s} | conf={o1.confidence.minimum:.2f}")
    print(f"Scenario 2 — Chest pain (complete):   {o2.status.value:10s} | conf={o2.confidence.minimum:.2f}")
    print(f"Scenario 3 — Contraindication:        {o3.status.value:10s} | conf={o3.confidence.minimum:.2f}")
    print(f"Scenario 4 — Continual learning:      {o4.status.value:10s} | conf={o4.confidence.minimum:.2f}")
