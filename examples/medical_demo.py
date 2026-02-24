import logging
from typing import Set

from nesy import NeSy
from nesy.core.types import Predicate, SymbolicRule, ConceptEdge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Medical demo:
      - Input: patient has fever
      - But missing vital sign: temperature_reading (Type3 critical null)
      - Expected behavior: OutputStatus becomes REJECTED (or at least FLAGGED)
    """

    m = NeSy(domain="medical")

    # --- Minimal symbolic rule pack ---
    m.add_rule(
        SymbolicRule(
            id="fever_infection_rule",
            antecedents=[Predicate("HasSymptom", ("?p", "fever"))],
            consequents=[Predicate("PossiblyHas", ("?p", "infection"))],
            weight=0.8,
            domain="medical",
            description="Fever may indicate infection (soft rule).",
        )
    )

    # --- Minimal NSI concept edges ---
    # fever -> temperature_reading (vital_sign, causally necessary)
    # We encode necessity via causal_strength=1.0, temporal_stability=1.0 and high P(b|a).
    m.add_concept_edge(
        ConceptEdge(
            source="fever",
            target="temperature_reading",
            cooccurrence_prob=0.95,
            causal_strength=1.0,
            temporal_stability=1.0,
        )
    )

    # fever -> blood_test (associated)
    m.add_concept_edge(
        ConceptEdge(
            source="fever",
            target="blood_test",
            cooccurrence_prob=0.80,
            causal_strength=0.5,
            temporal_stability=0.5,
        )
    )

    # --- Facts (explicit) ---
    facts: Set[Predicate] = {Predicate("HasSymptom", ("patient_1", "fever"))}

    # Case A: temperature missing => should be critical null
    out_a = m.reason(
        facts=facts,
        context_type="medical",
        raw_input="Patient has fever.",
    )
    print("\n=== CASE A: fever present, temperature_reading missing ===")
    print(out_a.summary())
    print("NullSet anomaly score:", out_a.null_set.total_anomaly_score)
    print("Critical nulls:", [i.concept for i in out_a.null_set.critical_items])
    print("Flags:", out_a.flags)

    # Case B: temperature provided => critical null should disappear
    facts_b: Set[Predicate] = set(facts)
    facts_b.add(Predicate("HasVital", ("patient_1", "temperature_reading")))
    out_b = m.reason(
        facts=facts_b,
        context_type="medical",
        raw_input="Patient has fever. Temperature recorded.",
    )
    print("\n=== CASE B: fever present + temperature_reading present ===")
    print(out_b.summary())
    print("Critical nulls:", [i.concept for i in out_b.null_set.critical_items])
    print("Flags:", out_b.flags)


if __name__ == "__main__":
    main()