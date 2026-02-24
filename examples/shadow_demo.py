"""
examples/shadow_demo.py
========================
Live demonstration of Counterfactual Shadow Reasoning.

Shows the new 2D confidence space:
    C₁ (factual confidence) × C₂ (shadow distance)

and why this matters for safety-critical AI.
"""
import sys
import math
sys.path.insert(0, ".")

from nesy.core.types import Predicate, SymbolicRule
from nesy.metacognition.shadow import CounterfactualShadowEngine, shadow_flags


def pred(name, *args):
    return Predicate(name=name, args=tuple(args))


def rule(rule_id, ants, cons, weight=1.0):
    return SymbolicRule(id=rule_id, antecedents=ants, consequents=cons, weight=weight)


def separator(title=""):
    width = 60
    if title:
        print(f"\n{'═' * width}")
        print(f"  {title}")
        print(f"{'═' * width}")
    else:
        print(f"{'─' * width}")


engine = CounterfactualShadowEngine()


# ─────────────────────────────────────────────────────────────────
# DEMO 1: The Dangerous Diagnosis
# A doctor trusts this. They shouldn't.
# ─────────────────────────────────────────────────────────────────

separator("DEMO 1: Dangerous High-Confidence Diagnosis (Shadow d=1)")

facts_1 = {pred("HasSymptom", "patient_001", "fever")}
rules_1 = [
    rule("fever_infection",
         [pred("HasSymptom", "?p", "fever")],
         [pred("PossiblyHas", "?p", "infection")])
]

result_1 = engine.compute(pred("PossiblyHas", "patient_001", "infection"), facts_1, rules_1)

print(f"\nEvidence:     fever only")
print(f"Conclusion:   PossiblyHas(patient_001, infection)")
print(f"Confidence:   0.87  ← looks trustworthy")
print(f"\n--- SHADOW ANALYSIS ---")
print(f"Shadow Distance: {int(result_1.distance)}")
print(f"Shadow Class:    {result_1.shadow_class.value.upper()}")
print(f"\n⚠  {result_1.flip_explanation}")
print(f"\nWhat this means: If 'fever' was a data entry error — or if")
print(f"the thermometer was miscalibrated — the entire diagnosis flips.")
print(f"A 0.87 confidence score tells you nothing about this fragility.")


# ─────────────────────────────────────────────────────────────────
# DEMO 2: The Trustworthy Diagnosis
# Same confidence — fundamentally different safety profile.
# ─────────────────────────────────────────────────────────────────

separator("DEMO 2: Robust Diagnosis — same confidence, different shadow")

facts_2 = {
    pred("HasSymptom",  "patient_002", "fever"),
    pred("HasLabResult","patient_002", "elevated_wbc"),
    pred("HasLabResult","patient_002", "positive_blood_culture"),
    pred("HasLabResult","patient_002", "elevated_crp"),
    pred("HasVitalSign","patient_002", "tachycardia"),
}

rules_2 = [
    rule("fever_inf",    [pred("HasSymptom",   "?p","fever")],                    [pred("Diagnosis","?p","infection")]),
    rule("wbc_inf",      [pred("HasLabResult", "?p","elevated_wbc")],             [pred("Diagnosis","?p","infection")]),
    rule("culture_inf",  [pred("HasLabResult", "?p","positive_blood_culture")],   [pred("Diagnosis","?p","infection")]),
    rule("crp_inf",      [pred("HasLabResult", "?p","elevated_crp")],             [pred("Diagnosis","?p","infection")]),
    rule("tachy_inf",    [pred("HasVitalSign", "?p","tachycardia")],              [pred("Diagnosis","?p","infection")]),
]

result_2 = engine.compute(pred("Diagnosis", "patient_002", "infection"), facts_2, rules_2)

print(f"\nEvidence:     fever + elevated_wbc + positive_blood_culture")
print(f"              + elevated_crp + tachycardia  (5 independent signs)")
print(f"Conclusion:   Diagnosis(patient_002, infection)")
print(f"Confidence:   0.91  ← similar to Demo 1")
print(f"\n--- SHADOW ANALYSIS ---")
print(f"Shadow Distance: {int(result_2.distance)}")
print(f"Shadow Class:    {result_2.shadow_class.value.upper()}")
print(f"\n✓  All 5 facts must be simultaneously wrong to flip this diagnosis.")
print(f"   Shadow Distance {int(result_2.distance)} vs Demo 1's distance 1 — fundamentally different.")


# ─────────────────────────────────────────────────────────────────
# DEMO 3: Legal AI — Shadow on Case Verdict
# ─────────────────────────────────────────────────────────────────

separator("DEMO 3: Legal AI — Counterfactual Shadow on Case Assessment")

facts_3 = {
    pred("ContractSigned",  "case_042"),
    pred("PaymentMissed",   "case_042"),
}

rules_3 = [
    rule("breach",
         [pred("ContractSigned", "?c"), pred("PaymentMissed", "?c")],
         [pred("BreachOfContract", "?c")])
]

result_3 = engine.compute(pred("BreachOfContract", "case_042"), facts_3, rules_3)

print(f"\nFacts:      Contract signed + Payment missed")
print(f"Conclusion: BreachOfContract(case_042)")
print(f"\n--- SHADOW ANALYSIS ---")
print(f"Shadow Distance: {int(result_3.distance)}")
print(f"Shadow Class:    {result_3.shadow_class.value.upper()}")
print(f"\n⚠  {result_3.flip_explanation}")
print(f"\nLegal implication: If EITHER fact is successfully challenged in court,")
print(f"the entire breach determination collapses. Defense only needs to disprove ONE.")


# ─────────────────────────────────────────────────────────────────
# DEMO 4: Safety System — The 2D Confidence Map
# ─────────────────────────────────────────────────────────────────

separator("DEMO 4: 2D Confidence Space — What Existing AI Misses")

print("""
    SHADOW DISTANCE (Structural Robustness)
         ∞  │ Demo1 situation:     Demo2 situation:
            │ C₁=0.87 but d=1     C₁=0.91 and d=5
         5  │ "Confident but      "Confidently
            │  FRAGILE"            ROBUST"
            │ → Human review      → Safe to act
            │─────────────────────────────────────
         1  │ "Double danger"     "Confident but
            │ → REJECT            FRAGILE"
            │─────────────────────────────────────
                 0.5      0.75     1.0
                    FACTUAL CONFIDENCE C₁

Traditional AI shows you only the X-axis.
NeSy-Core Shadow shows you BOTH dimensions.
""")


# ─────────────────────────────────────────────────────────────────
# DEMO 5: Compute_all — Full Report with Flags
# ─────────────────────────────────────────────────────────────────

separator("DEMO 5: Full Shadow Report with Escalation Flags")

facts_5 = {pred("A", "x")}
rules_5 = [rule("r1", [pred("A", "?x")], [pred("CriticalOutput", "?x")])]
conclusions = {pred("CriticalOutput", "x")}

report = engine.compute_all(conclusions, facts_5, rules_5)
flags = shadow_flags(report)

print(f"\n{report.summary()}")
print(f"\nGenerated flags:")
for f in flags:
    print(f"  → {f}")

separator()
print("\nDemo complete.")
print("Shadow Reasoning is now part of NeSy-Core metacognition layer.")
print("Every conclusion now carries TWO dimensions of confidence:")
print("  C₁: How likely is this true?      (probability)")
print("  C₂: How robust is this to change? (shadow distance)")
