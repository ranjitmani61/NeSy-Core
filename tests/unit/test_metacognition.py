"""tests/unit/test_metacognition.py"""
import pytest
from nesy.metacognition.monitor import MetaCognitionMonitor
from nesy.core.types import NullSet, NullItem, NullType, PresentSet


def _make_null_set(type3_count=0, type2_count=0):
    items = []
    present = PresentSet(concepts={"fever"}, context_type="medical")
    for i in range(type3_count):
        items.append(NullItem(
            concept=f"critical_{i}", weight=0.8,
            null_type=NullType.TYPE3_CRITICAL,
            expected_because_of=["fever"], criticality=2.0,
        ))
    for i in range(type2_count):
        items.append(NullItem(
            concept=f"meaningful_{i}", weight=0.5,
            null_type=NullType.TYPE2_MEANINGFUL,
            expected_because_of=["fever"], criticality=1.0,
        ))
    return NullSet(items=items, present_set=present)


def test_ok_status_when_confident():
    monitor = MetaCognitionMonitor(doubt_threshold=0.50)
    null_set = _make_null_set()
    conf, trace, status, flags = monitor.evaluate(
        answer="test", neural_confidence=0.90, symbolic_confidence=0.90,
        reasoning_steps=[], logic_clauses=[], null_set=null_set,
    )
    from nesy.core.types import OutputStatus
    assert status == OutputStatus.OK


def test_rejected_on_critical_null():
    monitor = MetaCognitionMonitor(doubt_threshold=0.50)
    null_set = _make_null_set(type3_count=1)
    conf, trace, status, flags = monitor.evaluate(
        answer="test", neural_confidence=0.90, symbolic_confidence=0.90,
        reasoning_steps=[], logic_clauses=[], null_set=null_set,
    )
    from nesy.core.types import OutputStatus
    assert status == OutputStatus.REJECTED
    assert len(flags) > 0


def test_flagged_on_low_confidence():
    monitor = MetaCognitionMonitor(doubt_threshold=0.80)
    null_set = _make_null_set()
    conf, trace, status, flags = monitor.evaluate(
        answer="test", neural_confidence=0.50, symbolic_confidence=0.50,
        reasoning_steps=[], logic_clauses=[], null_set=null_set,
    )
    from nesy.core.types import OutputStatus
    assert status in (OutputStatus.FLAGGED, OutputStatus.UNCERTAIN)


"""tests/unit/test_continual.py"""
from nesy.continual.learner import ContinualLearner, SymbolicAnchor
from nesy.core.types import Predicate, SymbolicRule
from nesy.core.exceptions import ContinualLearningConflict
import pytest


def test_ewc_penalty_zero_with_no_snapshots():
    learner = ContinualLearner(lambda_ewc=1000.0)
    params = {"w1": 0.5, "w2": 0.3}
    assert learner.ewc_penalty(params) == 0.0


def test_ewc_penalty_increases_with_drift():
    learner = ContinualLearner(lambda_ewc=1000.0)
    params_orig = {"w1": 0.5, "w2": 0.3}
    fisher = {"w1": 1.0, "w2": 1.0}
    learner._ewc_consolidated = None

    from nesy.continual.ewc import EWCRegularizer
    ewc = EWCRegularizer(lambda_ewc=1000.0)
    ewc.consolidate("task_a", params_orig, fisher)

    params_new = {"w1": 1.0, "w2": 0.3}   # w1 drifted
    penalty = ewc.penalty(params_new)
    assert penalty > 0.0


def test_symbolic_anchor_immutable():
    anchor = SymbolicAnchor()
    rule = SymbolicRule(
        id="test_anchor",
        antecedents=[Predicate("A", ("?x",))],
        consequents=[Predicate("B", ("?x",))],
        weight=1.0,
    )
    anchor.add(rule)
    with pytest.raises(ContinualLearningConflict):
        anchor.add(rule)   # adding same ID again


"""tests/unit/test_grounding.py"""
import math
from nesy.neural.grounding import SymbolGrounder, PredicatePrototype
from nesy.core.types import Predicate


def test_grounding_above_threshold():
    grounder = SymbolGrounder(threshold=0.90)
    prototype = [1.0, 0.0, 0.0]
    grounder.register(PredicatePrototype(
        predicate=Predicate("HasSymptom", ("?p", "fever")),
        prototype=prototype,
    ))
    embedding = [1.0, 0.0, 0.0]   # identical → cosine = 1.0
    grounded = grounder.ground(embedding)
    assert len(grounded) == 1
    assert grounded[0].grounding_confidence == pytest.approx(1.0, abs=1e-6)


def test_grounding_below_threshold_returns_empty():
    grounder = SymbolGrounder(threshold=0.90)
    prototype = [1.0, 0.0, 0.0]
    grounder.register(PredicatePrototype(
        predicate=Predicate("HasSymptom", ("?p", "fever")),
        prototype=prototype,
    ))
    embedding = [0.0, 1.0, 0.0]   # orthogonal → cosine = 0.0
    grounded = grounder.ground(embedding)
    assert len(grounded) == 0


"""tests/unit/test_confidence.py"""
from nesy.metacognition.confidence import (
    compute_factual, compute_reasoning, compute_boundary, build_confidence_report,
)
from nesy.core.types import NullSet, NullItem, NullType, PresentSet


def test_factual_perfect_with_no_nulls():
    ps = PresentSet(concepts={"fever"}, context_type="general")
    ns = NullSet(items=[], present_set=ps)
    assert compute_factual(ns) == pytest.approx(1.0)


def test_factual_decreases_with_critical_null():
    ps = PresentSet(concepts={"fever"}, context_type="general")
    ns = NullSet(items=[
        NullItem("bp", 0.8, NullType.TYPE3_CRITICAL, ["fever"], 2.0)
    ], present_set=ps)
    score = compute_factual(ns)
    assert score < 1.0 and score > 0.0


def test_reasoning_with_all_satisfied():
    from nesy.core.types import LogicClause, LogicConnective, Predicate
    clauses = [
        LogicClause([Predicate("A", ())], LogicConnective.AND, True, 0.9),
        LogicClause([Predicate("B", ())], LogicConnective.AND, True, 0.8),
    ]
    score = compute_reasoning(1.0, clauses)
    assert score == pytest.approx(1.0, abs=0.01)


def test_reasoning_decreases_with_unsatisfied():
    from nesy.core.types import LogicClause, LogicConnective, Predicate
    clauses = [
        LogicClause([Predicate("A", ())], LogicConnective.AND, True,  0.9),
        LogicClause([Predicate("B", ())], LogicConnective.AND, False, 0.8),
    ]
    score = compute_reasoning(0.9, clauses)
    assert score < 0.9
