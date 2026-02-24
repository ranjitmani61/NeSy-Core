"""tests/integration/test_neural_symbolic.py — Neural-Symbolic bridge integration"""

import math
import pytest
from nesy.api.nesy_model import NeSyModel
from nesy.continual.learner import SymbolicAnchor
from nesy.continual.memory_buffer import EpisodicMemoryBuffer, MemoryItem
from nesy.continual.replay import RandomReplay, SymbolicAnchorReplay
from nesy.continual.scheduler import ConsolidationScheduler
from nesy.core.types import Predicate, SymbolicRule
from nesy.neural.bridge import NeuralSymbolicBridge
from nesy.neural.grounding import SymbolGrounder, PredicatePrototype
from nesy.neural.loss import LukasiewiczLogic


@pytest.fixture
def grounder_with_prototypes():
    grounder = SymbolGrounder(threshold=0.70)
    # Register fever prototype
    grounder.register(
        PredicatePrototype(
            predicate=Predicate("HasSymptom", ("?p", "fever")),
            prototype=[1.0, 0.0, 0.0],
        )
    )
    grounder.register(
        PredicatePrototype(
            predicate=Predicate("HasLabResult", ("?p", "elevated_wbc")),
            prototype=[0.0, 1.0, 0.0],
        )
    )
    return grounder


class TestSymbolGrounder:
    def test_identical_embedding_grounds_correctly(self, grounder_with_prototypes):
        embedding = [1.0, 0.0, 0.0]
        grounded = grounder_with_prototypes.ground(embedding)
        assert len(grounded) == 1
        assert grounded[0].predicate.name == "HasSymptom"

    def test_zero_embedding_grounds_to_nothing(self, grounder_with_prototypes):
        embedding = [0.0, 0.0, 0.0]
        grounded = grounder_with_prototypes.ground(embedding)
        assert len(grounded) == 0

    def test_prototype_from_examples(self, grounder_with_prototypes):
        examples = [[1.0, 0.0, 0.1], [0.9, 0.0, 0.0], [1.0, 0.1, 0.0]]
        proto = grounder_with_prototypes.build_prototype_from_examples(
            predicate=Predicate("HasSymptom", ("?p", "cough")),
            example_embeddings=examples,
        )
        assert proto is not None
        # Prototype should be approximately unit-normalised
        norm = math.sqrt(sum(x * x for x in proto.prototype))
        assert abs(norm - 1.0) < 0.01


class TestNeuralSymbolicBridge:
    def test_neural_to_symbolic(self, grounder_with_prototypes):
        bridge = NeuralSymbolicBridge(grounder_with_prototypes)
        predicates, conf = bridge.neural_to_symbolic([1.0, 0.0, 0.0])
        assert len(predicates) > 0
        assert conf > 0.70

    def test_symbolic_loss_zero_no_violations(self, grounder_with_prototypes):
        bridge = NeuralSymbolicBridge(grounder_with_prototypes)
        penalty = bridge.symbolic_to_loss([1.0, 0.0, 0.0], violated_rules=[])
        assert penalty == 0.0

    def test_symbolic_loss_positive_with_violations(self, grounder_with_prototypes):
        bridge = NeuralSymbolicBridge(grounder_with_prototypes, symbolic_loss_alpha=0.5)
        violated = [SymbolicRule("r1", [Predicate("A", ())], [Predicate("B", ())], 1.0)]
        penalty = bridge.symbolic_to_loss([1.0, 0.0, 0.0], violated_rules=violated)
        assert penalty > 0.0


class TestLukasiewiczLogic:
    def test_and_one_one(self):
        assert LukasiewiczLogic.AND(1.0, 1.0) == pytest.approx(1.0)

    def test_and_half_half(self):
        assert LukasiewiczLogic.AND(0.5, 0.5) == pytest.approx(0.0)

    def test_or_zero_zero(self):
        assert LukasiewiczLogic.OR(0.0, 0.0) == pytest.approx(0.0)

    def test_or_one_zero(self):
        assert LukasiewiczLogic.OR(1.0, 0.0) == pytest.approx(1.0)

    def test_implies_true_true(self):
        assert LukasiewiczLogic.IMPLIES(1.0, 1.0) == pytest.approx(1.0)

    def test_implies_true_false(self):
        assert LukasiewiczLogic.IMPLIES(1.0, 0.0) == pytest.approx(0.0)

    def test_not(self):
        assert LukasiewiczLogic.NOT(0.3) == pytest.approx(0.7)

    def test_rule_satisfaction_satisfied(self):
        sat = LukasiewiczLogic.rule_satisfaction([1.0, 1.0], [1.0])
        assert sat == pytest.approx(1.0)

    def test_rule_satisfaction_violated(self):
        sat = LukasiewiczLogic.rule_satisfaction([1.0, 1.0], [0.0])
        assert sat == pytest.approx(0.0)


# tests/integration/test_continual_learning.py — Continual learning integration


class TestContinualLearningFull:
    """Integration tests: symbolic layer does NOT forget across tasks."""

    def test_two_tasks_no_forgetting(self):
        model = NeSyModel()

        # Task A
        model.learn(
            SymbolicRule("task_a", [Predicate("A", ("?x",))], [Predicate("B", ("?x",))], 0.9)
        )
        output_a1 = model.reason({Predicate("A", ("x1",))})
        assert "B" in output_a1.answer

        # Task B
        model.learn(
            SymbolicRule("task_b", [Predicate("C", ("?x",))], [Predicate("D", ("?x",))], 0.9)
        )
        output_b = model.reason({Predicate("C", ("x2",))})
        assert "D" in output_b.answer

        # Task A still works
        output_a2 = model.reason({Predicate("A", ("x1",))})
        assert "B" in output_a2.answer

    def test_anchor_cannot_be_overwritten(self):
        model = NeSyModel()
        model.learn(
            SymbolicRule("permanent", [Predicate("X", ("?a",))], [Predicate("Y", ("?a",))], 1.0),
            make_anchor=True,
        )
        with pytest.raises(Exception):
            model.learn(
                SymbolicRule(
                    "permanent", [Predicate("X", ("?a",))], [Predicate("Z", ("?a",))], 1.0
                ),
                make_anchor=True,
            )

    def test_anchor_rule_count_increases(self):
        model = NeSyModel()
        model.learn(
            SymbolicRule("anc1", [Predicate("A", ("?x",))], [Predicate("B", ("?x",))], 1.0),
            make_anchor=True,
        )
        model.learn(
            SymbolicRule("anc2", [Predicate("C", ("?x",))], [Predicate("D", ("?x",))], 1.0),
            make_anchor=True,
        )
        assert model.anchored_rules == 2


class TestReplayStrategies:
    def test_random_replay_samples(self):
        buf = EpisodicMemoryBuffer(max_size=100)
        for i in range(50):
            buf.add(MemoryItem(data=i, task_id="t"))
        replay = RandomReplay(buf, replay_ratio=0.2)
        samples = replay.sample(20)
        assert len(samples) == 4  # 0.2 × 20 = 4

    def test_symbolic_anchor_replay(self):
        from nesy.symbolic.engine import SymbolicEngine

        anchor = SymbolicAnchor()
        rule = SymbolicRule("r1", [Predicate("A", ("?x",))], [Predicate("B", ("?x",))], 1.0)
        anchor.add(rule)
        engine = SymbolicEngine()
        replayer = SymbolicAnchorReplay(anchor)
        n = replayer.replay_to_engine(engine)
        assert n == 1


class TestConsolidationScheduler:
    def test_triggers_on_count(self):
        sched = ConsolidationScheduler(samples_trigger=5, min_samples_before=1)
        for _ in range(5):
            sched.on_sample()
        assert sched.should_consolidate() is True

    def test_resets_after_consolidation(self):
        sched = ConsolidationScheduler(samples_trigger=3, min_samples_before=1)
        for _ in range(3):
            sched.on_sample()
        assert sched.should_consolidate()
        sched.reset()
        assert not sched.should_consolidate()

    def test_triggers_on_quality_drop(self):
        sched = ConsolidationScheduler(quality_threshold=0.70, min_samples_before=1)
        sched.on_sample()
        sched.on_quality_update(0.60)  # below threshold
        assert sched.should_consolidate() is True
