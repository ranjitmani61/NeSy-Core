"""
tests/unit/test_continual.py
===============================
Tests for continual learning: EWC, SymbolicAnchor, MemoryBuffer, Scheduler.
"""

import pytest
from nesy.continual.learner import ContinualLearner, SymbolicAnchor
from nesy.continual.ewc import EWCRegularizer
from nesy.continual.memory_buffer import EpisodicMemoryBuffer, MemoryItem
from nesy.continual.scheduler import ConsolidationScheduler
from nesy.core.types import Predicate, SymbolicRule
from nesy.core.exceptions import ContinualLearningConflict


# ─── EWCRegularizer ────────────────────────────────────────────


class TestEWCRegularizer:
    def test_penalty_zero_before_consolidation(self):
        """No snapshots → penalty = 0."""
        ewc = EWCRegularizer(lambda_ewc=1000.0)
        assert ewc.penalty({"w": 5.0}) == 0.0

    def test_penalty_zero_when_no_drift(self):
        """θ = θ* → Δ = 0 → penalty = 0."""
        ewc = EWCRegularizer(lambda_ewc=1000.0)
        params = {"w1": 0.5, "w2": 0.3}
        fisher = {"w1": 1.0, "w2": 1.0}
        ewc.consolidate("task_a", params, fisher)
        assert ewc.penalty(params) == 0.0

    def test_penalty_increases_with_drift(self):
        """Drift from θ* produces positive penalty proportional to F × Δ²."""
        ewc = EWCRegularizer(lambda_ewc=1000.0)
        ewc.consolidate("task_a", {"w": 0.0}, {"w": 2.0})
        penalty = ewc.penalty({"w": 1.0})
        # (λ/2) × F × Δ² = 500 × 2.0 × 1.0 = 1000.0
        assert penalty == pytest.approx(1000.0)

    def test_penalty_proportional_to_fisher(self):
        """Higher Fisher → higher penalty for same drift."""
        ewc = EWCRegularizer(lambda_ewc=100.0)
        ewc.consolidate("t", {"w": 0.0}, {"w": 1.0})
        p1 = ewc.penalty({"w": 1.0})

        ewc2 = EWCRegularizer(lambda_ewc=100.0)
        ewc2.consolidate("t", {"w": 0.0}, {"w": 10.0})
        p2 = ewc2.penalty({"w": 1.0})
        assert p2 > p1

    def test_multiple_tasks_accumulate(self):
        """Penalty accumulates across consolidated tasks."""
        ewc = EWCRegularizer(lambda_ewc=100.0)
        ewc.consolidate("a", {"w": 0.0}, {"w": 1.0})
        ewc.consolidate("b", {"w": 0.5}, {"w": 1.0})
        penalty = ewc.penalty({"w": 2.0})
        assert penalty > 0.0

    def test_unknown_param_ignored(self):
        """Params missing from snapshot are silently skipped."""
        ewc = EWCRegularizer(lambda_ewc=100.0)
        ewc.consolidate("a", {"w1": 0.0}, {"w1": 1.0})
        penalty = ewc.penalty({"w1": 1.0, "w2": 999.0})
        # w2 is not in snapshot → contributes nothing
        expected = (100.0 / 2.0) * 1.0 * 1.0
        assert penalty == pytest.approx(expected)


# ─── SymbolicAnchor ────────────────────────────────────────────


class TestSymbolicAnchor:
    def _make_rule(self, rule_id: str) -> SymbolicRule:
        return SymbolicRule(
            id=rule_id,
            antecedents=[Predicate("A", ("?x",))],
            consequents=[Predicate("B", ("?x",))],
            weight=1.0,
        )

    def test_add_and_retrieve(self):
        anchor = SymbolicAnchor()
        rule = self._make_rule("r1")
        anchor.add(rule)
        assert anchor.contains("r1")
        assert anchor.get("r1") is rule

    def test_duplicate_raises_conflict(self):
        anchor = SymbolicAnchor()
        anchor.add(self._make_rule("r1"))
        with pytest.raises(ContinualLearningConflict):
            anchor.add(self._make_rule("r1"))

    def test_immutable_flag_set(self):
        anchor = SymbolicAnchor()
        rule = self._make_rule("r1")
        anchor.add(rule)
        assert rule.immutable is True

    def test_all_rules(self):
        anchor = SymbolicAnchor()
        anchor.add(self._make_rule("r1"))
        anchor.add(self._make_rule("r2"))
        assert len(anchor.all_rules()) == 2

    def test_len(self):
        anchor = SymbolicAnchor()
        assert len(anchor) == 0
        anchor.add(self._make_rule("r1"))
        assert len(anchor) == 1


# ─── EpisodicMemoryBuffer ─────────────────────────────────────


class TestEpisodicMemoryBuffer:
    def test_add_within_capacity(self):
        buf = EpisodicMemoryBuffer(max_size=10)
        for i in range(5):
            buf.add(MemoryItem(data=i, task_id="t1"))
        assert buf.size == 5

    def test_reservoir_sampling_bounds(self):
        """Buffer never exceeds max_size even with many insertions."""
        buf = EpisodicMemoryBuffer(max_size=10)
        for i in range(1000):
            buf.add(MemoryItem(data=i, task_id="t1"))
        assert buf.size == 10
        assert buf.total_seen == 1000

    def test_sample_returns_requested_count(self):
        buf = EpisodicMemoryBuffer(max_size=20)
        for i in range(20):
            buf.add(MemoryItem(data=i, task_id="t1"))
        sampled = buf.sample(5)
        assert len(sampled) == 5

    def test_sample_clamped_to_buffer_size(self):
        buf = EpisodicMemoryBuffer(max_size=5)
        for i in range(3):
            buf.add(MemoryItem(data=i, task_id="t1"))
        sampled = buf.sample(10)
        assert len(sampled) == 3

    def test_get_by_task(self):
        buf = EpisodicMemoryBuffer(max_size=100)
        for i in range(5):
            buf.add(MemoryItem(data=i, task_id="t1"))
        for i in range(3):
            buf.add(MemoryItem(data=i, task_id="t2"))
        assert len(buf.get_by_task("t1")) == 5
        assert len(buf.get_by_task("t2")) == 3

    def test_clear_task(self):
        buf = EpisodicMemoryBuffer(max_size=100)
        for i in range(5):
            buf.add(MemoryItem(data=i, task_id="t1"))
        removed = buf.clear_task("t1")
        assert removed == 5
        assert buf.size == 0

    def test_task_distribution(self):
        buf = EpisodicMemoryBuffer(max_size=100)
        buf.add(MemoryItem(data=1, task_id="a"))
        buf.add(MemoryItem(data=2, task_id="a"))
        buf.add(MemoryItem(data=3, task_id="b"))
        dist = buf.task_distribution
        assert dist["a"] == 2
        assert dist["b"] == 1


# ─── ConsolidationScheduler ───────────────────────────────────


class TestConsolidationScheduler:
    def test_no_consolidation_below_min(self):
        sched = ConsolidationScheduler(samples_trigger=10, min_samples_before=5)
        for _ in range(3):
            sched.on_sample()
        assert sched.should_consolidate() is False

    def test_consolidation_at_trigger(self):
        sched = ConsolidationScheduler(samples_trigger=10, min_samples_before=5)
        for _ in range(10):
            sched.on_sample()
        assert sched.should_consolidate() is True

    def test_force_flag_triggers(self):
        sched = ConsolidationScheduler(samples_trigger=1000, min_samples_before=5)
        for _ in range(6):
            sched.on_sample()
        sched.signal_task_boundary()
        assert sched.should_consolidate() is True

    def test_reset_clears_counter(self):
        sched = ConsolidationScheduler(samples_trigger=10, min_samples_before=5)
        for _ in range(10):
            sched.on_sample()
        assert sched.should_consolidate() is True
        sched.reset()
        assert sched.should_consolidate() is False


# ─── ContinualLearner (integration) ───────────────────────────


class TestContinualLearner:
    def test_ewc_penalty_with_no_snapshots(self):
        learner = ContinualLearner(lambda_ewc=1000.0)
        assert learner.ewc_penalty({"w": 1.0}) == 0.0

    def test_add_symbolic_anchor(self):
        learner = ContinualLearner()
        rule = SymbolicRule(
            id="anchor_rule",
            antecedents=[Predicate("A", ("?x",))],
            consequents=[Predicate("B", ("?x",))],
            weight=1.0,
        )
        learner.add_symbolic_anchor(rule)
        assert learner.anchor_count == 1
        assert learner.anchor.contains("anchor_rule")

    def test_replay_rules_empty_initially(self):
        learner = ContinualLearner()
        assert learner.get_replay_rules() == []

    def test_consolidated_tasks_tracking(self):
        learner = ContinualLearner()
        learner.consolidate(
            task_id="t1",
            model_params={"w": 0.5},
            compute_fisher_fn=lambda p: {k: 1.0 for k in p},
        )
        assert "t1" in learner.consolidated_tasks
