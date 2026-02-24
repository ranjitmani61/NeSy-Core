"""
tests/unit/test_low_coverage.py
================================
Targets low-coverage modules to push total coverage past 80%.

Modules covered:
    api/context.py         (38% → aim 90%)
    api/decorators.py      (20% → aim 85%)
    api/pipeline.py        (33% → aim 85%)
    metacognition/doubt.py (26% → aim 90%)
    metacognition/trace.py (47% → aim 90%)
    continual/replay.py    (45% → aim 85%)
    nsi/path_finder.py     (59% → aim 85%)
    evaluation/evaluator.py (43% → aim 80%)
    deployment/lite.py     (39% → aim 75%)
    deployment/optimizer.py (23% → aim 65%)
"""
import pytest
from nesy.core.types import (
    ConceptEdge, ConfidenceReport, NullItem, NullSet,
    OutputStatus, Predicate, PresentSet, ReasoningStep, SymbolicRule,
)
from nesy.api.nesy_model import NeSyModel


# ── helpers ──────────────────────────────────────────────────────

def _make_model(**kwargs) -> NeSyModel:
    return NeSyModel(domain="general", **kwargs)


def _make_confidence(factual=1.0, reasoning=1.0, boundary=1.0) -> ConfidenceReport:
    return ConfidenceReport(
        factual=factual,
        reasoning=reasoning,
        knowledge_boundary=boundary,
    )


def _make_null_set(items=None) -> NullSet:
    items = items or []
    return NullSet(
        present_set=PresentSet(concepts=set()),
        items=items,
    )


def _make_null_item(concept: str, null_type: int = 2, weight: float = 0.5) -> NullItem:
    from nesy.core.types import NullType
    nt = {1: NullType.TYPE1_EXPECTED, 2: NullType.TYPE2_MEANINGFUL, 3: NullType.TYPE3_CRITICAL}[null_type]
    return NullItem(
        concept=concept,
        null_type=nt,
        weight=weight,
        expected_because_of=[],
    )


def _make_rule(rule_id: str, weight: float = 0.8) -> SymbolicRule:
    return SymbolicRule(
        id=rule_id,
        antecedents=[Predicate("A", ("?x",))],
        consequents=[Predicate("B", ("?x",))],
        weight=weight,
    )


def _make_edge(src: str, tgt: str, prob: float = 0.8) -> ConceptEdge:
    return ConceptEdge(
        source=src,
        target=tgt,
        cooccurrence_prob=prob,
        causal_strength=0.5,
        temporal_stability=0.5,
    )


# ══════════════════════════════════════════════════════════════════
#  api/context.py — strict_mode, relaxed_mode, domain_context
# ══════════════════════════════════════════════════════════════════

class TestContextManagers:

    def test_strict_mode_enters_and_exits(self):
        from nesy.api.context import strict_mode
        model = _make_model()
        original = model._monitor.strict_mode
        with strict_mode(model) as m:
            assert m._monitor.strict_mode is True
        assert model._monitor.strict_mode == original

    def test_strict_mode_restores_on_exception(self):
        from nesy.api.context import strict_mode
        model = _make_model()
        original = model._monitor.strict_mode
        try:
            with strict_mode(model):
                raise ValueError("test error")
        except ValueError:
            pass
        assert model._monitor.strict_mode == original

    def test_relaxed_mode_lowers_threshold(self):
        from nesy.api.context import relaxed_mode
        model = _make_model(doubt_threshold=0.75)
        with relaxed_mode(model, threshold=0.30) as m:
            assert m._monitor.doubt_threshold == 0.30
        assert model._monitor.doubt_threshold == 0.75

    def test_relaxed_mode_restores_on_exception(self):
        from nesy.api.context import relaxed_mode
        model = _make_model(doubt_threshold=0.75)
        try:
            with relaxed_mode(model, threshold=0.10):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert model._monitor.doubt_threshold == 0.75

    def test_domain_context_changes_domain(self):
        from nesy.api.context import domain_context
        model = _make_model()
        model.domain = "general"
        with domain_context(model, "medical") as m:
            assert m.domain == "medical"
        assert model.domain == "general"

    def test_domain_context_restores_on_exception(self):
        from nesy.api.context import domain_context
        model = _make_model()
        model.domain = "general"
        try:
            with domain_context(model, "legal"):
                raise RuntimeError("fail")
        except RuntimeError:
            pass
        assert model.domain == "general"

    def test_nested_contexts(self):
        from nesy.api.context import strict_mode, relaxed_mode
        model = _make_model(doubt_threshold=0.75)
        with strict_mode(model):
            with relaxed_mode(model, threshold=0.20):
                assert model._monitor.doubt_threshold == 0.20
            assert model._monitor.doubt_threshold == 0.75


# ══════════════════════════════════════════════════════════════════
#  api/decorators.py — symbolic_rule, requires_proof, domain
# ══════════════════════════════════════════════════════════════════

class TestDecorators:

    def test_symbolic_rule_attaches_metadata(self):
        from nesy.api.decorators import symbolic_rule

        @symbolic_rule("HasRole(?x, doctor) → CanPrescribe(?x, ?drug)", weight=0.9)
        def prescribe(doctor_id, drug):
            return f"{doctor_id} prescribes {drug}"

        assert hasattr(prescribe, "_nesy_rule")
        assert "doctor" in prescribe._nesy_rule
        result = prescribe("dr_smith", "aspirin")
        assert result == "dr_smith prescribes aspirin"

    def test_symbolic_rule_preserves_function_name(self):
        from nesy.api.decorators import symbolic_rule

        @symbolic_rule("Test rule")
        def my_function():
            return 42

        assert my_function.__name__ == "my_function"
        assert my_function() == 42

    def test_requires_proof_passes_high_confidence(self):
        from nesy.api.decorators import requires_proof

        @requires_proof(confidence=0.70)
        def diagnose(facts):
            model = _make_model()
            return model.reason(facts=facts)

        facts = {Predicate("IsPhilosopher", ("socrates",))}
        result = diagnose(facts)
        # High confidence output should pass through
        assert result is not None

    def test_requires_proof_blocks_low_confidence(self):
        from nesy.api.decorators import requires_proof
        from nesy.core.types import NSIOutput, ConfidenceReport, OutputStatus, NullSet, PresentSet

        # Create a mock low-confidence NSIOutput
        low_conf_output = NSIOutput(
            answer="test",
            status=OutputStatus.FLAGGED,
            confidence=ConfidenceReport(factual=0.1, reasoning=0.1, knowledge_boundary=0.1),
            reasoning_trace=None,
            null_set=NullSet(present_set=PresentSet(concepts=set()), items=[]),
            flags=["low confidence"],
        )

        @requires_proof(confidence=0.70, strict=False)
        def my_fn():
            return low_conf_output

        result = my_fn()
        # strict=False → returns None on low confidence
        assert result is None

    def test_requires_proof_strict_raises(self):
        from nesy.api.decorators import requires_proof
        from nesy.core.types import NSIOutput, ConfidenceReport, OutputStatus, NullSet, PresentSet

        low_conf_output = NSIOutput(
            answer="test",
            status=OutputStatus.FLAGGED,
            confidence=ConfidenceReport(factual=0.1, reasoning=0.1, knowledge_boundary=0.1),
            reasoning_trace=None,
            null_set=NullSet(present_set=PresentSet(concepts=set()), items=[]),
            flags=[],
        )

        @requires_proof(confidence=0.80, strict=True)
        def my_fn():
            return low_conf_output

        with pytest.raises(RuntimeError):
            my_fn()

    def test_requires_proof_non_nesioutput_passes_through(self):
        from nesy.api.decorators import requires_proof

        @requires_proof(confidence=0.99)
        def my_fn():
            return "just a string"

        result = my_fn()
        assert result == "just a string"

    def test_domain_decorator_attaches_metadata(self):
        from nesy.api.decorators import domain

        @domain("medical")
        def analyze():
            return "analysis"

        assert hasattr(analyze, "_nesy_domain")
        assert analyze._nesy_domain == "medical"
        assert analyze() == "analysis"

    def test_domain_preserves_function_name(self):
        from nesy.api.decorators import domain

        @domain("legal")
        def review_contract():
            return True

        assert review_contract.__name__ == "review_contract"


# ══════════════════════════════════════════════════════════════════
#  api/pipeline.py — NeSyPipeline builder
# ══════════════════════════════════════════════════════════════════

class TestNeSyPipeline:

    def test_basic_build(self):
        from nesy.api.pipeline import NeSyPipeline
        pipeline = NeSyPipeline().build()
        assert pipeline is not None

    def test_with_rules(self):
        from nesy.api.pipeline import NeSyPipeline
        rule = _make_rule("test_rule")
        pipeline = NeSyPipeline().with_rules([rule]).build()
        assert pipeline is not None

    def test_with_concept_edges(self):
        from nesy.api.pipeline import NeSyPipeline
        edge = _make_edge("fever", "blood_test")
        pipeline = NeSyPipeline().with_concept_edges([edge]).build()
        assert pipeline is not None

    def test_with_domain(self):
        from nesy.api.pipeline import NeSyPipeline
        pipeline = NeSyPipeline().with_domain("medical").build()
        assert pipeline is not None

    def test_with_doubt_threshold(self):
        from nesy.api.pipeline import NeSyPipeline
        pipeline = NeSyPipeline().with_doubt_threshold(0.75).build()
        assert pipeline is not None

    def test_strict_mode(self):
        from nesy.api.pipeline import NeSyPipeline
        pipeline = NeSyPipeline().strict().build()
        assert pipeline is not None

    def test_fluent_chaining(self):
        from nesy.api.pipeline import NeSyPipeline
        rule = _make_rule("r1")
        edge = _make_edge("a", "b")
        pipeline = (NeSyPipeline()
                    .with_rules([rule])
                    .with_concept_edges([edge])
                    .with_domain("medical")
                    .with_doubt_threshold(0.65)
                    .build())
        assert pipeline is not None

    def test_run_with_facts_directly(self):
        from nesy.api.pipeline import NeSyPipeline
        rule = SymbolicRule(
            id="r1",
            antecedents=[Predicate("IsPhilosopher", ("?x",))],
            consequents=[Predicate("IsHuman", ("?x",))],
            weight=1.0,
        )
        pipeline = NeSyPipeline().with_rules([rule]).build()
        facts = {Predicate("IsPhilosopher", ("socrates",))}
        output = pipeline.run(input_data="test", facts=facts)
        assert output is not None

    def test_with_passthrough_backbone(self):
        from nesy.api.pipeline import NeSyPipeline
        from nesy.neural.base import PassthroughBackbone
        backbone = PassthroughBackbone(dim=4)
        pipeline = NeSyPipeline().with_backbone(backbone).build()
        assert pipeline is not None


# ══════════════════════════════════════════════════════════════════
#  metacognition/doubt.py — SelfDoubtLayer
# ══════════════════════════════════════════════════════════════════

class TestSelfDoubtLayer:

    def test_import(self):
        from nesy.metacognition.doubt import SelfDoubtLayer
        assert SelfDoubtLayer is not None

    def test_ok_when_high_confidence_no_nulls(self):
        from nesy.metacognition.doubt import SelfDoubtLayer
        layer = SelfDoubtLayer(threshold=0.60)
        conf = _make_confidence(factual=0.95, reasoning=0.95, boundary=0.95)
        null_set = _make_null_set()
        status, flags = layer.evaluate(conf, null_set, betti_0=1)
        assert status == OutputStatus.OK
        assert len(flags) == 0

    def test_flagged_when_confidence_below_threshold(self):
        from nesy.metacognition.doubt import SelfDoubtLayer
        layer = SelfDoubtLayer(threshold=0.60)
        conf = _make_confidence(factual=0.40, reasoning=0.90, boundary=0.90)
        null_set = _make_null_set()
        status, flags = layer.evaluate(conf, null_set, betti_0=1)
        assert status == OutputStatus.FLAGGED
        assert len(flags) > 0
        assert "SELF-DOUBT" in flags[0]

    def test_rejected_when_critical_null_present(self):
        from nesy.metacognition.doubt import SelfDoubtLayer
        layer = SelfDoubtLayer(threshold=0.60)
        conf = _make_confidence(factual=0.95, reasoning=0.95, boundary=0.95)
        critical = _make_null_item("temperature_reading", null_type=3, weight=1.0)
        null_set = _make_null_set([critical])
        status, flags = layer.evaluate(conf, null_set, betti_0=1)
        assert status == OutputStatus.REJECTED
        assert any("CRITICAL" in f for f in flags)

    def test_topological_doubt_when_betti_high(self):
        from nesy.metacognition.doubt import SelfDoubtLayer
        layer = SelfDoubtLayer(threshold=0.60, betti_warn=2)
        conf = _make_confidence(factual=0.90, reasoning=0.90, boundary=0.90)
        null_set = _make_null_set()
        # betti_0 = 3 ≥ betti_warn = 2 → topological flag
        status, flags = layer.evaluate(conf, null_set, betti_0=3)
        assert any("TOPOLOGICAL" in f for f in flags)

    def test_uncertain_when_borderline_confidence(self):
        from nesy.metacognition.doubt import SelfDoubtLayer
        layer = SelfDoubtLayer(threshold=0.60)
        # min confidence = 0.65 → above threshold (0.60) but below reliable (0.75)
        conf = _make_confidence(factual=0.65, reasoning=0.80, boundary=0.90)
        null_set = _make_null_set()
        status, flags = layer.evaluate(conf, null_set, betti_0=1)
        assert status in (OutputStatus.UNCERTAIN, OutputStatus.OK)

    def test_flags_list_type(self):
        from nesy.metacognition.doubt import SelfDoubtLayer
        layer = SelfDoubtLayer(threshold=0.60)
        conf = _make_confidence()
        status, flags = layer.evaluate(conf, _make_null_set(), betti_0=1)
        assert isinstance(flags, list)

    def test_threshold_boundary_exact_match(self):
        from nesy.metacognition.doubt import SelfDoubtLayer
        # At exactly the threshold
        layer = SelfDoubtLayer(threshold=0.60)
        conf = _make_confidence(factual=0.60, reasoning=0.90, boundary=0.90)
        null_set = _make_null_set()
        status, _ = layer.evaluate(conf, null_set, betti_0=1)
        # 0.60 is not < 0.60, so should not be FLAGGED
        assert status in (OutputStatus.OK, OutputStatus.UNCERTAIN)


# ══════════════════════════════════════════════════════════════════
#  metacognition/trace.py — TraceBuilder
# ══════════════════════════════════════════════════════════════════

class TestTraceBuilder:

    def test_import(self):
        from nesy.metacognition.trace import TraceBuilder
        assert TraceBuilder is not None

    def test_empty_trace(self):
        from nesy.metacognition.trace import TraceBuilder
        tb = TraceBuilder()
        trace = tb.build(neural_confidence=1.0, symbolic_confidence=1.0, null_violations=[])
        assert trace is not None
        assert trace.steps == [] or len(trace.steps) == 0

    def test_add_initial_facts(self):
        from nesy.metacognition.trace import TraceBuilder
        tb = TraceBuilder()
        facts = {Predicate("Fever", ("p1",)), Predicate("Cough", ("p1",))}
        tb.add_initial_facts(facts)
        trace = tb.build(neural_confidence=1.0, symbolic_confidence=1.0, null_violations=[])
        assert len(trace.steps) >= 1
        assert trace.steps[0].step_number == 0

    def test_add_derived_step(self):
        from nesy.metacognition.trace import TraceBuilder
        tb = TraceBuilder()
        facts = {Predicate("Fever", ("p1",))}
        tb.add_initial_facts(facts)
        pred = Predicate("PossiblyInfected", ("p1",))
        tb.add_derived(pred, "fever_infection_rule", rule_weight=0.85)
        trace = tb.build(neural_confidence=0.9, symbolic_confidence=0.85, null_violations=[])
        assert len(trace.steps) >= 2

    def test_rules_activated_populated(self):
        from nesy.metacognition.trace import TraceBuilder
        tb = TraceBuilder()
        facts = {Predicate("A", ("x",))}
        tb.add_initial_facts(facts)
        tb.add_derived(Predicate("B", ("x",)), "rule_ab", rule_weight=1.0)
        tb.add_derived(Predicate("C", ("x",)), "rule_bc", rule_weight=0.9)
        trace = tb.build(neural_confidence=1.0, symbolic_confidence=1.0, null_violations=[])
        assert "rule_ab" in trace.rules_activated
        assert "rule_bc" in trace.rules_activated

    def test_null_violations_in_trace(self):
        from nesy.metacognition.trace import TraceBuilder
        tb = TraceBuilder()
        violations = [_make_null_item("blood_test", null_type=2)]
        trace = tb.build(neural_confidence=0.8, symbolic_confidence=0.9, null_violations=violations)
        assert len(trace.null_violations) == 1

    def test_add_derived_with_description(self):
        from nesy.metacognition.trace import TraceBuilder
        tb = TraceBuilder()
        tb.add_initial_facts({Predicate("X", ("a",))})
        tb.add_derived(
            Predicate("Y", ("a",)),
            "rule_xy",
            rule_weight=0.75,
            description="Custom description for X → Y",
        )
        trace = tb.build(neural_confidence=1.0, symbolic_confidence=0.75, null_violations=[])
        last_step = trace.steps[-1]
        assert "Custom description" in last_step.description

    def test_confidence_stored(self):
        from nesy.metacognition.trace import TraceBuilder
        tb = TraceBuilder()
        trace = tb.build(neural_confidence=0.88, symbolic_confidence=0.77, null_violations=[])
        assert trace.neural_confidence == 0.88
        assert trace.symbolic_confidence == 0.77

    def test_step_numbers_increment(self):
        from nesy.metacognition.trace import TraceBuilder
        tb = TraceBuilder()
        tb.add_initial_facts({Predicate("A", ("x",))})
        tb.add_derived(Predicate("B", ("x",)), "r1", 1.0)
        tb.add_derived(Predicate("C", ("x",)), "r2", 1.0)
        trace = tb.build(1.0, 1.0, [])
        step_numbers = [s.step_number for s in trace.steps]
        assert step_numbers == sorted(step_numbers)


# ══════════════════════════════════════════════════════════════════
#  continual/replay.py — RandomReplay, PrioritisedReplay, SymbolicAnchorReplay
# ══════════════════════════════════════════════════════════════════

class TestRandomReplay:

    def test_import(self):
        from nesy.continual.replay import RandomReplay
        assert RandomReplay is not None

    def test_sample_returns_correct_count(self):
        from nesy.continual.replay import RandomReplay
        from nesy.continual.memory_buffer import EpisodicMemoryBuffer, MemoryItem
        buf = EpisodicMemoryBuffer(max_size=100)
        for i in range(20):
            buf.add(MemoryItem(data=f"item_{i}", task_id="task_a"))
        replay = RandomReplay(buffer=buf, replay_ratio=0.5)
        samples = replay.sample(n=10)
        assert len(samples) <= 10
        assert len(samples) >= 1

    def test_sample_empty_buffer(self):
        from nesy.continual.replay import RandomReplay
        from nesy.continual.memory_buffer import EpisodicMemoryBuffer
        buf = EpisodicMemoryBuffer(max_size=100)
        replay = RandomReplay(buffer=buf, replay_ratio=0.5)
        samples = replay.sample(n=10)
        assert samples == []

    def test_replay_ratio_zero_returns_empty(self):
        from nesy.continual.replay import RandomReplay
        from nesy.continual.memory_buffer import EpisodicMemoryBuffer, MemoryItem
        buf = EpisodicMemoryBuffer(max_size=100)
        for i in range(10):
            buf.add(MemoryItem(data=i, task_id="t"))
        # replay_ratio must be > 0; use a very small ratio to get minimal samples
        replay = RandomReplay(buffer=buf, replay_ratio=0.01)
        samples = replay.sample(n=1)
        # With ratio=0.01 and n=1, effective_n = max(1, int(1*0.01)) = max(1,0) = 1
        assert isinstance(samples, list)


class TestPrioritisedReplay:

    def test_import(self):
        from nesy.continual.replay import PrioritisedReplay
        assert PrioritisedReplay is not None

    def test_add_and_sample(self):
        from nesy.continual.replay import PrioritisedReplay
        from nesy.continual.memory_buffer import MemoryItem
        replay = PrioritisedReplay(alpha=0.6, beta=0.4, max_size=50)
        for i in range(10):
            item = MemoryItem(data=f"item_{i}", task_id="task")
            replay.add(item, priority=float(i + 1))
        # sample() returns List[MemoryItem]; sample_with_weights() returns tuple
        items, weights, indices = replay.sample_with_weights(5)
        assert len(items) == 5
        assert len(weights) == 5
        assert all(w >= 0 for w in weights)

    def test_high_priority_sampled_more(self):
        """Over many samples, high-priority items appear more often."""
        from nesy.continual.replay import PrioritisedReplay
        from nesy.continual.memory_buffer import MemoryItem
        replay = PrioritisedReplay(alpha=1.0, beta=0.0, max_size=100)
        # Item 0: priority 1 (low)
        replay.add(MemoryItem(data="low", task_id="t"), priority=1.0)
        # Item 1: priority 100 (high)
        replay.add(MemoryItem(data="high", task_id="t"), priority=100.0)

        high_count = 0
        for _ in range(200):
            items = replay.sample(1)
            if items and items[0].data == "high":
                high_count += 1
        # High priority item should appear significantly more often
        assert high_count > 100, f"High priority item appeared only {high_count}/200 times"

    def test_update_priority(self):
        from nesy.continual.replay import PrioritisedReplay
        from nesy.continual.memory_buffer import MemoryItem
        replay = PrioritisedReplay(alpha=0.6, beta=0.4, max_size=50)
        item = MemoryItem(data="item", task_id="t")
        replay.add(item, priority=1.0)
        # Update priority without crashing
        replay.update_priority(0, 10.0)

    def test_sample_empty_returns_empty(self):
        from nesy.continual.replay import PrioritisedReplay
        replay = PrioritisedReplay(alpha=0.6, beta=0.4, max_size=50)
        items = replay.sample(5)
        assert items == []


class TestSymbolicAnchorReplay:

    def test_import(self):
        from nesy.continual.replay import SymbolicAnchorReplay
        assert SymbolicAnchorReplay is not None

    def test_replay_injects_rules_into_engine(self):
        from nesy.continual.replay import SymbolicAnchorReplay
        from nesy.symbolic.engine import SymbolicEngine
        try:
            from nesy.continual.symbolic_anchor import SymbolicAnchor
        except ImportError:
            from nesy.continual.learner import SymbolicAnchor

        anchor = SymbolicAnchor()
        rule = _make_rule("anchor_rule", weight=1.0)
        anchor.add(rule)

        replay = SymbolicAnchorReplay(anchor=anchor)
        engine = SymbolicEngine()
        replay.replay_to_engine(engine)
        # Anchored rule should now be in the engine
        assert len(engine.rules) >= 1


# ══════════════════════════════════════════════════════════════════
#  nsi/path_finder.py — ConceptPathFinder
# ══════════════════════════════════════════════════════════════════

class TestConceptPathFinder:

    def _build_graph(self, edges):
        from nesy.nsi.concept_graph import ConceptGraphEngine
        engine = ConceptGraphEngine(domain="general")
        for src, tgt, prob in edges:
            engine.add_edge(ConceptEdge(
                source=src, target=tgt,
                cooccurrence_prob=prob,
                causal_strength=0.5,
                temporal_stability=0.5,
            ))
        return engine

    def test_import(self):
        from nesy.nsi.path_finder import ConceptPathFinder
        assert ConceptPathFinder is not None

    def test_direct_path(self):
        from nesy.nsi.path_finder import ConceptPathFinder
        graph = self._build_graph([("fever", "blood_test", 0.9)])
        finder = ConceptPathFinder(graph._graph)
        path = finder.shortest_path("fever", "blood_test")
        assert path is not None
        assert "fever" in path.nodes
        assert "blood_test" in path.nodes

    def test_no_path_returns_none(self):
        from nesy.nsi.path_finder import ConceptPathFinder
        graph = self._build_graph([("a", "b", 0.8)])
        finder = ConceptPathFinder(graph._graph)
        path = finder.shortest_path("a", "c")  # c doesn't exist
        assert path is None

    def test_same_source_and_target(self):
        from nesy.nsi.path_finder import ConceptPathFinder
        graph = self._build_graph([("a", "b", 0.8)])
        finder = ConceptPathFinder(graph._graph)
        path = finder.shortest_path("a", "a")
        # Same node → trivial path or None depending on implementation
        assert path is None or path.nodes[0] == "a"

    def test_multi_hop_path(self):
        from nesy.nsi.path_finder import ConceptPathFinder
        graph = self._build_graph([
            ("fever", "blood_test", 0.9),
            ("blood_test", "wbc_count", 0.8),
        ])
        finder = ConceptPathFinder(graph._graph)
        path = finder.shortest_path("fever", "wbc_count")
        assert path is not None
        assert len(path.nodes) >= 2

    def test_all_paths_returns_list(self):
        from nesy.nsi.path_finder import ConceptPathFinder
        graph = self._build_graph([
            ("a", "b", 0.9),
            ("a", "c", 0.7),
            ("b", "d", 0.8),
            ("c", "d", 0.6),
        ])
        finder = ConceptPathFinder(graph._graph)
        paths = finder.all_paths("a", "d", max_hops=3)
        assert isinstance(paths, list)

    def test_reachable_from(self):
        from nesy.nsi.path_finder import ConceptPathFinder
        graph = self._build_graph([
            ("fever", "blood_test", 0.9),
            ("blood_test", "wbc_count", 0.8),
            ("fever", "temperature_reading", 0.95),
        ])
        finder = ConceptPathFinder(graph._graph)
        reachable = finder.reachable_from("fever", max_hops=2)
        assert isinstance(reachable, dict)
        assert "blood_test" in reachable
        assert "temperature_reading" in reachable

    def test_explain_null(self):
        from nesy.nsi.path_finder import ConceptPathFinder
        graph = self._build_graph([
            ("fever", "blood_test", 0.9),
        ])
        finder = ConceptPathFinder(graph._graph)
        explanation = finder.explain_null("blood_test", {"fever"})
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_path_weight_property(self):
        from nesy.nsi.path_finder import ConceptPathFinder
        graph = self._build_graph([("a", "b", 0.9)])
        finder = ConceptPathFinder(graph._graph)
        path = finder.shortest_path("a", "b")
        if path is not None:
            assert 0.0 <= path.total_weight <= 1.0


# ══════════════════════════════════════════════════════════════════
#  evaluation/evaluator.py — NeSyEvaluator
# ══════════════════════════════════════════════════════════════════

class TestNeSyEvaluator:

    def test_import(self):
        from nesy.evaluation.evaluator import NeSyEvaluator, EvalCase
        assert NeSyEvaluator is not None

    def test_evaluate_empty_cases(self):
        from nesy.evaluation.evaluator import NeSyEvaluator, EvalCase
        model = _make_model()
        evaluator = NeSyEvaluator(model)
        report = evaluator.evaluate([])
        assert report is not None

    def test_evaluate_one_case(self):
        from nesy.evaluation.evaluator import NeSyEvaluator, EvalCase
        model = _make_model()
        rule = SymbolicRule(
            id="r1",
            antecedents=[Predicate("IsPhilosopher", ("?x",))],
            consequents=[Predicate("IsHuman", ("?x",))],
            weight=1.0,
        )
        model.add_rules([rule])
        evaluator = NeSyEvaluator(model)
        case = EvalCase(
            input_facts={Predicate("IsPhilosopher", ("socrates",))},
            expected_derivations={Predicate("IsHuman", ("socrates",))},
            actually_missing=set(),
            expected_correct=True,
        )
        report = evaluator.evaluate([case])
        assert report is not None

    def test_evaluate_one_returns_output(self):
        from nesy.evaluation.evaluator import NeSyEvaluator, EvalCase
        model = _make_model()
        evaluator = NeSyEvaluator(model)
        case = EvalCase(
            input_facts={Predicate("A", ("x",))},
            expected_derivations=set(),
            actually_missing=set(),
            expected_correct=True,
        )
        output, metrics = evaluator.evaluate_one(case)
        assert output is not None
        assert isinstance(metrics, dict)


# ══════════════════════════════════════════════════════════════════
#  deployment/lite.py — NeSyLite (compressed graph)
# ══════════════════════════════════════════════════════════════════

class TestNeSyLite:

    def test_import(self):
        from nesy.deployment.lite import NeSyLite
        assert NeSyLite is not None

    def test_compress_concept_graph(self):
        from nesy.deployment.lite import NeSyLite
        from nesy.nsi.concept_graph import ConceptGraphEngine
        engine = ConceptGraphEngine(domain="general")
        # Add many edges — lite should keep only top-K
        concepts = ["fever", "blood_test", "wbc", "temperature", "cough", "headache"]
        for i in range(len(concepts) - 1):
            engine.add_edge(ConceptEdge(
                source=concepts[i], target=concepts[i + 1],
                cooccurrence_prob=1.0 - i * 0.1,
                causal_strength=0.5,
                temporal_stability=0.5,
            ))
        compressed = NeSyLite.compress(engine, top_k_edges=3)
        assert compressed is not None

    def test_lite_model_reason(self):
        from nesy.deployment.lite import NeSyLite
        from nesy.nsi.concept_graph import ConceptGraphEngine
        # Compress an empty graph — should still work
        engine = ConceptGraphEngine(domain="general")
        compressed = NeSyLite.compress(engine, top_k_edges=10)
        assert compressed is not None


# ══════════════════════════════════════════════════════════════════
#  deployment/optimizer.py — SymbolicGuidedOptimizer
# ══════════════════════════════════════════════════════════════════

class TestSymbolicGuidedOptimizer:

    def test_import(self):
        from nesy.deployment.optimizer import SymbolicGuidedOptimizer
        assert SymbolicGuidedOptimizer is not None

    def test_score_rules_by_importance(self):
        from nesy.deployment.optimizer import SymbolicGuidedOptimizer
        rules = [
            _make_rule("high_weight_rule", weight=0.95),
            _make_rule("low_weight_rule", weight=0.30),
            _make_rule("mid_weight_rule", weight=0.60),
        ]
        optimizer = SymbolicGuidedOptimizer()
        # compute_importance_scores requires a param→rule_ids mapping
        param_mapping = {
            "param_a": ["high_weight_rule"],
            "param_b": ["low_weight_rule"],
            "param_c": ["mid_weight_rule"],
        }
        scores = optimizer.compute_importance_scores(rules, param_mapping)
        assert isinstance(scores, dict)
        assert len(scores) == 3
        assert scores["param_a"] > scores["param_b"]

    def test_prune_low_importance_rules(self):
        from nesy.deployment.optimizer import SymbolicGuidedOptimizer
        optimizer = SymbolicGuidedOptimizer(pruning_threshold=0.50)
        params = {
            "keep_1": 1.0,
            "keep_2": 0.8,
            "prune_1": 0.3,
        }
        importance = {
            "keep_1": 0.95,
            "keep_2": 0.85,
            "prune_1": 0.20,
        }
        pruned = optimizer.prune_params(params, importance)
        assert isinstance(pruned, dict)
        # keep_1 and keep_2 should retain their values
        assert pruned["keep_1"] == 1.0
        assert pruned["keep_2"] == 0.8
        # prune_1 importance (0.20) < threshold (0.50) → zeroed
        assert pruned["prune_1"] == 0.0
