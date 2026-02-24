"""tests/unit/test_nsi.py — NSI null set tests"""
import pytest
from nesy.nsi.concept_graph import ConceptGraphEngine
from nesy.nsi.graph_builder import CorpusGraphBuilder, ExpertGraphBuilder, KGDerivedBuilder
from nesy.nsi.path_finder import ConceptPathFinder
from nesy.core.types import ConceptEdge, NullType, PresentSet


@pytest.fixture
def medical_graph():
    cge = ConceptGraphEngine(domain="medical")
    cge.add_edges([
        ConceptEdge("fever", "blood_test",
                    cooccurrence_prob=0.90, causal_strength=1.0, temporal_stability=1.0),
        ConceptEdge("fever", "temperature_reading",
                    cooccurrence_prob=0.95, causal_strength=1.0, temporal_stability=1.0),
        ConceptEdge("chest_pain", "ecg",
                    cooccurrence_prob=0.90, causal_strength=1.0, temporal_stability=1.0),
    ])
    cge.register_concept_class("blood_test", "diagnostic_test")
    cge.register_concept_class("temperature_reading", "vital_signs")
    return cge


class TestNullSetComputation:
    def test_fever_expects_blood_test(self, medical_graph):
        ps = PresentSet(concepts={"fever"}, context_type="medical")
        ns = medical_graph.compute_null_set(ps)
        concepts = {i.concept for i in ns.items}
        assert "blood_test" in concepts

    def test_present_concept_not_in_null_set(self, medical_graph):
        ps = PresentSet(concepts={"fever", "blood_test"}, context_type="medical")
        ns = medical_graph.compute_null_set(ps)
        concepts = {i.concept for i in ns.items}
        assert "blood_test" not in concepts

    def test_critical_class_becomes_type3(self, medical_graph):
        ps = PresentSet(concepts={"fever"}, context_type="medical")
        ns = medical_graph.compute_null_set(ps)
        type3 = [i for i in ns.items if i.null_type == NullType.TYPE3_CRITICAL]
        assert len(type3) > 0

    def test_anomaly_score_zero_for_empty_nullset(self, medical_graph):
        ps = PresentSet(concepts={"fever", "blood_test", "temperature_reading"}, context_type="medical")
        ns = medical_graph.compute_null_set(ps)
        # All expected concepts are present → no critical nulls
        assert len(ns.critical_items) == 0


class TestGraphBuilder:
    def test_expert_builder_creates_edges(self):
        builder = ExpertGraphBuilder()
        builder.add("fever", "blood_test", 0.90, causal="necessary", temporal="permanent")
        edges = builder.build_edges()
        assert len(edges) == 1
        assert edges[0].source == "fever"
        assert edges[0].causal_strength == 1.0

    def test_corpus_builder_processes_documents(self):
        builder = CorpusGraphBuilder(window_size=3, min_count=1)
        builder.add_document(["fever", "blood_test", "temperature"])
        builder.add_document(["fever", "blood_test", "elevated_wbc"])
        assert builder.vocab_size > 0
        assert builder.total_pairs > 0

    def test_kg_builder_converts_triples(self):
        builder = KGDerivedBuilder()
        builder.add_triple("fever", "causes", "elevated_wbc")
        builder.add_triple("elevated_wbc", "associated_with", "bacterial_infection")
        edges = builder.build_edges()
        assert len(edges) == 2
        # "causes" → causal_strength=1.0
        causes_edge = next(e for e in edges if e.source == "fever")
        assert causes_edge.causal_strength == 1.0


class TestPathFinder:
    def test_direct_path(self, medical_graph):
        finder = ConceptPathFinder(medical_graph._graph)
        path = finder.shortest_path("fever", "blood_test")
        assert path is not None
        assert path.hops == 1
        assert "fever" in path.nodes
        assert "blood_test" in path.nodes

    def test_no_path_returns_none(self, medical_graph):
        finder = ConceptPathFinder(medical_graph._graph)
        path = finder.shortest_path("blood_test", "fever")   # reverse — no edge
        assert path is None

    def test_reachable_from(self, medical_graph):
        finder = ConceptPathFinder(medical_graph._graph)
        reachable = finder.reachable_from("fever", max_hops=1)
        assert "blood_test" in reachable
        assert "temperature_reading" in reachable

    def test_same_source_target(self, medical_graph):
        finder = ConceptPathFinder(medical_graph._graph)
        path = finder.shortest_path("fever", "fever")
        assert path is not None
        assert path.hops == 0


class TestGraphPersistence:
    def test_save_and_load(self, medical_graph, tmp_path):
        path = str(tmp_path / "test_graph.json")
        medical_graph.save(path)
        loaded = ConceptGraphEngine.load(path)
        assert loaded.stats["edges"] == medical_graph.stats["edges"]
        assert loaded.domain == medical_graph.domain
