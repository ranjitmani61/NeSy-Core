"""
nesy/nsi/graph_builder.py
==========================
ConceptGraphBuilder — automatically construct the concept graph
from co-occurrence data, expert rules, or knowledge base triples.

Three construction strategies:

1. CORPUS-BASED (Statistical):
   Count co-occurrence P(b|a) across documents.
   causal_strength = 0.5 (statistical, not confirmed causal)
   temporal_stability = 0.5 (assumed stable, unvalidated)
   
   This requires no domain expertise but may include spurious edges.

2. EXPERT-DEFINED (Curated):
   Domain experts specify edges with full W(a→b) parameters.
   High quality. The recommended approach for medical/legal domains.

3. KG-DERIVED (Knowledge Graph):
   Import from Wikidata, UMLS (medical), Black's Law (legal) etc.
   Extract subgraph edges, compute weights from relation types.

All three strategies produce ConceptEdge objects that are loaded
into ConceptGraphEngine via add_edges().

Mathematical basis:
    Co-occurrence probability:
        P(b|a) = count(a,b) / count(a)
        
    Pointwise Mutual Information (PMI) filter:
        PMI(a,b) = log P(a,b) / (P(a) × P(b))
        Only keep edges where PMI > 0 (positive association)
    
    Normalised PMI → [0,1] for use as cooccurrence_prob:
        NPMI(a,b) = PMI(a,b) / (-log P(a,b))
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from nesy.core.types import ConceptEdge

logger = logging.getLogger(__name__)


class CorpusGraphBuilder:
    """Build concept graph from text corpus via co-occurrence statistics.
    
    Usage:
        builder = CorpusGraphBuilder(window_size=5, min_count=10, npmi_threshold=0.1)
        
        for document in corpus:
            builder.add_document(tokenize(document))
        
        edges = builder.build_edges(causal_strength=0.5, temporal_stability=0.5)
        cge.add_edges(edges)
    """

    def __init__(
        self,
        window_size:    int   = 5,      # co-occurrence window
        min_count:      int   = 10,     # minimum pair count to include
        npmi_threshold: float = 0.10,   # minimum NPMI to create edge
        max_edges:      int   = 10000,  # cap total edges
    ):
        self.window_size    = window_size
        self.min_count      = min_count
        self.npmi_threshold = npmi_threshold
        self.max_edges      = max_edges

        self._unigram:   Counter = Counter()    # concept → count
        self._bigram:    Counter = Counter()    # (a, b) → count
        self._total_docs = 0

    def add_document(self, tokens: List[str]) -> None:
        """Process one document (list of concept tokens)."""
        self._total_docs += 1
        seen = set()

        for i, token in enumerate(tokens):
            self._unigram[token] += 1
            seen.add(token)
            # Co-occurrence within window
            for j in range(i + 1, min(i + self.window_size + 1, len(tokens))):
                if tokens[j] != token:
                    self._bigram[(token, tokens[j])] += 1
                    self._bigram[(tokens[j], token)] += 1

    def add_documents_batch(self, documents: List[List[str]]) -> None:
        for doc in documents:
            self.add_document(doc)

    def build_edges(
        self,
        causal_strength:    float = 0.5,
        temporal_stability: float = 0.5,
    ) -> List[ConceptEdge]:
        """Build ConceptEdges from accumulated co-occurrence statistics.
        
        Returns edges sorted by NPMI descending.
        """
        total_counts = sum(self._unigram.values())
        if total_counts == 0:
            return []

        edges: List[Tuple[float, ConceptEdge]] = []

        for (a, b), count_ab in self._bigram.items():
            if count_ab < self.min_count:
                continue

            count_a = self._unigram[a]
            count_b = self._unigram[b]

            # Probabilities
            p_a  = count_a  / total_counts
            p_b  = count_b  / total_counts
            p_ab = count_ab / total_counts

            if p_a <= 0 or p_b <= 0 or p_ab <= 0:
                continue

            # NPMI (Normalised Pointwise Mutual Information)
            pmi  = math.log(p_ab / (p_a * p_b))
            npmi = pmi / (-math.log(p_ab))   # ∈ [-1, 1]

            if npmi < self.npmi_threshold:
                continue

            # P(b|a) as co-occurrence probability
            cooccurrence_prob = min(1.0, count_ab / count_a)

            # Clamp causal_strength and temporal_stability to valid discrete values
            cs = _nearest_valid(causal_strength)
            ts = _nearest_valid(temporal_stability)

            try:
                edge = ConceptEdge(
                    source=a,
                    target=b,
                    cooccurrence_prob=cooccurrence_prob,
                    causal_strength=cs,
                    temporal_stability=ts,
                )
                edges.append((npmi, edge))
            except AssertionError:
                pass

        # Sort by NPMI and return top max_edges
        edges.sort(key=lambda x: -x[0])
        result = [e for _, e in edges[:self.max_edges]]
        logger.info(
            f"CorpusGraphBuilder: built {len(result)} edges from "
            f"{len(self._bigram)//2} unique pairs "
            f"({self._total_docs} documents)"
        )
        return result

    @property
    def vocab_size(self) -> int:
        return len(self._unigram)

    @property
    def total_pairs(self) -> int:
        return len(self._bigram) // 2


class ExpertGraphBuilder:
    """Build concept graph from expert-defined edge specifications.
    
    The recommended approach for high-stakes domains (medical, legal).
    Experts define edges with explicit causal and temporal parameters.
    
    Usage:
        builder = ExpertGraphBuilder(domain="medical")
        
        builder.add(
            source="fever",
            target="blood_test",
            conditional_prob=0.90,
            causal="necessary",    # → 1.0
            temporal="permanent",  # → 1.0
            notes="Fever always warrants CBC panel",
        )
        
        edges = builder.build_edges()
    """

    CAUSAL_MAP   = {"necessary": 1.0, "associated": 0.5, "weak": 0.1}
    TEMPORAL_MAP = {"permanent": 1.0, "stable": 0.5, "transient": 0.1}

    def __init__(self, domain: str = "general"):
        self.domain = domain
        self._specs: List[Dict] = []

    def add(
        self,
        source:           str,
        target:           str,
        conditional_prob: float,
        causal:           str = "associated",   # "necessary" | "associated" | "weak"
        temporal:         str = "stable",       # "permanent" | "stable" | "transient"
        notes:            str = "",
    ) -> "ExpertGraphBuilder":
        """Add an expert-defined edge. Chainable."""
        self._specs.append({
            "source":           source,
            "target":           target,
            "conditional_prob": conditional_prob,
            "causal":           causal,
            "temporal":         temporal,
            "notes":            notes,
        })
        return self

    def build_edges(self) -> List[ConceptEdge]:
        """Build ConceptEdge objects from all defined specs."""
        edges = []
        for spec in self._specs:
            cs = self.CAUSAL_MAP.get(spec["causal"], 0.5)
            ts = self.TEMPORAL_MAP.get(spec["temporal"], 0.5)
            edge = ConceptEdge(
                source=spec["source"],
                target=spec["target"],
                cooccurrence_prob=spec["conditional_prob"],
                causal_strength=cs,
                temporal_stability=ts,
            )
            edges.append(edge)
        logger.info(f"ExpertGraphBuilder: built {len(edges)} expert-defined edges")
        return edges


class KGDerivedBuilder:
    """Build concept graph from external knowledge graph triples.
    
    Converts (subject, predicate, object) triples to ConceptEdges.
    Weight estimation from relation type:
        "causes"          → causal=1.0, temporal=1.0
        "associated_with" → causal=0.5, temporal=0.5
        "may_cause"       → causal=0.5, temporal=0.5
        "mentioned_with"  → causal=0.1, temporal=0.1
    """

    RELATION_WEIGHTS: Dict[str, Tuple[float, float]] = {
        "causes":           (1.0, 1.0),
        "directly_causes":  (1.0, 1.0),
        "associated_with":  (0.5, 0.5),
        "related_to":       (0.5, 0.5),
        "may_cause":        (0.5, 0.5),
        "co_occurs_with":   (0.1, 0.5),
        "mentioned_with":   (0.1, 0.1),
    }

    def __init__(self, default_prob: float = 0.60):
        self.default_prob = default_prob
        self._triples: List[Tuple[str, str, str]] = []

    def add_triple(self, subject: str, relation: str, obj: str) -> None:
        self._triples.append((subject, relation, obj))

    def add_triples(self, triples: List[Tuple[str, str, str]]) -> None:
        for t in triples:
            self.add_triple(*t)

    def build_edges(self) -> List[ConceptEdge]:
        edges = []
        for subject, relation, obj in self._triples:
            cs, ts = self.RELATION_WEIGHTS.get(
                relation.lower().replace(" ", "_"),
                (0.1, 0.5)
            )
            try:
                edge = ConceptEdge(
                    source=subject,
                    target=obj,
                    cooccurrence_prob=self.default_prob,
                    causal_strength=cs,
                    temporal_stability=ts,
                )
                edges.append(edge)
            except AssertionError:
                logger.debug(f"Skipped invalid edge: {subject} → {obj}")
        logger.info(f"KGDerivedBuilder: built {len(edges)} edges from {len(self._triples)} triples")
        return edges


def _nearest_valid(value: float) -> float:
    """Snap a float to the nearest valid discrete value: {0.1, 0.5, 1.0}"""
    valid = [0.1, 0.5, 1.0]
    return min(valid, key=lambda v: abs(v - value))
