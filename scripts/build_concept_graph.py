#!/usr/bin/env python3
"""
scripts/build_graph.py
=======================
Build a concept graph from a text corpus or expert definitions.

Usage:
    # Corpus-based
    python scripts/build_graph.py --mode corpus --input data/medical_notes.txt --output graphs/medical.json

    # Expert definitions (YAML/JSON)  
    python scripts/build_graph.py --mode expert --input configs/medical_edges.json --output graphs/medical.json
"""
import argparse
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def build_from_corpus(input_path: str, output_path: str, window: int, min_count: int):
    from nesy.nsi.graph_builder import CorpusGraphBuilder
    from nesy.nsi.concept_graph import ConceptGraphEngine

    builder = CorpusGraphBuilder(window_size=window, min_count=min_count)

    with open(input_path) as f:
        for line in f:
            tokens = line.strip().lower().split()
            if tokens:
                builder.add_document(tokens)

    print(f"Processed documents. Vocab: {builder.vocab_size} terms, Pairs: {builder.total_pairs}")

    edges = builder.build_edges()
    cge   = ConceptGraphEngine()
    cge.add_edges(edges)
    cge.save(output_path)

    print(f"Saved concept graph: {cge.stats}")


def build_from_expert(input_path: str, output_path: str):
    from nesy.nsi.graph_builder import ExpertGraphBuilder
    from nesy.nsi.concept_graph import ConceptGraphEngine

    data = json.load(open(input_path))
    builder = ExpertGraphBuilder()

    for edge_spec in data["edges"]:
        builder.add(
            source=edge_spec["source"],
            target=edge_spec["target"],
            conditional_prob=edge_spec["prob"],
            causal=edge_spec.get("causal", "associated"),
            temporal=edge_spec.get("temporal", "stable"),
        )

    edges = builder.build_edges()
    cge   = ConceptGraphEngine(domain=data.get("domain", "general"))
    cge.add_edges(edges)
    cge.save(output_path)
    print(f"Saved expert graph: {cge.stats}")


def main():
    parser = argparse.ArgumentParser(description="Build NeSy-Core Concept Graph")
    parser.add_argument("--mode",      choices=["corpus", "expert"], required=True)
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--window",    type=int, default=5)
    parser.add_argument("--min-count", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "corpus":
        build_from_corpus(args.input, args.output, args.window, args.min_count)
    else:
        build_from_expert(args.input, args.output)


if __name__ == "__main__":
    main()
