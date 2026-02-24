#!/usr/bin/env python3
"""
scripts/reason.py
==================
Run NeSy-Core reasoning from the command line.

Usage:
    python scripts/reason.py --facts "HasSymptom(p1,fever)" "HasLabResult(p1,elevated_wbc)"
                              --rules configs/medical_rules.json
                              --domain medical
"""
import argparse
import json
import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_predicate_str(s: str):
    """Parse 'Name(arg1,arg2)' or 'Name' to Predicate."""
    from nesy.core.types import Predicate
    m = re.match(r'(\w+)\(([^)]+)\)', s.strip())
    if m:
        name = m.group(1)
        args = tuple(a.strip() for a in m.group(2).split(","))
        return Predicate(name=name, args=args)
    return Predicate(name=s.strip(), args=())


def main():
    parser = argparse.ArgumentParser(description="NeSy-Core Reasoning CLI")
    parser.add_argument("--facts",   nargs="+", required=True,
                        help="Facts e.g. 'HasSymptom(p1,fever)'")
    parser.add_argument("--rules",   default=None,
                        help="Path to rules JSON")
    parser.add_argument("--graph",   default=None,
                        help="Path to concept graph JSON")
    parser.add_argument("--domain",  default="general")
    parser.add_argument("--confidence", type=float, default=0.90)
    args = parser.parse_args()

    from nesy.api.nesy_model import NeSyModel
    from nesy.symbolic.rules import RuleLoader

    model = NeSyModel(domain=args.domain)

    if args.rules:
        rules = RuleLoader.from_json(args.rules)
        model.add_rules(rules)
        print(f"Loaded {len(rules)} rules")

    if args.graph:
        model.load_concept_graph(args.graph)
        print(f"Loaded concept graph: {model.concept_graph_stats}")

    facts = {parse_predicate_str(f) for f in args.facts}
    print(f"Facts: {facts}")

    output = model.reason(
        facts=facts,
        context_type=args.domain,
        neural_confidence=args.confidence,
    )
    print(model.explain(output))


if __name__ == "__main__":
    main()
