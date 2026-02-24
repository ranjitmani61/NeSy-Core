"""
nesy/symbolic/ontology/loader.py
================================
Load external ontologies (OWL, RDF, custom JSON) into SymbolicRules.

Ontologies provide rich domain knowledge — class hierarchies,
property chains, disjointness axioms — that can be automatically
converted into SymbolicRules for NeSy-Core.

Supported:
    - OWL (via owlready2 if installed)
    - RDF/RDFS (via rdflib if installed)
    - NeSy JSON (native format, no dependencies)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from nesy.core.types import Predicate, SymbolicRule

logger = logging.getLogger(__name__)


class OntologyLoader:
    """Unified entry point for ontology loading.
    
    Auto-detects format from file extension.
    Falls back gracefully if optional dependencies are missing.
    """

    @classmethod
    def load(cls, path: str) -> List[SymbolicRule]:
        """Load ontology from file. Auto-detects format."""
        p = Path(path)
        ext = p.suffix.lower()

        if ext == ".json":
            return cls._load_nesy_json(path)
        elif ext in (".owl", ".rdf", ".xml"):
            return cls._load_owl(path)
        elif ext == ".ttl":
            return cls._load_rdf(path)
        else:
            raise ValueError(f"Unsupported ontology format: {ext}")

    @classmethod
    def _load_nesy_json(cls, path: str) -> List[SymbolicRule]:
        """Load NeSy-Core native JSON ontology format."""
        data = json.loads(Path(path).read_text())
        rules = []

        # Subclass axioms: A rdfs:subClassOf B → HasType(?x,A) → HasType(?x,B)
        for item in data.get("subclasses", []):
            rules.append(SymbolicRule(
                id=f"subclass_{item['child']}_{item['parent']}",
                antecedents=[Predicate("HasType", ("?x", item["child"]))],
                consequents=[Predicate("HasType", ("?x", item["parent"]))],
                weight=1.0,
                description=f"{item['child']} is a subclass of {item['parent']}",
            ))

        # Disjointness: HasType(?x,A) ∧ HasType(?x,B) → Contradiction
        for item in data.get("disjoint", []):
            rules.append(SymbolicRule(
                id=f"disjoint_{item[0]}_{item[1]}",
                antecedents=[
                    Predicate("HasType", ("?x", item[0])),
                    Predicate("HasType", ("?x", item[1])),
                ],
                consequents=[Predicate("OntologyConflict", ("?x", f"{item[0]}_{item[1]}"))],
                weight=1.0,
                immutable=True,
                description=f"{item[0]} and {item[1]} are disjoint classes",
            ))

        # Property chains: A → B → C becomes A → C
        for item in data.get("property_chains", []):
            rules.append(SymbolicRule(
                id=f"chain_{item['id']}",
                antecedents=[Predicate(p[0], tuple(p[1:])) for p in item["chain"]],
                consequents=[Predicate(item["result"][0], tuple(item["result"][1:]))],
                weight=item.get("weight", 1.0),
                description=item.get("description", "Property chain"),
            ))

        logger.info(f"Loaded {len(rules)} rules from NeSy JSON ontology: {path}")
        return rules

    @classmethod
    def _load_owl(cls, path: str) -> List[SymbolicRule]:
        """Load OWL ontology via owlready2."""
        try:
            from nesy.symbolic.ontology.adapters.owl import OWLAdapter
            return OWLAdapter.load(path)
        except ImportError:
            logger.warning("owlready2 not installed. pip install owlready2. Returning empty rules.")
            return []

    @classmethod
    def _load_rdf(cls, path: str) -> List[SymbolicRule]:
        """Load RDF/Turtle ontology via rdflib."""
        try:
            from nesy.symbolic.ontology.adapters.rdf import RDFAdapter
            return RDFAdapter.load(path)
        except ImportError:
            logger.warning("rdflib not installed. pip install rdflib. Returning empty rules.")
            return []
