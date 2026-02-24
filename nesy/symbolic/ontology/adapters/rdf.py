"""
nesy/symbolic/ontology/adapters/rdf.py
=======================================
RDF/RDFS ontology adapter.
Requires: pip install rdflib
"""
from __future__ import annotations
import logging
from typing import List
from nesy.core.types import Predicate, SymbolicRule

logger = logging.getLogger(__name__)

RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDFS_DOMAIN   = "http://www.w3.org/2000/01/rdf-schema#domain"
RDFS_RANGE    = "http://www.w3.org/2000/01/rdf-schema#range"


class RDFAdapter:
    @classmethod
    def load(cls, path: str) -> List[SymbolicRule]:
        from rdflib import Graph, URIRef
        g = Graph()
        g.parse(path)
        rules = []

        def local(uri) -> str:
            return str(uri).split("/")[-1].split("#")[-1]

        for s, p, o in g:
            if str(p) == RDFS_SUBCLASS:
                s_name, o_name = local(s), local(o)
                rules.append(SymbolicRule(
                    id=f"rdf_sub_{s_name}_{o_name}",
                    antecedents=[Predicate("HasType", ("?x", s_name))],
                    consequents=[Predicate("HasType", ("?x", o_name))],
                    weight=1.0,
                    description=f"RDF: {s_name} subClassOf {o_name}",
                ))

        logger.info(f"Loaded {len(rules)} rules from RDF: {path}")
        return rules
