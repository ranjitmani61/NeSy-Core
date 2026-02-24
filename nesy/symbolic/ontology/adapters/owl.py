"""
nesy/symbolic/ontology/adapters/owl.py
=======================================
OWL ontology adapter — converts OWL axioms to SymbolicRules.
Requires: pip install owlready2
"""
from __future__ import annotations
import logging
from typing import List
from nesy.core.types import Predicate, SymbolicRule

logger = logging.getLogger(__name__)


class OWLAdapter:
    @classmethod
    def load(cls, path: str) -> List[SymbolicRule]:
        import owlready2
        onto = owlready2.get_ontology(f"file://{path}").load()
        rules = []

        # Extract subclass axioms
        for cls_obj in onto.classes():
            for parent in cls_obj.is_a:
                if hasattr(parent, "name"):
                    rules.append(SymbolicRule(
                        id=f"owl_sub_{cls_obj.name}_{parent.name}",
                        antecedents=[Predicate("HasType", ("?x", cls_obj.name))],
                        consequents=[Predicate("HasType", ("?x", parent.name))],
                        weight=1.0,
                        description=f"OWL: {cls_obj.name} subClassOf {parent.name}",
                    ))

        logger.info(f"Loaded {len(rules)} rules from OWL: {path}")
        return rules
