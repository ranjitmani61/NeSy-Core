"""
nesy/core/registry.py
=====================
Plugin registry — allows third parties to register custom components
(rules, ontologies, concept graph builders, backbones) without
modifying core framework code.

Pattern: Registry.register("name", ComponentClass)
         Registry.get("name") → ComponentClass instance
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Type
import logging

logger = logging.getLogger(__name__)


class Registry:
    """Generic component registry with validation.
    
    Usage:
        # Register a custom backbone
        Registry.register("my_backbone", MyBackboneClass, category="backbone")
        
        # Retrieve it
        cls = Registry.get("my_backbone", category="backbone")
        instance = cls(**kwargs)
    """
    _store: Dict[str, Dict[str, Any]] = {}    # category → {name → class}

    @classmethod
    def register(
        cls,
        name:     str,
        component: Any,
        category: str = "default",
        override: bool = False,
    ) -> None:
        if category not in cls._store:
            cls._store[category] = {}
        if name in cls._store[category] and not override:
            raise KeyError(
                f"Component '{name}' already registered in category '{category}'. "
                "Use override=True to replace."
            )
        cls._store[category][name] = component
        logger.debug(f"Registered [{category}] '{name}'")

    @classmethod
    def get(cls, name: str, category: str = "default") -> Any:
        try:
            return cls._store[category][name]
        except KeyError:
            available = list(cls._store.get(category, {}).keys())
            raise KeyError(
                f"Component '{name}' not found in category '{category}'. "
                f"Available: {available}"
            )

    @classmethod
    def list_all(cls, category: Optional[str] = None) -> Dict:
        if category:
            return dict(cls._store.get(category, {}))
        return {cat: list(items.keys()) for cat, items in cls._store.items()}

    @classmethod
    def decorator(cls, name: str, category: str = "default"):
        """Use as decorator: @Registry.decorator('my_component')"""
        def _register(component):
            cls.register(name, component, category=category)
            return component
        return _register
