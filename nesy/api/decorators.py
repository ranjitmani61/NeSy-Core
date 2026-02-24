"""
nesy/api/decorators.py
=======================
Decorators for adding symbolic constraints to functions.

Usage:
    @nesy.symbolic_rule("HasRole(?x, doctor) → CanPrescribe(?x, ?drug)")
    def prescribe(doctor_id, drug):
        ...

    @nesy.requires_proof(confidence=0.8)
    def make_diagnosis(symptoms):
        ...
"""
from __future__ import annotations
import functools
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def symbolic_rule(rule_description: str, weight: float = 1.0):
    """Decorator: attach a symbolic rule description to a function.
    
    This is metadata — the rule is registered in the framework's
    rule registry when the decorated function is called through
    a NeSyModel.
    """
    def decorator(fn: Callable) -> Callable:
        fn._nesy_rule = rule_description
        fn._nesy_weight = weight

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        wrapper._nesy_rule = rule_description
        return wrapper
    return decorator


def requires_proof(confidence: float = 0.70, strict: bool = False):
    """Decorator: function result is only returned if confidence ≥ threshold.
    
    If confidence is below threshold:
        strict=False → returns None + logs warning
        strict=True  → raises RuntimeError
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            # Extract confidence from NSIOutput if result is one
            from nesy.core.types import NSIOutput
            if isinstance(result, NSIOutput):
                if result.confidence.minimum < confidence:
                    msg = (
                        f"@requires_proof: confidence {result.confidence.minimum:.3f} "
                        f"< required {confidence:.3f} in '{fn.__name__}'"
                    )
                    if strict:
                        raise RuntimeError(msg)
                    logger.warning(msg)
                    return None
            return result
        return wrapper
    return decorator


def domain(domain_name: str):
    """Decorator: set the domain context for a function's NeSy calls."""
    def decorator(fn: Callable) -> Callable:
        fn._nesy_domain = domain_name
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        wrapper._nesy_domain = domain_name
        return wrapper
    return decorator
