"""
nesy/api/context.py
====================
Context managers for NeSy-Core reasoning modes.

Usage:
    with nesy.strict_mode():
        output = model.reason(facts)   # raises on critical nulls

    with nesy.domain("medical"):
        output = model.reason(facts)   # uses medical thresholds
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Generator, Optional


@contextmanager
def strict_mode(model) -> Generator:
    """Temporarily enable strict mode: Type3 nulls raise exceptions."""
    original = model._monitor.strict_mode
    model._monitor.strict_mode = True
    try:
        yield model
    finally:
        model._monitor.strict_mode = original


@contextmanager
def relaxed_mode(model, threshold: float = 0.40) -> Generator:
    """Temporarily lower the doubt threshold for exploratory reasoning."""
    original = model._monitor.doubt_threshold
    model._monitor.doubt_threshold = threshold
    try:
        yield model
    finally:
        model._monitor.doubt_threshold = original


@contextmanager
def domain_context(model, domain: str) -> Generator:
    """Temporarily change the domain context."""
    original = model.domain
    model.domain = domain
    try:
        yield model
    finally:
        model.domain = original
