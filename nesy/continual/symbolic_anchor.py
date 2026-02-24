"""
nesy/continual/symbolic_anchor.py
==================================
Re-export of ``SymbolicAnchor`` from ``nesy.continual.learner``.

The canonical ``SymbolicAnchor`` class lives in ``learner.py`` because it
is tightly coupled to ``ContinualLearner``.  This module exists so that
downstream code can import from the intuitively named path:

    from nesy.continual.symbolic_anchor import SymbolicAnchor

Design principle:
    Neural weights can drift during continual learning.
    Symbolic anchors are permanent and immutable.

SymbolicAnchor stores rules that must never be overwritten or removed
by any learning process — they represent verified, irrevocable domain
knowledge (e.g. "division by zero is undefined").
"""

from __future__ import annotations

from nesy.continual.learner import SymbolicAnchor  # canonical location

__all__ = ["SymbolicAnchor"]
