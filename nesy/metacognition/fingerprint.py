"""
nesy/metacognition/fingerprint.py
=================================
Reasoning Fingerprint — Revolution #8.

A **Reasoning Fingerprint** is a stable SHA-256 digest that represents
the *semantic reasoning event*, not the raw text.

It remains identical across machines if:
    - the same rules fire
    - the same derived facts are produced
    - the same null set classification is produced
    - the same config snapshot is used
    - the same unsat core IDs/explanations occur (if contradiction path)
    - the same NSIL integrity report occurs (if neural is enabled)

Different from PCAP checksum:
    PCAP checksum ← verifies the file contents weren't tampered.
    Fingerprint   ← verifies the **semantic reasoning identity**.

Canonicalization:
    ``json.dumps(obj, sort_keys=True, separators=(",", ":"))``
    then ``sha256(canonical_utf8).hexdigest()``.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from nesy.core.types import (
    ConfidenceReport,
    NSIOutput,
    NullItem,
    NullSet,
    OutputStatus,
    UnsatCore,
)
from nesy.neural.nsil import IntegrityReport


# ─────────────────────────────────────────────
#  CONFIG HASH
# ─────────────────────────────────────────────

def compute_config_hash(config_snapshot: Dict[str, Any]) -> str:
    """Stable SHA-256 hash of a config dictionary.

    The config is serialised with sorted keys and compact separators
    so the hash is identical regardless of insertion order.

    Parameters
    ----------
    config_snapshot : dict
        Arbitrary config dictionary. All values must be JSON-serialisable.

    Returns
    -------
    str
        64-char hex SHA-256 digest.
    """
    canonical = json.dumps(
        config_snapshot, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────
#  CANONICALIZATION
# ─────────────────────────────────────────────

def _round_float(value: float, decimals: int = 6) -> float:
    """Round a float to fixed decimals. Maps NaN/Inf to 0.0."""
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return round(value, decimals)


def _canonicalize_null_item(item: NullItem) -> List:
    """Canonical tuple representation of a NullItem."""
    return [
        item.null_type.name,
        item.concept,
        _round_float(item.weight),
        sorted(item.expected_because_of),
    ]


def _canonicalize_unsat_core(core: Optional[UnsatCore]) -> Optional[Dict[str, Any]]:
    """Canonical dict for an UnsatCore (or None)."""
    if core is None:
        return None
    return {
        "conflicting_rule_ids": sorted(core.conflicting_rule_ids),
        "constraint_ids": sorted(core.constraint_ids),
        "explanation": core.explanation,
    }


def _canonicalize_nsil(report: Optional[IntegrityReport]) -> Optional[Dict[str, Any]]:
    """Canonical dict for an IntegrityReport (or None)."""
    if report is None:
        return None
    items = sorted(
        [
            {
                "schema": item.schema,
                "residual": _round_float(item.residual),
            }
            for item in report.items
        ],
        key=lambda x: x["schema"],
    )
    return {
        "integrity_score": _round_float(report.integrity_score),
        "items": items,
        "is_neutral": report.is_neutral,
    }


def canonicalize_output(
    output: NSIOutput,
    config_snapshot: Optional[Dict[str, Any]] = None,
    unsat_core: Optional[UnsatCore] = None,
    nsil_report: Optional[IntegrityReport] = None,
) -> Dict[str, Any]:
    """Build the canonical payload from an ``NSIOutput``.

    All lists are sorted so that ordering differences across machines
    do not affect the fingerprint.

    Parameters
    ----------
    output : NSIOutput
        The reasoning output.
    config_snapshot : dict, optional
        Config dict at time of reasoning.
    unsat_core : UnsatCore, optional
        If contradiction occurred.
    nsil_report : IntegrityReport, optional
        NSIL integrity report (if neural enabled).

    Returns
    -------
    dict
        Canonical payload ready for ``json.dumps(sort_keys=True)``.
    """
    # rules_activated (sorted)
    rules_activated = sorted(output.reasoning_trace.rules_activated)

    # derived_predicates (sorted str representation)
    # Derived = all preds from non-initial steps
    derived_preds: List[str] = []
    for step in output.reasoning_trace.steps:
        if step.rule_applied is not None:
            for p in step.predicates:
                derived_preds.append(str(p))
    derived_preds = sorted(set(derived_preds))

    # null_items (sorted canonical tuples)
    null_items = sorted(
        [_canonicalize_null_item(item) for item in output.null_set.items],
        key=lambda x: json.dumps(x, sort_keys=True),
    )

    # confidence triple (rounded)
    confidence_triple = [
        _round_float(output.confidence.factual),
        _round_float(output.confidence.reasoning),
        _round_float(output.confidence.knowledge_boundary),
    ]

    # config hash
    config_hash = compute_config_hash(config_snapshot) if config_snapshot else ""

    payload: Dict[str, Any] = {
        "rules_activated": rules_activated,
        "derived_predicates": derived_preds,
        "null_items": null_items,
        "status": output.status.value,
        "confidence_triple": confidence_triple,
        "unsat_core": _canonicalize_unsat_core(unsat_core),
        "nsil": _canonicalize_nsil(nsil_report),
        "config_hash": config_hash,
    }

    return payload


# ─────────────────────────────────────────────
#  FINGERPRINT COMPUTATION
# ─────────────────────────────────────────────

def compute_reasoning_fingerprint(
    output: NSIOutput,
    config_snapshot: Optional[Dict[str, Any]] = None,
    unsat_core: Optional[UnsatCore] = None,
    nsil_report: Optional[IntegrityReport] = None,
) -> str:
    """Compute the SHA-256 reasoning fingerprint.

    Parameters
    ----------
    output : NSIOutput
        The reasoning output.
    config_snapshot : dict, optional
        Config dict at time of reasoning.
    unsat_core : UnsatCore, optional
        If contradiction occurred.
    nsil_report : IntegrityReport, optional
        NSIL integrity report (if neural enabled).

    Returns
    -------
    str
        64-char hex SHA-256 digest (the reasoning fingerprint).
    """
    payload = canonicalize_output(
        output=output,
        config_snapshot=config_snapshot,
        unsat_core=unsat_core,
        nsil_report=nsil_report,
    )
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
