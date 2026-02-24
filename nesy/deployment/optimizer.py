"""
nesy/deployment/optimizer.py
==============================
Deployment optimizer — prunes and quantizes NeSy models for production.

Key insight: symbolic rules have explicit importance weights.
We use these weights to guide pruning:
    - High-weight rules → keep their associated neural paths
    - Low-weight rules  → candidates for pruning
    
This is smarter than magnitude-based pruning because it
uses domain knowledge (rule weights) to guide what to keep.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from nesy.core.types import SymbolicRule

logger = logging.getLogger(__name__)


class SymbolicGuidedOptimizer:
    """Prune and quantize neural components using symbolic rule importance.
    
    Symbolic importance score for parameter p:
        importance(p) = max over rules r that use p: weight(r)
    
    Parameters associated with low-weight rules are pruned first.
    """

    def __init__(
        self,
        quantization_bits: int   = 8,
        pruning_threshold: float = 0.20,
    ):
        self.quantization_bits = quantization_bits
        self.pruning_threshold = pruning_threshold

    def compute_importance_scores(
        self,
        rules: List[SymbolicRule],
        param_rule_mapping: Dict[str, List[str]],  # param_name → [rule_ids]
    ) -> Dict[str, float]:
        """Compute importance score for each parameter.
        
        importance(param) = max(weight of rules that reference this param)
        """
        rule_weights = {r.id: r.weight for r in rules}
        scores: Dict[str, float] = {}

        for param, rule_ids in param_rule_mapping.items():
            if not rule_ids:
                scores[param] = 0.0
            else:
                scores[param] = max(
                    rule_weights.get(rid, 0.0) for rid in rule_ids
                )
        return scores

    def prune_params(
        self,
        params: Dict[str, float],
        importance: Dict[str, float],
    ) -> Dict[str, float]:
        """Set low-importance parameters to zero (structured pruning)."""
        pruned = {}
        n_pruned = 0
        for name, val in params.items():
            imp = importance.get(name, 0.0)
            if imp < self.pruning_threshold:
                pruned[name] = 0.0
                n_pruned += 1
            else:
                pruned[name] = val
        logger.info(f"Pruned {n_pruned}/{len(params)} parameters (threshold={self.pruning_threshold})")
        return pruned

    def quantize(
        self,
        params: Dict[str, float],
        bits:   Optional[int] = None,
    ) -> Dict[str, float]:
        """Uniform quantization to specified bit width.
        
        Quantization formula:
            q = round( (v - min) / (max - min) × (2^bits - 1) )
            v_q = q × (max - min) / (2^bits - 1) + min
        """
        bits = bits or self.quantization_bits
        if not params:
            return params

        vals  = list(params.values())
        v_min = min(vals)
        v_max = max(vals)
        levels = (2 ** bits) - 1

        if v_max == v_min:
            return params

        quantized = {}
        for name, val in params.items():
            q = round((val - v_min) / (v_max - v_min) * levels)
            quantized[name] = q * (v_max - v_min) / levels + v_min

        logger.info(f"Quantized {len(params)} params to {bits}-bit")
        return quantized
