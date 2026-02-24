"""
nesy/core/config.py
===================
Global configuration for NeSy-Core.
All hyperparameters in one place — validated at startup.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SymbolicConfig:
    max_forward_chain_depth: int   = 50
    hard_rule_weight_threshold: float = 0.95
    satisfiability_max_steps: int  = 1000
    betti_warn_threshold: int      = 2    # β₀ > this → log warning


@dataclass
class NSIConfig:
    context_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "medical":  0.35,
        "legal":    0.30,
        "code":     0.25,
        "general":  0.15,
    })
    max_null_items_returned: int = 50
    type3_weight_cutoff: float   = 0.60   # above this + critical class → Type3
    type2_weight_cutoff: float   = 0.35   # above this → Type2


@dataclass
class MetaCognitionConfig:
    doubt_threshold:     float = 0.60
    reliable_threshold:  float = 0.75
    strict_mode:         bool  = False
    trace_all_steps:     bool  = True
    calibration_min_samples: int = 100

    # ── Shadow Reasoning Config ─────────────────────────────────
    shadow_enabled:              bool       = True
    shadow_critical_distance:    int        = 1
    shadow_policy:               str        = "none"     # "none" | "flag" | "reject"
    shadow_apply_domains:        List[str]  = field(default_factory=lambda: ["medical", "legal"])
    shadow_escalation_flag_prefix: str      = "SHADOW"
    shadow_max_exact_facts:      int        = 15
    shadow_max_depth:            int        = 7


@dataclass
class ContinualConfig:
    lambda_ewc:            float = 1000.0
    replay_buffer_size:    int   = 500
    consolidation_trigger: int   = 100    # consolidate every N new samples


@dataclass
class DeploymentConfig:
    quantization_bits:    int   = 8
    max_batch_size:       int   = 32
    inference_timeout_ms: int   = 200
    npu_enabled:          bool  = False
    lite_mode:            bool  = False   # reduced graph for edge


@dataclass
class NeSyConfig:
    domain:        str              = "general"
    symbolic:      SymbolicConfig   = field(default_factory=SymbolicConfig)
    nsi:           NSIConfig        = field(default_factory=NSIConfig)
    metacognition: MetaCognitionConfig = field(default_factory=MetaCognitionConfig)
    continual:     ContinualConfig  = field(default_factory=ContinualConfig)
    deployment:    DeploymentConfig = field(default_factory=DeploymentConfig)

    @classmethod
    def for_domain(cls, domain: str) -> "NeSyConfig":
        """Pre-tuned configs per domain."""
        cfg = cls(domain=domain)
        if domain == "medical":
            cfg.metacognition.doubt_threshold = 0.70    # medical needs higher confidence
            cfg.metacognition.strict_mode     = True    # hard fail on critical nulls
            cfg.metacognition.shadow_enabled  = True
            cfg.metacognition.shadow_policy   = "flag"  # downgrade to FLAGGED on CRITICAL shadow
            cfg.metacognition.shadow_critical_distance = 1
        elif domain == "legal":
            cfg.metacognition.shadow_enabled  = True
            cfg.metacognition.shadow_policy   = "flag"
            cfg.metacognition.shadow_critical_distance = 1
        elif domain == "code":
            cfg.nsi.context_thresholds["code"] = 0.20
        return cfg


# Singleton default config
DEFAULT_CONFIG = NeSyConfig()
