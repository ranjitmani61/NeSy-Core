# Changelog

All notable changes to NeSy-Core are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [PEP 440](https://peps.python.org/pep-0440/).

---

## [0.1.1] — 2026-02-24

### Added

- **Shadow Reasoning integration into MetaCognitionMonitor.**
  Every call to `MetaCognitionMonitor.evaluate()` now automatically computes
  counterfactual shadow distances for all derived conclusions when
  `shadow_enabled=True` (the default).
- New `MetaCognitionConfig` fields:
  `shadow_enabled`, `shadow_critical_distance`, `shadow_policy`,
  `shadow_apply_domains`, `shadow_escalation_flag_prefix`,
  `shadow_max_exact_facts`, `shadow_max_depth`.
- `NeSyConfig.for_domain("medical")` and `for_domain("legal")` now default
  `shadow_policy="flag"`, automatically flagging CRITICAL-distance conclusions.
- `NeSyModel` accepts `shadow_enabled`, `shadow_policy`,
  `shadow_critical_distance`, and `shadow_apply_domains` constructor arguments.
  Medical and legal domains auto-configure `shadow_policy="flag"`.
- `NeSyModel.reason()` passes `input_facts`, `derived_facts`, and `rules`
  to the monitor so shadow analysis runs end-to-end with zero extra code.
- YAML config files (`configs/medical.yaml`, `configs/legal.yaml`,
  `configs/default.yaml`) include shadow configuration blocks.
- `shadow_demo.py` added to the Makefile `examples` target and CI workflow.
- 21 new tests in `tests/unit/test_shadow_integration.py` covering:
  flag policy, reject policy, none policy, non-derivable conclusions,
  robust conclusions, domain scoping, disabled shadow, config presets,
  end-to-end via `NeSyModel.reason()`, confidence regression, custom
  thresholds, and policy validation.

### Changed

- `MetaCognitionMonitor.evaluate()` signature extended with optional
  `input_facts`, `derived_facts`, and `rules` parameters (backward-compatible).
- `MetaCognitionMonitor.__init__()` accepts shadow configuration parameters
  (all have safe defaults; existing callers require no changes).

### Fixed

- Nothing; this is a feature release.

---

## [0.1.0] — 2026-02-22

### Added

- Initial public release of NeSy-Core.
- Symbolic Engine: FOL forward-chaining, Robinson resolution, Betti topology,
  unsat core explanation.
- Negative Space Intelligence (NSI): null set computation N(X) = E(X) - P(X),
  Type 1/2/3 null classification.
- 3D Meta-Cognition: factual, reasoning, and knowledge-boundary confidence.
  Self-doubt mechanism holds output below configurable threshold.
- Continual Learning: EWC weight protection + symbolic anchors.
  No catastrophic forgetting across sequential tasks.
- Counterfactual Shadow Reasoning module (`nesy.metacognition.shadow`):
  computes minimum fact-cut distance for every derived conclusion.
- NSIL (Neural-Symbolic Integrity Link): auditable integrity score between
  neural grounding evidence and symbolic constraint expectations.
- Reasoning Fingerprint: deterministic SHA-256 digest of the full reasoning
  event for reproducibility and audit.
- Proof Capsule (PCAP): self-contained export of reasoning trace,
  confidence report, and fingerprint.
- Unsat Core Explanation: minimal conflicting rule sets with human-readable
  explanations when symbolic rules contradict.
- Developer API: `NeSyModel`, `Pipeline`, decorators, context managers.
- Deployment: ONNX export, pruning, quantization, NPU offload stubs.
- FastAPI inference server (`nesy.deployment.server`).
- Examples: `basic_reasoning.py`, `medical_diagnosis.py`,
  `continual_learning.py`, `edge_deployment.py`, `shadow_demo.py`.
- CI: GitHub Actions on 3 OS × 3 Python versions, benchmarks, release-on-tag.
- 964 tests passing (unit + integration + benchmarks).
