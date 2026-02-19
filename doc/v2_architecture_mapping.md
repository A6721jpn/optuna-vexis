# v2 Architecture Mapping (from v1)

## Goal

Keep the existing optimization pipeline shape while improving feasible-region coverage under strong parameter interactions.

Pipeline (unchanged high-level):

`runner -> search_space -> objective -> cad_gate/geometry/cae -> persistence/reporting`

## Mapping Strategy

1. Keep `v1` untouched and create an isolated `v2` package.
2. Copy only files needed by `runner` execution graph.
3. Implement the new exploration architecture only in `v2/search_space.py` and `v2/config.py`, then wire in `v2/runner.py`.

## File Mapping

- Entrypoint:
  - `scripts/run_v1.py` -> `scripts/run_v2.py`
  - `config/v1_0_limitations.yaml` -> `config/v2_limitations.yaml` (v2 default limits)

- Core package:
  - `src/v1/__init__.py` -> `src/v2/__init__.py`
  - `src/v1/runner.py` -> `src/v2/runner.py`
  - `src/v1/config.py` -> `src/v2/config.py`
  - `src/v1/search_space.py` -> `src/v2/search_space.py`
  - `src/v1/cad_gate.py` -> `src/v2/cad_gate.py`
  - `src/v1/objective.py` -> `src/v2/objective.py`
  - `src/v1/constraints.py` -> `src/v2/constraints.py`
  - `src/v1/types.py` -> `src/v2/types.py`
  - `src/v1/persistence.py` -> `src/v2/persistence.py`
  - `src/v1/reporting.py` -> `src/v2/reporting.py`
  - `src/v1/geometry_adapter.py` -> `src/v2/geometry_adapter.py`
  - `src/v1/freecad_worker.py` -> `src/v2/freecad_worker.py`
  - `src/v1/freecad_engine.py` -> `src/v2/freecad_engine.py`
  - `src/v1/cae_evaluator.py` -> `src/v2/cae_evaluator.py`
  - `src/v1/versioning.py` -> `src/v2/versioning.py`
  - `src/v1/README.md` -> `src/v2/README.md`

## v2 Exploration Design

### 1) Mixed candidate generation

`FeasibilityAwareSampler` in v2 will mix three proposal modes per retry:

- `global`: random full-space candidate (coverage)
- `boundary`: candidate closest to CAD gate threshold (frontier discovery)
- `local`: perturbation around archived feasible points (region refinement)

### 2) Uncertainty-aware acceptance

For candidates predicted infeasible but near gate threshold, allow probabilistic pass-through:

- input: `score` (feasible probability), `threshold`
- rule: if `abs(score - threshold) <= uncertainty_band`, accept with `uncertainty_accept_prob`

This reduces false-negative pruning around interaction boundaries.

### 3) Fixed-dimension safety

Dimensions where `low == high` are excluded from relative sampling and injected after sampling.

This prevents sampler-side key errors caused by fixed dimensions while preserving full candidate vectors for CAD gate checks.

## Non-goals (this phase)

- No change to objective semantics.
- No change to CAD gate model format.
- No change to CAE execution protocol.
