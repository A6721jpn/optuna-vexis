# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

FreeCAD + VEXIS (CAE solver) + Optuna optimization pipeline. Optimizes CAD sketch constraints to achieve target mechanical behavior (click ratio, peak force, etc.). Production code lives in `src/v1/`.

## Commands

```bash
# Run optimization (production)
python scripts/run_v1.py --config config/optimizer_config.yaml --limits config/v1_0_limitations.yaml

# Run all v1 tests
pytest tests/v1/

# Run a single test file
pytest tests/v1/test_constraints_domain.py

# Run a single test
pytest tests/v1/test_constraints_domain.py::test_name -v
```

No build step. No linter configured.

## Architecture

Pipeline flow:
```
runner → search_space → objective → (cad_gate + geometry_adapter + cae_evaluator) → persistence / reporting
```

**Core modules** (`src/v1/`):
- `runner.py` — CLI entrypoint, Optuna study lifecycle
- `config.py` — YAML loading, typed dataclasses (BoundsSpec, OptimizationSpec, etc.)
- `search_space.py` — Sampler creation (TPE/AUTO/NSGA-II/CMA-ES/GP), parameter discretization, constraints_func
- `objective.py` — Orchestrator: hard constraints → CAD gate → geometry → CAE → metrics
- `cad_gate.py` — ML feasibility predictor (joblib model + scaler)
- `geometry_adapter.py` / `freecad_engine.py` / `freecad_worker.py` — FreeCAD subprocess for STEP generation
- `cae_evaluator.py` — VEXIS subprocess execution, CSV result parsing, feature extraction
- `constraints.py` — Hard constraint checking, penalty computation
- `types.py` — Shared DTOs: DesignPoint, CadFeasibilityResult, CaeResult, TrialOutcome, TrialRecord
- `persistence.py` — Per-trial JSON artifacts
- `reporting.py` — Post-run Markdown report

**Config structure**: `config/optimizer_config.yaml` (shared settings) + `config/v1_0_limitations.yaml` (per-version constraints, CAD gate, CAE settings, penalty weights).

## Key Patterns

- **Constraint domains**: Config can specify `constraints_domain: "physical"` (real mm/rad) or `"ratio"` (0–1). Physical bounds are converted to ratio internally via `base_value`.
- **Feasibility violation score**: Stored in `trial.user_attrs["v1_0_feasibility_violation"]`; ≤0 = feasible, >0 = infeasible. Used by `constraints_func` to guide sampler.
- **Dimension discretization**: When enabled, dimensions with "ANGLE" in the name use `angle_step`, others use `non_angle_step`.
- **Penalty system**: `weight × (base_penalty + alpha × distance_from_bounds)` for failed trials.
- **Subprocess isolation**: FreeCAD and VEXIS both run as subprocesses to avoid ABI conflicts and enable timeout/monitoring.
- **Loading/unloading cycle**: CAE curves split at max displacement point for feature extraction.

## Code Style

- `from __future__ import annotations` in all modules
- Dataclass-based configs with optional fields and defaults
- Logging via `logging.getLogger(__name__)`
- Type hints throughout; DTOs in `types.py`

## Test Notes

Tests use a custom module loader (not a regular package install). Each test file calls `_ensure_v1_package_loaded()` to dynamically load `src/v1` as the `v1` package. No conftest.py setup needed.

## Progressive Disclosure (for LLM context efficiency)

Start from `doc/llm_progressive_disclosure/README.md`. Load `L0_project_snapshot.md` first, then only one additional layer based on task:
- Routing → `L1_task_router.md`
- Runtime/dataflow → `L2_runtime_pipeline.md`
- Cross-module deep dive → `L3_module_deep_dive.md`

Default implementation line is `src/v1`. Avoid loading `src/proto1–3`, `devlog/`, or debug files unless the task requires them.

## Workspace Policy

- Writable: this repository only (`C:\github_repo\optuna-for-vexis`)
- Read-only: any other directory under `C:\github_repo`
- UTF-8 encoding for all text files; fix display encoding before modifying Japanese content

## Dependencies

Core: optuna, pandas, numpy, scipy, pyyaml, tqdm
Optional: joblib + scikit-learn (CAD gate), optunahub + cmaes + torch (AUTO sampler)
System: FreeCAD, VEXIS (submodule), Python 3.11+, Windows
