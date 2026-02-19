# Proto4 Design Document

## 1. Purpose
`proto4` integrates:
- Feasibility prediction from `cad-automation` (ML model predicting CAD breakage risk)
- Optimization flow from `optuna-for-vexis`
- CAE evaluation via submodule `vexis`

Goal: optimize only realistic geometries by filtering out CAD-breaking dimension sets before expensive CAE execution.

## 2. Scope
Included in `proto4`:
- End-to-end optimization pipeline orchestration
- CAD feasibility gate interface (ML-based)
- CAE execution orchestration interface (Vexis)
- Objective/constraint computation policy
- Logging, artifact persistence, reproducibility controls

Not included now:
- Implementation of CAD ML model training itself
- Deep changes in Vexis internals
- Production deployment infra

## 3. System Context
Input: design variable definitions, constraints, optimization config.
Flow:
1. Optuna samples a dimension set.
2. CAD feasibility gate predicts if CAD will break.
3. If infeasible: trial is pruned or penalized, no CAE run.
4. If feasible: geometry is passed to Vexis CAE.
5. CAE metrics are transformed into objective value(s).
6. Optuna updates search state.

Output: best design candidates, trial history, feasibility/CAE traceability artifacts.

## 4. Architecture (planned modules)
Under `src/proto4`:
- `runner.py`: CLI entrypoint, study lifecycle.
- `config.py`: configuration schema/validation.
- `search_space.py`: trial-to-parameter sampling logic.
- `cad_gate.py`: adapter to CAD feasibility predictor.
- `geometry_adapter.py`: CAD output to Vexis input conversion.
- `cae_evaluator.py`: Vexis execution and metric extraction.
- `constraints.py`: hard/soft constraint handling.
- `objective.py`: unified objective function flow.
- `persistence.py`: artifact and metadata persistence.
- `types.py`: shared DTOs/type definitions.

## 5. Key Interfaces (design level)
### 5.1 CAD Gate
Input: `DesignPoint`
Output: `CadFeasibilityResult`
- `is_feasible: bool`
- `confidence: float | None`
- `reason_code: str | None`
- `metadata: dict`

Behavior:
- Predict feasibility using ML model from `cad-automation`.
- Return deterministic decision for objective pipeline.

### 5.2 CAE Evaluator
Input: feasible design (or converted geometry payload)
Output: `CaeResult`
- `status: SUCCESS | FAIL`
- `metrics: dict[str, float]`
- `runtime_sec: float`
- `artifact_paths: list[str]`

### 5.3 Objective Orchestrator
Input: `optuna.trial.Trial`
Output: objective scalar or tuple
Policy:
- Infeasible CAD => prune or penalty
- Feasible CAD + CAE failure => retry policy then penalty
- Feasible CAD + CAE success => compute objective/constraints

## 6. Optimization Policy
- Default first phase: single-objective weighted score.
- Extension path: multi-objective (`NSGA-II`) with Pareto outputs.
- Pruner: aggressive for infeasible/early-bad trials.
- Sampler: TPE default, configurable.

## 7. Failure Handling Policy
- CAD infeasible prediction: no CAE execution.
- CAE transient failure: bounded retries.
- CAE terminal failure: penalty and tagged reason.
- All failures logged with trial ID and reproducible context.

## 8. Traceability & Reproducibility
Per trial, persist:
- sampled parameters
- feasibility decision and reason
- CAE inputs/outputs metadata
- objective/constraint values
- wall-clock runtime

Reproducibility requirements:
- explicit random seeds
- config snapshot per run
- deterministic path convention for artifacts

## 9. MVP Definition
Proto4 MVP is done when:
1. CAD feasibility gate is called before CAE every trial.
2. Infeasible trials never reach Vexis execution.
3. Feasible trials execute CAE and return objective values.
4. Trial-level logs/artifacts allow post-hoc diagnosis.
5. A small batch study (e.g., 20-30 trials) runs end-to-end.

## 10. Risks and Mitigations
- False negative feasibility predictions: tune threshold, add uncertainty guard band.
- False positives causing CAE waste: use stricter feasibility threshold.
- Interface drift between repos: define thin adapters and versioned contracts.
- Runtime explosion: use pruning, caching, and retry limits.

## 11. Milestones (planning)
- M1: module skeleton and config contract
- M2: CAD gate integration and decision policy
- M3: Vexis evaluation path integration
- M4: objective/constraints + persistence
- M5: smoke validation with representative study
