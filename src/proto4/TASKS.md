# Proto4 Task List

## A. Setup and Structure
- [ ] Create `src/proto4` module layout.
- [ ] Define file ownership and boundaries for each module.
- [ ] Add `README` entry for proto4 purpose and usage scope.

## B. Configuration and Contracts
- [ ] Define optimization config schema (search space, objective, constraints, runtime).
- [ ] Define CAD gate contract (`DesignPoint` -> `CadFeasibilityResult`).
- [ ] Define CAE evaluator contract (`FeasibleInput` -> `CaeResult`).
- [ ] Define standardized reason/status codes.

## C. CAD Feasibility Gate Integration
- [ ] Connect to cad-automation ML predictor through adapter.
- [ ] Implement threshold policy for feasible/infeasible decision.
- [ ] Implement optional uncertainty guard band policy.
- [ ] Log feasibility decision with confidence and reason code.

## D. Objective Pipeline Policy
- [ ] Implement trial flow: sample -> CAD gate -> (optional) CAE -> objective.
- [ ] Decide and configure prune vs penalty behavior for infeasible trials.
- [ ] Implement CAE failure retry policy and terminal penalty policy.
- [ ] Implement constraint handling (hard reject / soft penalty).

## E. Vexis CAE Integration
- [ ] Implement geometry/data conversion adapter for Vexis input.
- [ ] Implement Vexis run wrapper and metric extraction.
- [ ] Map extracted metrics to objective function inputs.
- [ ] Add timeout/resource guardrails for CAE execution.

## F. Persistence and Observability
- [ ] Define trial artifact directory convention.
- [ ] Save config snapshot and random seed info per run.
- [ ] Persist trial-level parameter/decision/result records.
- [ ] Add run summary outputs (best trial, failure stats, feasibility rate).

## G. Validation Plan (before production)
- [ ] Unit tests for CAD gate decision logic.
- [ ] Unit tests for objective branching (prune/penalty/success).
- [ ] Integration smoke test with mocked CAD/CAE adapters.
- [ ] Limited real-path trial run (small N) with artifact review.

## H. Rollout Plan
- [ ] Prepare default baseline config for first internal run.
- [ ] Compare with existing pipeline baseline (runtime and objective quality).
- [ ] Document known limitations and follow-up backlog.

## Suggested Execution Order
1. B (contracts)
2. C (CAD gate)
3. D (objective policy)
4. E (Vexis integration)
5. F (persistence)
6. G (validation)
7. H (rollout)
