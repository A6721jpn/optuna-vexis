# Production v2

`src/v2` is the interaction-aware exploration line built from `v1`.

## Run

```bash
python scripts/run_v2.py \
  --config config/optimizer_config.yaml \
  --limits config/v2_limitations.yaml
```

## Dependencies

- CAD gate (`src/ml-prep/models/cad_gate_model/model.joblib` + `scaler.joblib`) requires:
  - `joblib`
  - `scikit-learn`
- `AUTO` sampler (OptunaHub AutoSampler) requires:
  - `optunahub`
  - `cmaes`
  - `scipy`
  - `torch` (CPU build is fine)

Install in the interpreter used to run `scripts/run_v2.py`:

```bash
python -m pip install joblib scikit-learn optunahub cmaes scipy
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## CAD Gate Retraining

`v2` can retrain CAD gate labels with the active relative-constraint logic:

```bash
python src/ml-prep/scripts/retrain_cad_gate.py \
  --config config/optimizer_config.yaml \
  --limits config/v2_limitations.yaml \
  --samples 2000 \
  --trials 120
```

## Supported Samplers

- `AUTO` (OptunaHub AutoSampler)
- `GP` (experimental)
- `TPE`
- `MOTPE` / `MO-TPE` (multi-objective TPE preset)
- `NSGAII`
- `NSGAIII`
- `RANDOM`
- `CMA-ES`

## CAD Gate Exploration (v2)

`limits.yaml` supports optional exploration settings under `cad_gate.exploration`:

- `enabled`
- `global_ratio`, `boundary_ratio`, `local_ratio`
- `boundary_candidate_pool`
- `uncertainty_band`, `uncertainty_accept_prob`
- `local_perturbation_scale`, `local_archive_size`

## Versioning

- Product line: `Production`
- Version: `2.0.0`
- Baseline: `v2`

`python scripts/run_v2.py --version` prints runtime version metadata including git revision.
