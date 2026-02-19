# Production v1.0

`src/v1` is the production package for the first release line.

## Run

```bash
python scripts/run_v1.py \
  --config config/optimizer_config.yaml \
  --limits config/v1_0_limitations.yaml
```

## Dependencies

- CAD gate (`input/cad_gate_model/model.joblib` + `scaler.joblib`) requires:
  - `joblib`
  - `scikit-learn`
- `AUTO` sampler (OptunaHub AutoSampler) requires:
  - `optunahub`
  - `cmaes`
  - `scipy`
  - `torch` (CPU build is fine)

Install in the interpreter used to run `scripts/run_v1.py`:

```bash
python -m pip install joblib scikit-learn optunahub cmaes scipy
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
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

## Versioning

- Product line: `Production`
- Version: `1.0.0`
- Baseline: `v1.0`

`python scripts/run_v1.py --version` prints runtime version metadata including git revision.
