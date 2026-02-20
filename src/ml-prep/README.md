# ml-prep

`src/ml-prep` is the isolated workspace for CAD-gate ML data prep and retraining.

## Docs

- `src/ml-prep/COMMANDS.md`: copy-paste command list for model creation/retraining.
- `src/ml-prep/TRAINING_RUNBOOK_JA.md`: Japanese step-by-step training guide.

## Layout

- `scripts/generate_cad_gate_dataset.py`
  - Samples v2 design points and labels CAD feasibility by running FreeCAD via `v2/geometry_adapter`.
  - Uses the active v2 relative-constraint algorithm (including repair solver).
  - Supports parallel FreeCAD labeling via `--workers`.
- `scripts/train_cad_gate_automl.py`
  - Trains CAD gate classifier with Optuna-based AutoML search.
  - Exports `model.joblib` + `scaler.joblib` compatible with `src/v2/cad_gate.py`.
  - Supports parallel tuning via `--cv-n-jobs`, `--optuna-n-jobs`, `--tree-n-jobs`.
- `scripts/sweep_cad_gate_sample_sizes.py`
  - Records current model metrics to CSV, then retrains at larger sample sizes and appends.
- `scripts/active_learn_cad_gate_cycle.py`
  - Appends one active-learning batch (default +2000) to existing dataset by uncertainty sampling.
  - Optionally retrains model in the same run.
- `models/cad_gate_model/`
  - Active model directory consumed by v2.
- `archive/legacy_ai_v0/`
  - Archived legacy ai-v0 training scripts and related docs/data.

## Retrain (with relative constraints enabled)

1. Generate labels by executing current v2 geometry checks:

```bash
python src/ml-prep/scripts/generate_cad_gate_dataset.py \
  --config config/optimizer_config.yaml \
  --limits config/v2_limitations.yaml \
  --samples 2000 \
  --output src/ml-prep/data/cad_gate_dataset.csv
```

2. Train gate model (Optuna AutoML):

```bash
python src/ml-prep/scripts/train_cad_gate_automl.py \
  --dataset src/ml-prep/data/cad_gate_dataset.csv \
  --model-dir src/ml-prep/models/cad_gate_model \
  --trials 120 \
  --timeout-sec 1800
```

## Runtime dependency note

Training script requires:

- `numpy`
- `scikit-learn`
- `optuna`
- `joblib`

Install in the interpreter used for training.
