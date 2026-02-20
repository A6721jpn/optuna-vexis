# CAD Gate ML Commands

## 1) One-shot retraining (recommended)

```bash
python src/ml-prep/scripts/retrain_cad_gate.py \
  --config config/optimizer_config.yaml \
  --limits config/v2_limitations.yaml \
  --samples 2000 \
  --trials 120 \
  --timeout-sec 1800
```

## 2) Step 1: dataset generation (with v2 relative-constraint algorithm)

```bash
python src/ml-prep/scripts/generate_cad_gate_dataset.py \
  --config config/optimizer_config.yaml \
  --limits config/v2_limitations.yaml \
  --samples 2000 \
  --sampler lhs \
  --seed 42 \
  --output src/ml-prep/data/cad_gate_dataset.csv
```

## 3) Step 2: Optuna AutoML training

```bash
python src/ml-prep/scripts/train_cad_gate_automl.py \
  --dataset src/ml-prep/data/cad_gate_dataset.csv \
  --model-dir src/ml-prep/models/cad_gate_model \
  --trials 120 \
  --timeout-sec 1800 \
  --seed 42
```

## 4) Quick smoke check (no training run)

```bash
python src/ml-prep/scripts/generate_cad_gate_dataset.py --help
python src/ml-prep/scripts/train_cad_gate_automl.py --help
python src/ml-prep/scripts/retrain_cad_gate.py --help
```

## 5) Windows venv examples

```powershell
.\.venv\Scripts\python.exe src\ml-prep\scripts\retrain_cad_gate.py --config config\optimizer_config.yaml --limits config\v2_limitations.yaml --samples 2000 --trials 120 --timeout-sec 1800
```

## 6) High-CPU retraining example (16 logical threads)

```powershell
.\.venv\Scripts\python.exe src\ml-prep\scripts\retrain_cad_gate.py --config config\optimizer_config.yaml --limits config\v2_limitations.yaml --samples 5000 --dataset-workers 6 --trials 120 --timeout-sec 5400 --cv-n-jobs 4 --optuna-n-jobs 4 --tree-n-jobs 1 --blas-threads 1
```

## 7) Sample-size sweep CSV (record current 5k -> train 10k/20k -> append)

```powershell
.\.venv\Scripts\python.exe src\ml-prep\scripts\sweep_cad_gate_sample_sizes.py --config config\optimizer_config.yaml --limits config\v2_limitations.yaml --sizes 10000,20000 --dataset-workers 6 --trials 120 --timeout-sec 5400 --cv-n-jobs 4 --optuna-n-jobs 4 --tree-n-jobs 1 --blas-threads 1 --csv-output src\ml-prep\reports\cad_gate_sample_size_sweep.csv
```

## 8) Active Learning cycle (+2000 rows per run)

```powershell
.\.venv\Scripts\python.exe src\ml-prep\scripts\active_learn_cad_gate_cycle.py --config config\optimizer_config.yaml --limits config\v2_limitations.yaml --dataset src\ml-prep\data\cad_gate_dataset.csv --add-samples 2000 --pool-size 20000 --dataset-workers 6
```

## 9) Active Learning cycle + retrain in one run

```powershell
.\.venv\Scripts\python.exe src\ml-prep\scripts\active_learn_cad_gate_cycle.py --config config\optimizer_config.yaml --limits config\v2_limitations.yaml --dataset src\ml-prep\data\cad_gate_dataset.csv --add-samples 2000 --pool-size 20000 --dataset-workers 6 --retrain --trials 120 --timeout-sec 5400 --cv-n-jobs 4 --optuna-n-jobs 4 --tree-n-jobs 1 --blas-threads 1
```
