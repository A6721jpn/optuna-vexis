# Optuna for VEXIS (Proto2)

Automated optimization system for OGDEN material model coefficients using Optuna.
Integrates with VEXIS (CAE solver) to find parameters that minimize the RMSE against target curves (experimental data).

## LLM Context Docs (Progressive Disclosure)

For Codex sessions, start with:

- `doc/llm_progressive_disclosure/README.md`

This keeps initial context small and loads deeper details only when needed.

## Key Features

- **Coefficient Optimization**: Efficient search using Optuna (TPE/NSGA-II).
- **Auto Discretization**: Coefficient rounding to match CAE significant digits (supports `step` parameter).
- **Visualization**: Auto-plotting of optimization history, best curve comparison, and Pareto front.
- **Study Management**: Pause/Resume with database (`--resume`) and backup functionality.

## Directory Structure

```
.
├── config/                 # Configuration files
│   └── optimizer_config.yaml
├── input/                  # Input files (STEP, Target CSV)
├── output/                 # Output results (Log, Plots, DB, XML/YAML)
├── src/
│   └── proto2/            # Source code
│       ├── main.py        # Entry point
│       ├── optimizer.py   # Optuna wrapper
│       ├── visualizer.py  # Visualization module
│       └── ...
└── vexis/                  # CAE Solver (Submodule)
```

## Requirements

- Windows OS
- Python 3.11+
- Dependencies: `optuna`, `pandas`, `matplotlib`, `pyyaml`, `scipy`, etc.

## Setup

1.  **Install dependencies**:
    ```bash
    pip install -r src/proto2/requirements.txt
    ```

2.  **Prepare VEXIS submodule**:
    ```bash 
    git submodule update --init --recursive
    # Perform VEXIS side setup if necessary.
    ```

## Usage

### 1. Edit Configuration

Adjust optimization parameters in `config/optimizer_config.yaml`.

```yaml
optimization:
  max_trials: 30              # Number of trials
  discretization_step: 0.0001 # Discretization step (4 decimal places)
  objective_type: "single"    # "single" (RMSE) or "multi" (Multi-objective)
```

### 2. Run Optimization

Run the following commands from the project root.

```bash
# New Run (Backs up existing DB if present)
python -m src.proto2.main --config config/optimizer_config.yaml

# Resume Run (Continues from existing DB)
python -m src.proto2.main --config config/optimizer_config.yaml --resume
```

### 3. Check Results

Results are output to the `output/` directory.

- **`output/plots/`**:
    - `optimization_history.png`: Optimization history graph.
    - `best_result_comparison.png`: Comparison of target (dotted) and best result (red line).
- **`optimized_material.yaml`**: Optimized OGDEN coefficient definitions.
- **`summary_proto2.json`**: Execution summary.
- **`optuna_study_proto2.db`**: Study database.
