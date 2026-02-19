# Proto4 E2E Tests and Harness

## What changed

This repo now includes a proto4-codex E2E harness plus extra logging controls and
unit tests for IO and failure paths. The harness focuses on end-to-end flow with
observability and captures subprocess output to both console and log files.

## Key config files

- `config/optimizer_config.yaml`
- `config/proto4_limitations.yaml`

### New config fields

- `optimization.logging.level` (default: `INFO`)
- `optimization.logging.output_dir` (default: `output/logs`)
- `cae.stream_stdout` (default: `false`)
- `cae.stdout_log_dir` (default: `null`)
- `cae.stdout_console_level` (default: `INFO`)

## Harness

Entry point: `scripts/proto4_harness.py`

Examples:

```bash
python scripts/proto4_harness.py --generate-freecad --run-vexis --run-proto4 --max-trials 3
```

### FreeCAD fallback

If FreeCAD is not found in the conda env, the harness can fall back to a
system install by setting `FREECAD_BIN` (or `freecad_bin` in
`config/proto4_harness.yaml`). For this setup:

```bash
python scripts/proto4_harness.py --freecad-bin "C:/Program Files/FreeCAD 1.0/bin"
```

## Test data generation (FreeCAD + VEXIS)

```bash
python scripts/generate_test_data.py --conda-env fcad-codex
```

This will:
1. Use FreeCAD (in conda env `fcad-codex`) to export a STEP from `input/model.FCStd`
2. Run VEXIS once and write `input/target_curve_generated.csv`

## CI strategy

- smoke: run unit tests + small harness run with `--max-trials 3` and no VEXIS
- nightly: run harness with FreeCAD + VEXIS (long-running)

## Pass/Fail (initial)

- smoke: `pytest tests/proto4_codex -v` passes, harness runs without error
- nightly: FreeCAD export works, VEXIS returns result CSV, proto4 run completes

## Notes

The harness logs:
- `output/harness/<run_id>/logs/harness.log` (meta)
- `output/harness/<run_id>/logs/freecad_generate.log`
- `output/harness/<run_id>/logs/vexis_run.log`
- `output/harness/<run_id>/logs/proto4_run.log`
- VEXIS stdout is also written to `output/harness/<run_id>/logs/vexis/` when enabled.
