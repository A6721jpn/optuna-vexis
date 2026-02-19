"""Generate a minimal STEP + target curve dataset using FreeCAD + VEXIS."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def _run(cmd: list[str]) -> int:
    print(" ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda-env", default="fcad-codex")
    parser.add_argument("--fcstd", default="input/model.FCStd")
    parser.add_argument("--step", default="input/step/harness_step.step")
    parser.add_argument("--target", default="input/target_curve_generated.csv")
    args = parser.parse_args()

    step_cmd = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        "scripts/generate_step_freecad.py",
        "--fcstd",
        args.fcstd,
        "--output",
        args.step,
    ]
    rc = _run(step_cmd)
    if rc != 0:
        return rc

    target_cmd = [
        "python",
        "scripts/generate_target_curve.py",
        "--step",
        args.step,
        "--out",
        args.target,
    ]
    return _run(target_cmd)


if __name__ == "__main__":
    raise SystemExit(main())
