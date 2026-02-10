"""Generate a small target curve using VEXIS output.

Runs VEXIS once with an input STEP and writes an adjusted target CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import subprocess
import os
import sys

import pandas as pd


def _stream(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line)
            fh.write(line + "\n")
        return proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vexis-root", default="vexis")
    parser.add_argument("--step", required=True)
    parser.add_argument("--out", default="input/target_curve_generated.csv")
    parser.add_argument("--log", default="output/logs/vexis_generate_target.log")
    args = parser.parse_args()

    vexis_root = Path(args.vexis_root).resolve()
    step_path = Path(args.step).resolve()
    out_path = Path(args.out).resolve()
    log_path = Path(args.log).resolve()

    python_cmd = sys.executable
    venv_py = vexis_root.parent / ".venv" / "Scripts" / "python.exe"
    if not venv_py.exists():
        venv_py = vexis_root.parent / ".venv" / "bin" / "python"
    if venv_py.exists():
        python_cmd = str(venv_py)

    env = os.environ.copy()
    env.update({
        "QT_QPA_PLATFORM": "offscreen",
        "DISPLAY": "",
        "VTK_DEFAULT_OPENGL_WINDOW": "vtkOSOpenGLRenderWindow",
        "PYVISTA_OFF_SCREEN": "true",
    })

    job_name = f"gen_target_{int(time.time())}"
    target_step = vexis_root / "input" / f"{job_name}.step"
    target_step.parent.mkdir(parents=True, exist_ok=True)
    target_step.write_bytes(step_path.read_bytes())

    cmd = [python_cmd, str(vexis_root / "main.py")]
    rc = _stream(cmd, vexis_root, log_path)
    if rc != 0:
        return rc

    result_csv = vexis_root / "results" / f"{job_name}_result.csv"
    if not result_csv.exists():
        return 2

    df = pd.read_csv(result_csv)
    # normalize to displacement/force columns if needed
    cols = [c.lower() for c in df.columns]
    disp = None
    force = None
    for cand in ("stroke", "displacement"):
        if cand in cols:
            disp = df.columns[cols.index(cand)]
            break
    for cand in ("reaction_force", "force"):
        if cand in cols:
            force = df.columns[cols.index(cand)]
            break

    if disp is None or force is None:
        disp = df.columns[0]
        force = df.columns[1]

    out_df = pd.DataFrame({"displacement": df[disp], "force": df[force]})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
