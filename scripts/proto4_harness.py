"""
Proto4 end-to-end harness with observability.

Runs:
  - config validation
  - optional FreeCAD data generation
  - optional VEXIS run check
  - optional proto4 optimization run

Streams subprocess output to console and captures detailed logs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RunContext:
    run_id: str
    start_time: str
    output_dir: Path
    logs_dir: Path
    artifacts_dir: Path
    run_log: Path


def _now_iso() -> str:
    return datetime.now().isoformat()


def _make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}


def _load_harness_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_yaml(path)


def _stream_subprocess(
    cmd: list[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    log_path: Optional[Path] = None,
    timeout_sec: Optional[int] = None,
) -> int:
    log_fh = None
    try:
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_path, "w", encoding="utf-8")

        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_path, "w", encoding="utf-8")
            log_fh.write("CMD: " + " ".join(cmd) + "\n")
            if cwd:
                log_fh.write("CWD: " + str(cwd) + "\n")
            if env and env.get("VEXIS_NONINTERACTIVE") is not None:
                log_fh.write("VEXIS_NONINTERACTIVE=" + env.get("VEXIS_NONINTERACTIVE", "") + "\n")
            log_fh.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        start = time.time()
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line)
            if log_fh:
                log_fh.write(line + "\n")
            if timeout_sec and (time.time() - start) > timeout_sec:
                proc.kill()
                return 124
        return proc.wait()
    finally:
        if log_fh:
            log_fh.close()


def _resolve_path(root: Path, p: str) -> Path:
    return (root / p).resolve()


def _prepare_context(run_root: Path) -> RunContext:
    run_id = _make_run_id()
    output_dir = run_root / run_id
    logs_dir = output_dir / "logs"
    artifacts_dir = output_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_log = logs_dir / "harness.log"
    return RunContext(
        run_id=run_id,
        start_time=_now_iso(),
        output_dir=output_dir,
        logs_dir=logs_dir,
        artifacts_dir=artifacts_dir,
        run_log=run_log,
    )


def _check_config(optimizer_cfg: Path, limits_cfg: Path) -> dict[str, Any]:
    from proto4_codex_alias import load_config

    cfg = load_config(optimizer_cfg, limits_cfg)
    return {
        "bounds_count": len(cfg.bounds),
        "cad_gate_enabled": cfg.cad_gate.enabled,
        "cad_gate_model_path": cfg.cad_gate.model_path,
        "vexis_path": cfg.paths.vexis_path,
        "fcstd_path": cfg.freecad.fcstd_path,
    }


def _patch_limits_for_harness(limits_cfg: Path, run_ctx: RunContext) -> Path:
    raw = _load_yaml(limits_cfg)
    raw.setdefault("cae", {})
    raw["cae"].setdefault("stream_stdout", True)
    raw["cae"].setdefault("stdout_log_dir", str(run_ctx.logs_dir / "vexis"))
    raw["cae"].setdefault("stdout_console_level", "INFO")
    raw.setdefault("cad_gate", {})
    patched = run_ctx.artifacts_dir / "limits_patched.yaml"
    patched.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return patched


def _run_freecad_generate(
    fcstd_path: Path,
    output_step: Path,
    conda_env: str,
    log_path: Path,
    freecad_bin: Optional[str] = None,
) -> int:
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda_exe:
        raise FileNotFoundError("CONDA_EXE is not set and conda is not on PATH")
    script = PROJECT_ROOT / "scripts" / "generate_step_freecad.py"
    env = os.environ.copy()
    if freecad_bin:
        env["FREECAD_BIN"] = freecad_bin
    cmd = [
        conda_exe,
        "run",
        "-n",
        conda_env,
        "python",
        str(script),
        "--fcstd",
        str(fcstd_path),
        "--output",
        str(output_step),
    ]
    return _stream_subprocess(cmd, cwd=PROJECT_ROOT, env=env, log_path=log_path, timeout_sec=900)


def _run_vexis_once(
    step_path: Path,
    vexis_root: Path,
    log_path: Path,
    *,
    isolate_input: bool,
    backup_dir: Path,
    skip_mesh: bool,
    skip_prompt: bool,
) -> int:
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

    input_dir = vexis_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    try:
        if isolate_input:
            backup_dir.mkdir(parents=True, exist_ok=True)
            for f in input_dir.glob("*.stp"):
                dest = backup_dir / f.name
                f.rename(dest)
                moved.append((dest, f))
            for f in input_dir.glob("*.step"):
                dest = backup_dir / f.name
                f.rename(dest)
                moved.append((dest, f))

        # copy STEP into VEXIS input with unique name
        job_name = f"harness_{int(time.time())}"
        target_step = input_dir / f"{job_name}.step"
        target_step.write_bytes(step_path.read_bytes())

        cmd = [python_cmd, str(vexis_root / "main.py")]
        if skip_mesh:
            cmd.append("--skip-mesh")
        env.setdefault("VEXIS_NONINTERACTIVE", "1" if skip_prompt else "0")
        rc = _stream_subprocess(cmd, cwd=vexis_root, env=env, log_path=log_path, timeout_sec=900)
        return rc
    finally:
        # restore prior input files
        for src, dest in moved:
            if dest.exists():
                dest.unlink()
            src.rename(dest)


def _run_proto4(
    optimizer_cfg: Path,
    limits_cfg: Path,
    max_trials: int,
    log_path: Path,
    freecad_bin: Optional[str] = None,
) -> int:
    run_script = PROJECT_ROOT / "scripts" / "run_proto4_codex.py"
    env = os.environ.copy()
    if freecad_bin:
        env["FREECAD_BIN"] = freecad_bin
    cmd = [
        "python",
        str(run_script),
        "--config",
        str(optimizer_cfg),
        "--limits",
        str(limits_cfg),
        "--max-trials",
        str(max_trials),
    ]
    return _stream_subprocess(cmd, cwd=PROJECT_ROOT, env=env, log_path=log_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Proto4 E2E harness")
    parser.add_argument("--harness-config", default="config/proto4_harness.yaml")
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--limits", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--generate-freecad", action="store_true", default=None)
    parser.add_argument("--run-vexis", action="store_true", default=None)
    parser.add_argument("--run-proto4", action="store_true", default=None)
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--vexis-isolate-input", action="store_true", default=None)
    parser.add_argument("--vexis-skip-mesh", action="store_true", default=None)
    parser.add_argument("--vexis-skip-prompt", action="store_true", default=None)
    parser.add_argument("--freecad-bin", default=None)
    args = parser.parse_args()

    harness_cfg_path = _resolve_path(PROJECT_ROOT, args.harness_config)
    defaults = {
        "optimizer": "config/optimizer_config.yaml",
        "limits": "config/proto4_limitations.yaml",
        "run_root": "output/harness",
        "max_trials": 5,
        "generate_freecad": False,
        "run_vexis": False,
        "run_proto4": True,
        "conda_env": "fcad",
        "vexis_isolate_input": True,
        "vexis_skip_mesh": False,
        "vexis_skip_prompt": True,
        "freecad_bin": None,
    }
    file_cfg = _load_harness_config(harness_cfg_path)
    merged = {**defaults, **file_cfg}

    if args.optimizer is not None:
        merged["optimizer"] = args.optimizer
    if args.limits is not None:
        merged["limits"] = args.limits
    if args.run_root is not None:
        merged["run_root"] = args.run_root
    if args.max_trials is not None:
        merged["max_trials"] = args.max_trials
    if args.generate_freecad is not None:
        merged["generate_freecad"] = args.generate_freecad
    if args.run_vexis is not None:
        merged["run_vexis"] = args.run_vexis
    if args.run_proto4 is not None:
        merged["run_proto4"] = args.run_proto4
    if args.conda_env is not None:
        merged["conda_env"] = args.conda_env
    if args.freecad_bin is not None:
        merged["freecad_bin"] = args.freecad_bin
    if args.vexis_isolate_input is not None:
        merged["vexis_isolate_input"] = args.vexis_isolate_input
    if args.vexis_skip_mesh is not None:
        merged["vexis_skip_mesh"] = args.vexis_skip_mesh
    if args.vexis_skip_prompt is not None:
        merged["vexis_skip_prompt"] = args.vexis_skip_prompt

    run_ctx = _prepare_context(PROJECT_ROOT / merged["run_root"])
    summary: dict[str, Any] = {
        "run_id": run_ctx.run_id,
        "start_time": run_ctx.start_time,
        "steps": [],
    }

    optimizer_cfg = _resolve_path(PROJECT_ROOT, merged["optimizer"])
    limits_cfg = _resolve_path(PROJECT_ROOT, merged["limits"])

    try:
        cfg_info = _check_config(optimizer_cfg, limits_cfg)
        summary["config"] = cfg_info
    except Exception as exc:
        summary["error"] = f"Config check failed: {exc}"
        _write_json(run_ctx.output_dir / "summary.json", summary)
        return 1

    patched_limits = _patch_limits_for_harness(limits_cfg, run_ctx)

    step_path = _resolve_path(PROJECT_ROOT, "input/step/harness_step.step")

    if merged["generate_freecad"]:
        try:
            rc = _run_freecad_generate(
                fcstd_path=_resolve_path(PROJECT_ROOT, cfg_info["fcstd_path"]),
                output_step=step_path,
                conda_env=merged["conda_env"],
                log_path=run_ctx.logs_dir / "freecad_generate.log",
                freecad_bin=merged["freecad_bin"],
            )
        except Exception as exc:
            rc = 1
            summary["error"] = f"FreeCAD generation failed: {exc}"
        summary["steps"].append({"freecad_generate": rc})
        if rc != 0:
            _write_json(run_ctx.output_dir / "summary.json", summary)
            return rc

    if merged["run_vexis"]:
        rc = _run_vexis_once(
            step_path=step_path,
            vexis_root=_resolve_path(PROJECT_ROOT, cfg_info["vexis_path"]),
            log_path=run_ctx.logs_dir / "vexis_run.log",
            isolate_input=merged["vexis_isolate_input"],
            backup_dir=run_ctx.artifacts_dir / "vexis_input_backup",
            skip_mesh=merged["vexis_skip_mesh"],
            skip_prompt=merged["vexis_skip_prompt"],
        )
        summary["steps"].append({"vexis_run": rc})
        if rc != 0:
            summary["error"] = "VEXIS run failed"
            _write_json(run_ctx.output_dir / "summary.json", summary)
            return rc

    if merged["run_proto4"]:
        rc = _run_proto4(
            optimizer_cfg=optimizer_cfg,
            limits_cfg=patched_limits,
            max_trials=merged["max_trials"],
            log_path=run_ctx.logs_dir / "proto4_run.log",
            freecad_bin=merged["freecad_bin"],
        )
        summary["steps"].append({"proto4_run": rc})
        if rc != 0:
            summary["error"] = "Proto4 run failed"
            _write_json(run_ctx.output_dir / "summary.json", summary)
            return rc

    summary["end_time"] = _now_iso()
    _write_json(run_ctx.output_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
