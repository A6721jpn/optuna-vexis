r"""A/B harness to compare CAE convergence rate between two mesh engines.

Usage example (PowerShell):
  .venv\Scripts\python.exe Test\cae_convergence_ab_test.py `
    --steps-dir input/step `
    --step-glob "ab_v2_step_*.step" `
    --python-exe .venv\Scripts\python.exe `
    --case-timeout-sec 5400
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SINGULAR_WARN_RE = re.compile(
    r"singular node\s+\d+,\s+failed to assign to irregular vertex",
    flags=re.IGNORECASE,
)
SOLVER_RC_RE = re.compile(
    r"Solver Finished with Return Code\s*=\s*(-?\d+)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class EngineSpec:
    name: str
    module: str
    mesh_config_path: Path


def _resolve_under(base: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    if p.exists():
        return p.resolve()
    return (base / p).resolve()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_step_files(
    *,
    base_dir: Path,
    step_paths: list[str],
    steps_dir: str,
    step_globs: list[str],
    max_files: int | None,
) -> list[Path]:
    found: list[Path] = []
    for raw in step_paths:
        p = _resolve_under(base_dir, raw)
        if not p.exists():
            raise FileNotFoundError(f"--step not found: {p}")
        if p.is_dir():
            raise IsADirectoryError(f"--step must be file: {p}")
        found.append(p)

    if not found:
        root = _resolve_under(base_dir, steps_dir)
        if not root.exists():
            raise FileNotFoundError(f"--steps-dir not found: {root}")
        for pattern in step_globs:
            for p in sorted(root.glob(pattern)):
                if p.is_file():
                    found.append(p.resolve())

    seen: set[str] = set()
    unique: list[Path] = []
    for p in sorted(found):
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)

    if max_files is not None and max_files > 0:
        unique = unique[:max_files]
    return unique


def _decode_output(text_or_bytes: str | bytes | None) -> str:
    if text_or_bytes is None:
        return ""
    if isinstance(text_or_bytes, bytes):
        return text_or_bytes.decode("utf-8", errors="replace")
    return text_or_bytes


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _acquire_json_lock(
    lock_path: Path,
    *,
    lock_scope: str,
    details: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_id = f"{os.getpid()}-{int(time.time() * 1000)}"
    payload = {
        "lock_id": lock_id,
        "scope": lock_scope,
        "pid": os.getpid(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "details": details,
    }
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)

    # Try once, and if stale lock is found, clear and retry once more.
    for _ in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = _read_json_if_exists(lock_path)
            existing_pid = int(existing.get("pid", 0)) if isinstance(existing, dict) else 0
            if existing and not _pid_exists(existing_pid):
                try:
                    lock_path.unlink()
                    continue
                except OSError:
                    pass
            return None, existing
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload_text)
            return lock_id, None
    return None, _read_json_if_exists(lock_path)


def _release_json_lock(lock_path: Path, lock_id: str | None) -> None:
    if not lock_id:
        return
    existing = _read_json_if_exists(lock_path)
    if not existing:
        return
    if str(existing.get("lock_id")) != str(lock_id):
        return
    try:
        lock_path.unlink()
    except OSError:
        pass


def _classify_solver_failure(log_text: str, result_csv_exists: bool) -> tuple[str, int | None]:
    low = log_text.lower()
    rc_match = SOLVER_RC_RE.search(log_text)
    solver_rc = int(rc_match.group(1)) if rc_match else None

    if result_csv_exists:
        return "none", solver_rc
    if "dll not found" in low or "status_dll_not_found" in low:
        return "solver_dll_not_found", solver_rc
    if solver_rc is not None and solver_rc != 0:
        return f"solver_exit_{solver_rc}", solver_rc
    if "data file not found" in low:
        return "result_data_missing", solver_rc
    if "force_displacement.csv" in low and "failed to move" in low:
        return "result_csv_move_failed", solver_rc
    return "solver_failed", solver_rc


def _worker_run(args: argparse.Namespace) -> int:
    start_total = time.perf_counter()

    vexis_root = Path(args.vexis_root).resolve()
    step_file = Path(args.step_file).resolve()
    case_dir = Path(args.case_dir).resolve()
    result_json_path = Path(args.worker_result_json).resolve()
    mesh_python_exe = args.mesh_python_exe

    logs_dir = case_dir / "logs"
    temp_dir = case_dir / "temp"
    results_dir = case_dir / "results"
    for d in (logs_dir, temp_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    step_stem = step_file.stem
    mesh_output = temp_dir / f"{step_stem}.vtk"
    feb_path = temp_dir / f"{step_stem}.feb"
    result_csv = results_dir / f"{step_stem}_result.csv"

    mesh_log = logs_dir / "mesh.log"
    prep_log = logs_dir / "integration.log"
    solver_log = logs_dir / "solver.log"
    case_lock_path = case_dir / ".worker_active.lock"

    out: dict[str, Any] = {
        "step_file": str(step_file),
        "engine": args.engine_name,
        "module": args.engine_module,
        "mesh_config": str(Path(args.mesh_config).resolve()),
        "analysis_config": str(Path(args.analysis_config).resolve()),
        "success": False,
        "converged": False,
        "stage": "init",
        "failure_reason": None,
        "returncode": 0,
        "duration_sec": 0.0,
        "mesh_duration_sec": None,
        "prep_duration_sec": None,
        "solver_duration_sec": None,
        "mesh_returncode": None,
        "solver_returncode": None,
        "mesh_singular_warning_count": 0,
        "mesh_output_path": str(mesh_output),
        "feb_path": str(feb_path),
        "result_csv_path": str(result_csv),
        "mesh_log_path": str(mesh_log),
        "prep_log_path": str(prep_log),
        "solver_log_path": str(solver_log),
    }
    case_lock_id, case_lock_existing = _acquire_json_lock(
        case_lock_path,
        lock_scope="worker_case",
        details={
            "step_file": str(step_file),
            "engine": args.engine_name,
            "case_dir": str(case_dir),
            "worker_result_json": str(result_json_path),
        },
    )
    if case_lock_id is None:
        out["stage"] = "worker"
        out["failure_reason"] = "case_already_running"
        out["active_lock_path"] = str(case_lock_path)
        out["active_lock"] = case_lock_existing
        out["duration_sec"] = time.perf_counter() - start_total
        result_json_path.parent.mkdir(parents=True, exist_ok=True)
        result_json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    try:
        # 1) Mesh generation
        mesh_cmd = [
            mesh_python_exe,
            "-m",
            args.engine_module,
            args.mesh_config,
            str(step_file),
            "-o",
            str(mesh_output),
        ]
        mesh_t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                mesh_cmd,
                cwd=str(vexis_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=int(args.mesh_timeout_sec),
                check=False,
            )
            mesh_output_text = _decode_output(proc.stdout)
            mesh_rc = int(proc.returncode)
        except subprocess.TimeoutExpired as exc:
            mesh_output_text = _decode_output(exc.stdout)
            mesh_rc = 124
            out["stage"] = "mesh"
            out["failure_reason"] = f"mesh_timeout_{int(args.mesh_timeout_sec)}s"
        mesh_dt = time.perf_counter() - mesh_t0
        out["mesh_duration_sec"] = mesh_dt
        out["mesh_returncode"] = mesh_rc
        out["mesh_singular_warning_count"] = len(SINGULAR_WARN_RE.findall(mesh_output_text))
        mesh_log.write_text(
            "CMD: " + " ".join(shlex.quote(c) for c in mesh_cmd) + "\n\n" + mesh_output_text,
            encoding="utf-8",
        )

        if out["failure_reason"] is None:
            if mesh_rc != 0:
                out["stage"] = "mesh"
                out["failure_reason"] = f"mesh_exit_{mesh_rc}"
            elif not mesh_output.exists():
                out["stage"] = "mesh"
                out["failure_reason"] = "mesh_output_missing"

        if out["failure_reason"] is not None:
            return 0

        # 2) Integration + solver
        if str(vexis_root) not in sys.path:
            sys.path.insert(0, str(vexis_root))
        from src.config_loader import AnalysisConfig  # type: ignore
        import analysis_helpers as helpers  # type: ignore

        analysis_cfg = AnalysisConfig.from_yaml(args.analysis_config)
        template_feb = Path(analysis_cfg.template_feb)
        if not template_feb.is_absolute():
            template_feb = (vexis_root / template_feb).resolve()
        material_yaml = (vexis_root / "config" / "material.yaml").resolve()
        if not template_feb.exists():
            out["stage"] = "prep"
            out["failure_reason"] = f"template_not_found:{template_feb}"
            return 0
        if not material_yaml.exists():
            out["stage"] = "prep"
            out["failure_reason"] = f"material_yaml_not_found:{material_yaml}"
            return 0

        prep_t0 = time.perf_counter()
        try:
            helpers.run_integration(
                str(mesh_output),
                str(template_feb),
                str(feb_path),
                push_dist_override=-1.0 * abs(float(analysis_cfg.total_stroke)),
                steps=int(analysis_cfg.time_steps),
                material_name=str(analysis_cfg.material_name),
                material_config_path=str(material_yaml),
                contact_penalty=float(analysis_cfg.contact_penalty),
                log_path=str(prep_log),
            )
        except Exception as exc:  # noqa: BLE001
            out["stage"] = "prep"
            out["failure_reason"] = f"prep_failed:{exc.__class__.__name__}"
            prep_log.write_text(
                _read_text_if_exists(prep_log) + f"\n\n[worker] prep exception: {exc}\n",
                encoding="utf-8",
            )
            return 0
        out["prep_duration_sec"] = time.perf_counter() - prep_t0
        if not feb_path.exists():
            out["stage"] = "prep"
            out["failure_reason"] = "feb_missing_after_prep"
            return 0

        solver_t0 = time.perf_counter()
        try:
            solver_ok = helpers.run_solver_and_extract(
                str(feb_path),
                str(results_dir),
                log_path=str(solver_log),
                num_threads=analysis_cfg.num_threads,
                febio_exe=str(analysis_cfg.febio_path) if analysis_cfg.febio_path else None,
            )
        except Exception as exc:  # noqa: BLE001
            solver_ok = False
            out["stage"] = "solver"
            out["failure_reason"] = f"solver_exception:{exc.__class__.__name__}"
            solver_log.write_text(
                _read_text_if_exists(solver_log) + f"\n\n[worker] solver exception: {exc}\n",
                encoding="utf-8",
            )
        out["solver_duration_sec"] = time.perf_counter() - solver_t0

        result_csv_exists = result_csv.exists()
        out["converged"] = bool(solver_ok and result_csv_exists)
        if out["converged"]:
            out["stage"] = "done"
            out["success"] = True
            out["failure_reason"] = None
        else:
            out["stage"] = "solver"
            solver_text = _read_text_if_exists(solver_log)
            reason, solver_rc = _classify_solver_failure(solver_text, result_csv_exists=result_csv_exists)
            out["solver_returncode"] = solver_rc
            if out["failure_reason"] is None:
                out["failure_reason"] = reason if reason != "none" else "result_csv_missing"

    except Exception as exc:  # noqa: BLE001
        out["returncode"] = 1
        if out["stage"] == "init":
            out["stage"] = "worker"
        if out["failure_reason"] is None:
            out["failure_reason"] = f"worker_exception:{exc.__class__.__name__}"
        worker_exc = logs_dir / "worker_exception.log"
        worker_exc.write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        _release_json_lock(case_lock_path, case_lock_id)
        if not out["converged"] and out["failure_reason"] is None:
            out["failure_reason"] = "unknown_failure"
        if out["stage"] == "init" and not out["converged"]:
            out["stage"] = "worker"
        out["duration_sec"] = time.perf_counter() - start_total
        result_json_path.parent.mkdir(parents=True, exist_ok=True)
        result_json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_engine: dict[str, list[dict[str, Any]]] = {}
    by_step: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        eng = str(r.get("engine", "unknown"))
        by_engine.setdefault(eng, []).append(r)
        by_step.setdefault(str(r.get("step_file", "")), {})[eng] = r

    engines_summary: dict[str, Any] = {}
    for eng, items in by_engine.items():
        n = len(items)
        conv = sum(1 for x in items if bool(x.get("converged")))
        mesh_ok = sum(1 for x in items if str(x.get("stage")) != "mesh")
        prep_ok = sum(1 for x in items if str(x.get("stage")) not in ("mesh", "prep"))
        solver_ok = sum(1 for x in items if bool(x.get("converged")))

        total_times = [float(x.get("duration_sec", 0.0) or 0.0) for x in items]
        mesh_times = [float(x.get("mesh_duration_sec", 0.0) or 0.0) for x in items]
        prep_times = [float(x.get("prep_duration_sec", 0.0) or 0.0) for x in items if x.get("prep_duration_sec") is not None]
        solver_times = [float(x.get("solver_duration_sec", 0.0) or 0.0) for x in items if x.get("solver_duration_sec") is not None]
        singular_vals = [int(x.get("mesh_singular_warning_count", 0) or 0) for x in items]

        reasons: dict[str, int] = {}
        for x in items:
            key = str(x.get("failure_reason") or "none")
            reasons[key] = reasons.get(key, 0) + 1

        engines_summary[eng] = {
            "runs": n,
            "converged": conv,
            "convergence_rate": (conv / n) if n else 0.0,
            "mesh_success_rate": (mesh_ok / n) if n else 0.0,
            "prep_success_rate": (prep_ok / n) if n else 0.0,
            "solver_success_rate": (solver_ok / n) if n else 0.0,
            "avg_total_sec": (sum(total_times) / n) if n else 0.0,
            "avg_mesh_sec": (sum(mesh_times) / n) if n else 0.0,
            "avg_prep_sec": (sum(prep_times) / len(prep_times)) if prep_times else 0.0,
            "avg_solver_sec": (sum(solver_times) / len(solver_times)) if solver_times else 0.0,
            "avg_mesh_singular_warning_count": (sum(singular_vals) / n) if n else 0.0,
            "failure_reasons": reasons,
        }

    comparisons: list[dict[str, Any]] = []
    for step_file, pair in sorted(by_step.items()):
        b = pair.get("baseline")
        e = pair.get("experimental")
        if not b or not e:
            continue
        b_conv = bool(b.get("converged"))
        e_conv = bool(e.get("converged"))
        b_t = float(b.get("duration_sec", 0.0) or 0.0)
        e_t = float(e.get("duration_sec", 0.0) or 0.0)
        comparisons.append(
            {
                "step_file": step_file,
                "baseline_converged": b_conv,
                "experimental_converged": e_conv,
                "baseline_failure_reason": b.get("failure_reason"),
                "experimental_failure_reason": e.get("failure_reason"),
                "baseline_total_sec": b_t,
                "experimental_total_sec": e_t,
                "time_ratio_exp_over_base": (e_t / b_t) if b_t > 0.0 else None,
                "baseline_mesh_singular": b.get("mesh_singular_warning_count"),
                "experimental_mesh_singular": e.get("mesh_singular_warning_count"),
                "delta_mesh_singular_exp_minus_base": int(e.get("mesh_singular_warning_count", 0) or 0)
                - int(b.get("mesh_singular_warning_count", 0) or 0),
            }
        )

    return {
        "engines": engines_summary,
        "comparisons": comparisons,
    }


def _run_parent(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    vexis_root = _resolve_under(project_root, args.vexis_root)
    script_path = Path(__file__).resolve()
    output_root = _resolve_under(project_root, args.output_root)
    run_lock_path = output_root / ".active_run.lock"

    python_exe = args.python_exe
    if any(sep in python_exe for sep in ("/", "\\")):
        py_path = Path(python_exe)
        if not py_path.is_absolute():
            py_path = (Path.cwd() / py_path).resolve()
        if py_path.exists():
            python_exe = str(py_path)

    if not vexis_root.exists():
        print(f"[ERROR] vexis root not found: {vexis_root}")
        return 2

    run_lock_id, run_lock_existing = _acquire_json_lock(
        run_lock_path,
        lock_scope="ab_run",
        details={
            "project_root": str(project_root),
            "vexis_root": str(vexis_root),
            "python_exe": str(python_exe),
        },
    )
    if run_lock_id is None:
        print(f"[ERROR] active run lock exists: {run_lock_path}")
        if isinstance(run_lock_existing, dict):
            print(
                "[ERROR] lock owner: pid=%s created_at=%s scope=%s"
                % (
                    run_lock_existing.get("pid"),
                    run_lock_existing.get("created_at"),
                    run_lock_existing.get("scope"),
                )
            )
        print("[ERROR] previous run is still active. wait for completion before rerun.")
        return 2

    mesh_config_base = _resolve_under(vexis_root, args.mesh_config_base)
    mesh_config_exp = _resolve_under(vexis_root, args.mesh_config_exp)
    analysis_config = _resolve_under(vexis_root, args.analysis_config)
    try:
        for p in (mesh_config_base, mesh_config_exp, analysis_config):
            if not p.exists():
                print(f"[ERROR] config not found: {p}")
                return 2

        step_globs = args.step_glob if args.step_glob else ["*.step", "*.stp"]
        try:
            steps = _collect_step_files(
                base_dir=project_root,
                step_paths=args.step,
                steps_dir=args.steps_dir,
                step_globs=step_globs,
                max_files=args.max_files,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] failed to collect STEP files: {exc}")
            return 2
        if not steps:
            print("[ERROR] no STEP files matched")
            return 2

        engines = [
            EngineSpec(name="baseline", module=args.baseline_module, mesh_config_path=mesh_config_base),
            EngineSpec(name="experimental", module=args.experimental_module, mesh_config_path=mesh_config_exp),
        ]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        plan = {
            "project_root": str(project_root),
            "vexis_root": str(vexis_root),
            "python_exe": python_exe,
            "steps": [str(s) for s in steps],
            "engines": [
                {"name": e.name, "module": e.module, "mesh_config": str(e.mesh_config_path)}
                for e in engines
            ],
            "analysis_config": str(analysis_config),
            "mesh_timeout_sec": int(args.mesh_timeout_sec),
            "case_timeout_sec": int(args.case_timeout_sec),
            "dry_run": bool(args.dry_run),
        }
        (run_dir / "run_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[INFO] CAE A/B run dir: {run_dir}")
        print(f"[INFO] STEP files: {len(steps)}")
        for s in steps:
            print(f"[INFO]   - {s}")

        rows: list[dict[str, Any]] = []
        for step in steps:
            for eng in engines:
                case_dir = run_dir / "cases" / step.stem / eng.name
                result_json = case_dir / "worker_result.json"
                worker_log = case_dir / "logs" / "worker.log"
                cmd = [
                    python_exe,
                    str(script_path),
                    "--worker-run",
                    "--vexis-root",
                    str(vexis_root),
                    "--engine-name",
                    eng.name,
                    "--engine-module",
                    eng.module,
                    "--mesh-config",
                    str(eng.mesh_config_path),
                    "--analysis-config",
                    str(analysis_config),
                    "--step-file",
                    str(step),
                    "--case-dir",
                    str(case_dir),
                    "--mesh-python-exe",
                    python_exe,
                    "--mesh-timeout-sec",
                    str(int(args.mesh_timeout_sec)),
                    "--worker-result-json",
                    str(result_json),
                ]

                if args.dry_run:
                    print("[DRY-RUN]", " ".join(shlex.quote(c) for c in cmd))
                    continue

                print(f"[INFO] run start: engine={eng.name} step={step.name}")
                t0 = time.perf_counter()
                timed_out = False
                try:
                    proc = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=int(args.case_timeout_sec),
                        check=False,
                    )
                    output = _decode_output(proc.stdout)
                    rc = int(proc.returncode)
                except subprocess.TimeoutExpired as exc:
                    output = _decode_output(exc.stdout)
                    rc = 124
                    timed_out = True
                dt = time.perf_counter() - t0

                worker_log.parent.mkdir(parents=True, exist_ok=True)
                worker_log.write_text(
                    "CMD: " + " ".join(shlex.quote(c) for c in cmd) + "\n\n" + output,
                    encoding="utf-8",
                )

                if result_json.exists():
                    try:
                        row = json.loads(result_json.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        row = {}
                else:
                    row = {}

                if not isinstance(row, dict) or not row:
                    row = {
                        "step_file": str(step),
                        "engine": eng.name,
                        "module": eng.module,
                        "mesh_config": str(eng.mesh_config_path),
                        "analysis_config": str(analysis_config),
                        "success": False,
                        "converged": False,
                        "stage": "case",
                        "failure_reason": "case_timeout" if timed_out else f"worker_exit_{rc}",
                        "returncode": rc,
                        "duration_sec": dt,
                        "mesh_duration_sec": None,
                        "prep_duration_sec": None,
                        "solver_duration_sec": None,
                        "mesh_returncode": None,
                        "solver_returncode": None,
                        "mesh_singular_warning_count": 0,
                        "mesh_output_path": None,
                        "feb_path": None,
                        "result_csv_path": None,
                        "mesh_log_path": None,
                        "prep_log_path": None,
                        "solver_log_path": None,
                    }
                else:
                    row["returncode"] = rc
                    if timed_out:
                        row["success"] = False
                        row["converged"] = False
                        row["stage"] = "case"
                        row["failure_reason"] = "case_timeout"
                        row["duration_sec"] = dt
                    else:
                        row["duration_sec"] = float(row.get("duration_sec", dt) or dt)

                rows.append(row)
                print(
                    "[INFO] run end: engine=%s step=%s converged=%s stage=%s reason=%s time=%.2fs"
                    % (
                        eng.name,
                        step.name,
                        bool(row.get("converged")),
                        row.get("stage"),
                        row.get("failure_reason"),
                        float(row.get("duration_sec", dt) or dt),
                    )
                )

        if args.dry_run:
            print("[INFO] dry-run finished (no execution)")
            return 0

        _write_csv(run_dir / "ab_results.csv", rows)
        summary = _aggregate(rows)
        (run_dir / "ab_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        comp = list(summary.get("comparisons", []))
        if comp:
            _write_csv(run_dir / "ab_comparison.csv", comp)

        print(f"[INFO] result csv: {run_dir / 'ab_results.csv'}")
        print(f"[INFO] summary json: {run_dir / 'ab_summary.json'}")
        if comp:
            print(f"[INFO] comparison csv: {run_dir / 'ab_comparison.csv'}")
        return 0
    finally:
        _release_json_lock(run_lock_path, run_lock_id)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A/B CAE convergence harness for baseline vs experimental mesh engines.")

    # Parent mode
    p.add_argument("--project-root", default=".")
    p.add_argument("--vexis-root", default="vexis")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--mesh-config-base", default="config/config.yaml")
    p.add_argument("--mesh-config-exp", default="config/config_mesh_experimental.yaml")
    p.add_argument("--analysis-config", default="config/config.yaml")
    p.add_argument("--baseline-module", default="src.mesh_gen.main")
    p.add_argument("--experimental-module", default="src.mesh_gen_experimental.main")
    p.add_argument("--steps-dir", default="input/step")
    p.add_argument("--step", action="append", default=[])
    p.add_argument("--step-glob", action="append", default=[])
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--mesh-timeout-sec", type=int, default=1800)
    p.add_argument("--case-timeout-sec", type=int, default=7200)
    p.add_argument("--output-root", default="output/cae_convergence_ab")
    p.add_argument("--dry-run", action="store_true")

    # Worker mode
    p.add_argument("--worker-run", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--engine-name", default="", help=argparse.SUPPRESS)
    p.add_argument("--engine-module", default="", help=argparse.SUPPRESS)
    p.add_argument("--mesh-config", default="", help=argparse.SUPPRESS)
    p.add_argument("--step-file", default="", help=argparse.SUPPRESS)
    p.add_argument("--case-dir", default="", help=argparse.SUPPRESS)
    p.add_argument("--mesh-python-exe", default=sys.executable, help=argparse.SUPPRESS)
    p.add_argument("--worker-result-json", default="", help=argparse.SUPPRESS)
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.worker_run:
        if not args.worker_result_json:
            print("[ERROR] --worker-result-json is required in worker mode")
            return 2
        return _worker_run(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
