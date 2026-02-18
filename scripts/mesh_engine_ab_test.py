"""A/B harness for baseline vs experimental mesh engines.

Requested workflow:
  1) Run CAD gate + FreeCAD to generate STEP files (default: 10) into `input/step`.
  2) Run A/B mesh generation on the same STEP set:
       - baseline:     python -m src.mesh_gen.main
       - experimental: python -m src.mesh_gen_experimental.main
  3) Collect runtime / success / singular-node location summary.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SINGULAR_WARN_RE = re.compile(
    r"singular node\s+\d+,\s+failed to assign to irregular vertex",
    flags=re.IGNORECASE,
)
SINGULAR_DETAIL_RE = re.compile(
    r"\[mesh-singular\]\s*face=(\d+)\s*hits=(\d+)\s*radial=\[([^\]]+)\]\s*axial=\[([^\]]+)\]",
    flags=re.IGNORECASE,
)
SINGULAR_NODE_RE = re.compile(
    r"\[mesh-singular-node\]\s*face=(\d+)\s*node=(\d+)\s*hits=(\d+)",
    flags=re.IGNORECASE,
)
SINGULAR_TOTAL_RE = re.compile(
    r"\[mesh-singular\]\s*total_hits=(\d+)\s*unique_faces=(\d+)\s*unique_face_nodes=(\d+)",
    flags=re.IGNORECASE,
)
SUMMARY_WARN_RE = re.compile(r"\b(\d+)\s+warnings\b", flags=re.IGNORECASE)
FINAL_MESH_RE = re.compile(r"Final mesh:\s*nodes=(\d+),\s*elements=(\d+)", flags=re.IGNORECASE)
SELECTED_STRATEGY_RE = re.compile(r"selected strategy=([A-Za-z0-9_]+)", flags=re.IGNORECASE)
INVFIX_RE = re.compile(
    r"\[invfix\]\s*([A-Za-z0-9_]+)\s*:\s*fixed\s+(\d+)\s+inverted",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class EngineSpec:
    name: str
    module: str
    config_path: Path


@dataclass
class RunResult:
    step_file: str
    engine: str
    module: str
    command: str
    returncode: int
    success: bool
    duration_sec: float
    singular_warning_count: int
    singular_total_hits_reported: int
    singular_unique_faces_reported: int
    singular_unique_face_nodes_reported: int
    singular_faces_json: str
    singular_nodes_json: str
    summary_warning_count: int
    warning_line_count: int
    final_nodes: int | None
    final_elements: int | None
    invfix_ring: int | None
    invfix_core: int | None
    invfix_final: int | None
    selected_strategy: str | None
    output_mesh: str
    log_path: str
    failure_reason: str | None


def _ensure_v2_importable(project_root: Path) -> None:
    """Bootstrap `v2` package exactly like scripts/run_v2.py does."""
    if "v2" in sys.modules:
        return

    pkg_dir = project_root / "src" / "v2"
    init_py = pkg_dir / "__init__.py"
    if not init_py.exists():
        raise RuntimeError(f"v2 package init not found: {init_py}")

    spec = importlib.util.spec_from_file_location(
        "v2",
        str(init_py),
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create import spec for v2 package")

    mod = importlib.util.module_from_spec(spec)
    sys.modules["v2"] = mod
    spec.loader.exec_module(mod)

    for py_file in pkg_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        sub_name = f"v2.{py_file.stem}"
        if sub_name in sys.modules:
            continue
        sub_spec = importlib.util.spec_from_file_location(sub_name, str(py_file))
        if sub_spec is None or sub_spec.loader is None:
            continue
        sub_mod = importlib.util.module_from_spec(sub_spec)
        sys.modules[sub_name] = sub_mod
        sub_spec.loader.exec_module(sub_mod)


def _resolve_under(base: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    if p.exists():
        return p.resolve()
    return (base / p).resolve()


def _decode_output(text_or_bytes: str | bytes | None) -> str:
    if text_or_bytes is None:
        return ""
    if isinstance(text_or_bytes, bytes):
        return text_or_bytes.decode("utf-8", errors="replace")
    return text_or_bytes


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
            raise IsADirectoryError(f"--step must be a file: {p}")
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


def _parse_range_token(token: str) -> list[float] | None:
    parts = [x.strip() for x in token.split(",")]
    if len(parts) != 2:
        return None
    try:
        return [float(parts[0]), float(parts[1])]
    except ValueError:
        return None


def _parse_metrics(output: str) -> dict[str, Any]:
    singular_warning_count = len(SINGULAR_WARN_RE.findall(output))
    summary_warning_count = 0
    for m in SUMMARY_WARN_RE.finditer(output):
        summary_warning_count = max(summary_warning_count, int(m.group(1)))
    warning_line_count = 0
    for line in output.splitlines():
        if "warning" in line.lower():
            warning_line_count += 1

    final_nodes: int | None = None
    final_elements: int | None = None
    final_match = FINAL_MESH_RE.search(output)
    if final_match:
        final_nodes = int(final_match.group(1))
        final_elements = int(final_match.group(2))

    selected_strategy: str | None = None
    strategy_matches = SELECTED_STRATEGY_RE.findall(output)
    if strategy_matches:
        selected_strategy = strategy_matches[-1]
    invfix_ring: int | None = None
    invfix_core: int | None = None
    invfix_final: int | None = None

    singular_total_hits_reported = 0
    singular_unique_faces_reported = 0
    singular_unique_face_nodes_reported = 0
    faces: dict[str, Any] = {}
    nodes: dict[str, int] = {}

    for line in output.splitlines():
        tm = SINGULAR_TOTAL_RE.search(line)
        if tm:
            singular_total_hits_reported = int(tm.group(1))
            singular_unique_faces_reported = int(tm.group(2))
            singular_unique_face_nodes_reported = int(tm.group(3))
            continue

        fm = SINGULAR_DETAIL_RE.search(line)
        if fm:
            face_id = fm.group(1)
            hits = int(fm.group(2))
            radial = _parse_range_token(fm.group(3))
            axial = _parse_range_token(fm.group(4))
            faces[face_id] = {
                "hits": hits,
                "radial": radial,
                "axial": axial,
            }
            continue

        nm = SINGULAR_NODE_RE.search(line)
        if nm:
            key = f"{nm.group(1)}:{nm.group(2)}"
            nodes[key] = int(nm.group(3))
            continue

        im = INVFIX_RE.search(line)
        if im:
            label = im.group(1).lower()
            value = int(im.group(2))
            if label.startswith("ring_3d"):
                invfix_ring = value
            elif label.startswith("core_3d"):
                invfix_core = value
            elif label.startswith("final"):
                invfix_final = value

    if singular_total_hits_reported == 0 and faces:
        singular_total_hits_reported = sum(int(v.get("hits", 0)) for v in faces.values())
        singular_unique_faces_reported = len(faces)
    if singular_unique_face_nodes_reported == 0 and nodes:
        singular_unique_face_nodes_reported = len(nodes)

    return {
        "singular_warning_count": singular_warning_count,
        "summary_warning_count": summary_warning_count,
        "warning_line_count": warning_line_count,
        "final_nodes": final_nodes,
        "final_elements": final_elements,
        "invfix_ring": invfix_ring,
        "invfix_core": invfix_core,
        "invfix_final": invfix_final,
        "selected_strategy": selected_strategy,
        "singular_total_hits_reported": singular_total_hits_reported,
        "singular_unique_faces_reported": singular_unique_faces_reported,
        "singular_unique_face_nodes_reported": singular_unique_face_nodes_reported,
        "singular_faces_json": json.dumps(faces, ensure_ascii=False, sort_keys=True),
        "singular_nodes_json": json.dumps(nodes, ensure_ascii=False, sort_keys=True),
    }


def _run_engine_once(
    *,
    vexis_root: Path,
    python_exe: str,
    engine: EngineSpec,
    step_file: Path,
    output_mesh: Path,
    log_path: Path,
    timeout_sec: int,
) -> RunResult:
    cmd = [
        python_exe,
        "-m",
        engine.module,
        str(engine.config_path),
        str(step_file),
        "-o",
        str(output_mesh),
    ]
    cmd_text = " ".join(shlex.quote(c) for c in cmd)

    start = time.perf_counter()
    output = ""
    returncode = 1
    failure_reason: str | None = None
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(vexis_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        output = _decode_output(proc.stdout)
        returncode = int(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        output = _decode_output(exc.stdout)
        returncode = 124
        failure_reason = f"timeout_{timeout_sec}s"
    except Exception as exc:  # noqa: BLE001
        output = f"[harness-error] {exc}\n"
        returncode = 1
        failure_reason = f"exception: {exc}"
    elapsed = time.perf_counter() - start

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "CMD: " + cmd_text + "\n\n" + output,
        encoding="utf-8",
    )

    metrics = _parse_metrics(output)
    mesh_exists = output_mesh.exists()
    success = (returncode == 0) and mesh_exists
    if not success and failure_reason is None:
        if returncode != 0:
            failure_reason = f"exit_{returncode}"
        elif not mesh_exists:
            failure_reason = "mesh_output_missing"

    return RunResult(
        step_file=str(step_file),
        engine=engine.name,
        module=engine.module,
        command=cmd_text,
        returncode=returncode,
        success=success,
        duration_sec=elapsed,
        singular_warning_count=int(metrics["singular_warning_count"]),
        singular_total_hits_reported=int(metrics["singular_total_hits_reported"]),
        singular_unique_faces_reported=int(metrics["singular_unique_faces_reported"]),
        singular_unique_face_nodes_reported=int(metrics["singular_unique_face_nodes_reported"]),
        singular_faces_json=str(metrics["singular_faces_json"]),
        singular_nodes_json=str(metrics["singular_nodes_json"]),
        summary_warning_count=int(metrics["summary_warning_count"]),
        warning_line_count=int(metrics["warning_line_count"]),
        final_nodes=metrics["final_nodes"],
        final_elements=metrics["final_elements"],
        invfix_ring=metrics["invfix_ring"],
        invfix_core=metrics["invfix_core"],
        invfix_final=metrics["invfix_final"],
        selected_strategy=metrics["selected_strategy"],
        output_mesh=str(output_mesh),
        log_path=str(log_path),
        failure_reason=failure_reason,
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _convert_physical_bounds_to_ratio(cfg: Any) -> None:
    if getattr(cfg.freecad, "constraints_domain", "ratio") != "physical":
        return

    for b in cfg.bounds:
        base = float(b.base_value)
        if not math.isfinite(base) or base == 0.0:
            continue
        raw_min = float(b.min)
        raw_max = float(b.max)
        ratio_min = raw_min / base
        ratio_max = raw_max / base
        if ratio_min <= ratio_max:
            b.min = ratio_min
            b.max = ratio_max
        else:
            b.min = ratio_max
            b.max = ratio_min


def _generate_steps_with_v2(
    *,
    project_root: Path,
    v2_config_path: Path,
    v2_limits_path: Path,
    target_count: int,
    max_attempts: int,
    step_output_dir: Path,
    step_prefix: str,
    trial_id_start: int,
) -> tuple[list[Path], dict[str, Any]]:
    import optuna

    _ensure_v2_importable(project_root)

    from v2.cad_gate import CadGate
    from v2.config import load_config as load_v2_config
    from v2.constraints import check_hard_constraints
    from v2.geometry_adapter import GeometryAdapter, GeometryError
    from v2.search_space import (
        FeasibilityAwareSampler,
        build_fixed_search_space,
        create_sampler,
        make_constraints_func,
        normalize_bounds_to_sampling_grid,
        suggest_design_point,
    )
    from v2.types import DesignPoint

    cfg = load_v2_config(v2_config_path, v2_limits_path)
    cfg.freecad.step_output_dir = str(step_output_dir)
    cfg.freecad.step_filename_template = f"{step_prefix}tmp_{{trial_id}}.step"

    cad_gate = CadGate(cfg.cad_gate)
    if cfg.cad_gate.enabled and cfg.cad_gate.model_path and cad_gate._model is None:
        raise RuntimeError(
            "CAD gate is enabled but model failed to load. "
            "Check dependencies and model path before STEP generation."
        )

    geometry_adapter = GeometryAdapter(cfg.freecad, project_root, cfg.optimization)
    probe_ok = False
    try:
        base_values = geometry_adapter.probe_base_values([b.name for b in cfg.bounds])
        for b in cfg.bounds:
            if b.name in base_values:
                base = float(base_values[b.name])
                if math.isfinite(base) and base != 0.0:
                    b.base_value = base
        probe_ok = True
    except Exception:
        probe_ok = False

    _convert_physical_bounds_to_ratio(cfg)
    normalize_bounds_to_sampling_grid(
        cfg.bounds,
        optimization=cfg.optimization,
        discretization_step=cfg.optimization.discretization_step,
    )

    constraints_func = make_constraints_func()
    base_sampler = create_sampler(
        cfg.optimization,
        storage=None,
        constraints_func=constraints_func,
        n_objectives=1,
    )
    sampler: Any = base_sampler

    if cad_gate._model is not None:
        bounds_map = {b.name: b for b in cfg.bounds}
        baseline_params = {
            b.name: (1.0 if b.min <= 1.0 <= b.max else (b.min + b.max) / 2.0)
            for b in cfg.bounds
        }
        baseline_feas = cad_gate.predict(DesignPoint(trial_id=-1, params=baseline_params))

        def _predict_fn(params: dict[str, float]) -> bool:
            return cad_gate.predict(DesignPoint(trial_id=-1, params=params)).is_feasible

        def _predict_score_fn(params: dict[str, float]) -> float:
            feas = cad_gate.predict(DesignPoint(trial_id=-1, params=params))
            if feas.confidence is None:
                return 1.0 if feas.is_feasible else 0.0
            return float(feas.confidence)

        def _clip_params(params: dict[str, float]) -> dict[str, float]:
            clipped: dict[str, float] = {}
            for name, b in bounds_map.items():
                raw = float(params.get(name, baseline_params[name]))
                clipped[name] = min(max(raw, b.min), b.max)
            return clipped

        def _repair_fn(params: dict[str, float]) -> dict[str, float] | None:
            candidate = _clip_params(params)
            if _predict_fn(candidate):
                return candidate
            if not baseline_feas.is_feasible:
                return None
            best = dict(baseline_params)
            lo = 0.0
            hi = 1.0
            for _ in range(14):
                mid = (lo + hi) / 2.0
                mixed = {
                    name: baseline_params[name] + mid * (candidate[name] - baseline_params[name])
                    for name in baseline_params
                }
                if _predict_fn(mixed):
                    best = mixed
                    lo = mid
                else:
                    hi = mid
            return best

        sampler = FeasibilityAwareSampler(
            base_sampler=base_sampler,
            predict_fn=_predict_fn,
            predict_score_fn=_predict_score_fn,
            expected_param_names=list(bounds_map.keys()),
            fixed_search_space=build_fixed_search_space(
                cfg.bounds,
                optimization=cfg.optimization,
                discretization_step=cfg.optimization.discretization_step,
            ),
            repair_fn=_repair_fn,
            max_retries=cfg.cad_gate.rejection_max_retries,
            threshold_hint=cfg.cad_gate.threshold,
            exploration_enabled=cfg.cad_gate.exploration.enabled,
            global_ratio=cfg.cad_gate.exploration.global_ratio,
            boundary_ratio=cfg.cad_gate.exploration.boundary_ratio,
            local_ratio=cfg.cad_gate.exploration.local_ratio,
            boundary_candidate_pool=cfg.cad_gate.exploration.boundary_candidate_pool,
            uncertainty_band=cfg.cad_gate.exploration.uncertainty_band,
            uncertainty_accept_prob=cfg.cad_gate.exploration.uncertainty_accept_prob,
            local_perturbation_scale=cfg.cad_gate.exploration.local_perturbation_scale,
            local_archive_size=cfg.cad_gate.exploration.local_archive_size,
            random_seed=cfg.optimization.seed,
        )

    study = optuna.create_study(direction="minimize", sampler=sampler)

    step_output_dir.mkdir(parents=True, exist_ok=True)
    generated_steps: list[Path] = []
    generation_log: list[dict[str, Any]] = []

    attempts = 0
    hard_rejects = 0
    gate_rejects = 0
    geometry_failures = 0
    while len(generated_steps) < target_count and attempts < max_attempts:
        attempts += 1
        trial = study.ask()
        trial_id = trial_id_start + attempts
        point = suggest_design_point(
            trial,
            trial_id=trial_id,
            bounds=cfg.bounds,
            discretization_step=cfg.optimization.discretization_step,
            optimization=cfg.optimization,
        )

        hard_violation = check_hard_constraints(
            point,
            cfg.bounds,
            optimization=cfg.optimization,
        )
        if hard_violation:
            hard_rejects += 1
            study.tell(trial, 1.0)
            generation_log.append(
                {
                    "attempt": attempts,
                    "trial_id": trial_id,
                    "status": "hard_constraint_violation",
                    "reason": hard_violation,
                }
            )
            continue

        feas = cad_gate.predict(point)
        if not feas.is_feasible:
            gate_rejects += 1
            study.tell(trial, 1.0)
            generation_log.append(
                {
                    "attempt": attempts,
                    "trial_id": trial_id,
                    "status": "cad_gate_rejected",
                    "confidence": feas.confidence,
                    "reason": feas.reason_code,
                }
            )
            continue

        try:
            tmp_step = geometry_adapter.generate_step(point)
        except GeometryError as exc:
            geometry_failures += 1
            study.tell(trial, 1.0)
            generation_log.append(
                {
                    "attempt": attempts,
                    "trial_id": trial_id,
                    "status": "freecad_failed",
                    "reason": str(exc),
                }
            )
            continue

        idx = len(generated_steps) + 1
        final_step = step_output_dir / f"{step_prefix}{idx:02d}.step"
        if final_step.exists():
            final_step.unlink()
        tmp_step.replace(final_step)

        generated_steps.append(final_step)
        study.tell(trial, 0.0)
        generation_log.append(
            {
                "attempt": attempts,
                "trial_id": trial_id,
                "status": "generated",
                "confidence": feas.confidence,
                "step_path": str(final_step),
                "params": point.params,
                "physical_params": point.physical_params,
            }
        )

    geometry_adapter.close()

    if len(generated_steps) < target_count:
        raise RuntimeError(
            "Failed to generate required STEP count via CAD gate + FreeCAD. "
            f"generated={len(generated_steps)} target={target_count} attempts={attempts}"
        )

    generation_summary = {
        "target_count": target_count,
        "generated_count": len(generated_steps),
        "attempts": attempts,
        "hard_constraint_rejects": hard_rejects,
        "cad_gate_rejects": gate_rejects,
        "freecad_failures": geometry_failures,
        "probe_base_values_succeeded": probe_ok,
        "rejection_stats": (
            sampler.rejection_stats
            if hasattr(sampler, "rejection_stats")
            else {}
        ),
        "generated_steps": [str(p) for p in generated_steps],
        "attempt_log": generation_log,
    }
    return generated_steps, generation_summary


def _aggregate(results: list[RunResult]) -> dict[str, Any]:
    by_engine: dict[str, list[RunResult]] = {}
    for r in results:
        by_engine.setdefault(r.engine, []).append(r)

    engines: dict[str, dict[str, Any]] = {}
    hotspot_by_engine: dict[str, list[dict[str, Any]]] = {}
    for engine, runs in by_engine.items():
        n = len(runs)
        succ = sum(1 for r in runs if r.success)
        avg_time = sum(r.duration_sec for r in runs) / n if n else 0.0
        avg_singular = sum(r.singular_warning_count for r in runs) / n if n else 0.0
        avg_singular_reported = (
            sum(r.singular_total_hits_reported for r in runs) / n if n else 0.0
        )
        ring_invfix_vals = [int(r.invfix_ring) for r in runs if r.invfix_ring is not None]
        core_invfix_vals = [int(r.invfix_core) for r in runs if r.invfix_core is not None]
        final_invfix_vals = [int(r.invfix_final) for r in runs if r.invfix_final is not None]
        engines[engine] = {
            "runs": n,
            "success": succ,
            "success_rate": (succ / n) if n else 0.0,
            "avg_duration_sec": avg_time,
            "avg_singular_warning_count": avg_singular,
            "avg_singular_total_hits_reported": avg_singular_reported,
            "avg_invfix_ring": (sum(ring_invfix_vals) / len(ring_invfix_vals)) if ring_invfix_vals else None,
            "avg_invfix_core": (sum(core_invfix_vals) / len(core_invfix_vals)) if core_invfix_vals else None,
            "avg_invfix_final": (sum(final_invfix_vals) / len(final_invfix_vals)) if final_invfix_vals else None,
        }

        hotspot_totals: dict[str, dict[str, Any]] = {}
        for r in runs:
            try:
                faces = json.loads(r.singular_faces_json) if r.singular_faces_json else {}
            except json.JSONDecodeError:
                faces = {}
            if not isinstance(faces, dict):
                continue
            for face_id, info in faces.items():
                hits = int(info.get("hits", 0))
                radial = info.get("radial")
                axial = info.get("axial")
                key = f"face={face_id}|radial={radial}|axial={axial}"
                cur = hotspot_totals.get(key)
                if cur is None:
                    hotspot_totals[key] = {
                        "face_id": face_id,
                        "radial": radial,
                        "axial": axial,
                        "hits_total": hits,
                        "runs": 1,
                    }
                else:
                    cur["hits_total"] = int(cur["hits_total"]) + hits
                    cur["runs"] = int(cur["runs"]) + 1

        hotspot_by_engine[engine] = sorted(
            hotspot_totals.values(),
            key=lambda x: (-int(x["hits_total"]), str(x["face_id"])),
        )

    paired: dict[str, dict[str, RunResult]] = {}
    for r in results:
        paired.setdefault(r.step_file, {})[r.engine] = r

    comparisons: list[dict[str, Any]] = []
    for step_file, pair in sorted(paired.items()):
        b = pair.get("baseline")
        e = pair.get("experimental")
        if not b or not e:
            continue
        duration_ratio = (e.duration_sec / b.duration_sec) if b.duration_sec > 0 else None
        comparisons.append(
            {
                "step_file": step_file,
                "baseline_success": b.success,
                "experimental_success": e.success,
                "baseline_singular": b.singular_warning_count,
                "experimental_singular": e.singular_warning_count,
                "baseline_singular_reported": b.singular_total_hits_reported,
                "experimental_singular_reported": e.singular_total_hits_reported,
                "delta_singular_exp_minus_base": e.singular_warning_count - b.singular_warning_count,
                "baseline_time_sec": b.duration_sec,
                "experimental_time_sec": e.duration_sec,
                "time_ratio_exp_over_base": duration_ratio,
                "baseline_invfix_core": b.invfix_core,
                "experimental_invfix_core": e.invfix_core,
                "delta_invfix_core_exp_minus_base": (
                    (e.invfix_core - b.invfix_core)
                    if (e.invfix_core is not None and b.invfix_core is not None)
                    else None
                ),
            }
        )

    return {
        "engines": engines,
        "comparisons": comparisons,
        "singular_hotspots": hotspot_by_engine,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B comparison for baseline vs experimental mesh engines.",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--vexis-root", default="vexis")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--config-base", default="config/config.yaml")
    parser.add_argument("--config-exp", default="config/config_mesh_experimental.yaml")
    parser.add_argument("--steps-dir", default="input/step")
    parser.add_argument("--step", action="append", default=[])
    parser.add_argument("--step-glob", action="append", default=[])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--output-root", default="output/mesh_engine_ab")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--generate-steps-v2", action="store_true")
    parser.add_argument("--v2-config", default="config/optimizer_config.yaml")
    parser.add_argument("--v2-limits", default="config/v2_limitations.yaml")
    parser.add_argument("--generated-step-count", type=int, default=10)
    parser.add_argument("--max-generate-attempts", type=int, default=400)
    parser.add_argument("--generated-step-dir", default="input/step")
    parser.add_argument("--generated-step-prefix", default="ab_v2_step_")
    parser.add_argument("--trial-id-start", type=int, default=700000)

    args = parser.parse_args()

    python_exe = args.python_exe
    if any(sep in python_exe for sep in ("/", "\\")):
        py_path = Path(python_exe)
        if not py_path.is_absolute():
            py_path = (Path.cwd() / py_path).resolve()
        if py_path.exists():
            python_exe = str(py_path)

    project_root = Path(args.project_root).resolve()
    vexis_root = _resolve_under(project_root, args.vexis_root)
    if not vexis_root.exists():
        print(f"[ERROR] vexis root not found: {vexis_root}")
        return 2

    config_base = _resolve_under(vexis_root, args.config_base)
    config_exp = _resolve_under(vexis_root, args.config_exp)
    if not config_base.exists():
        print(f"[ERROR] baseline config not found: {config_base}")
        return 2
    if not config_exp.exists():
        print(f"[ERROR] experimental config not found: {config_exp}")
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _resolve_under(project_root, args.output_root) / f"run_{ts}"
    logs_dir = run_dir / "logs"
    meshes_dir = run_dir / "meshes"
    run_dir.mkdir(parents=True, exist_ok=True)

    steps: list[Path] = []
    generation_summary: dict[str, Any] = {}
    if args.generate_steps_v2:
        v2_config_path = _resolve_under(project_root, args.v2_config)
        v2_limits_path = _resolve_under(project_root, args.v2_limits)
        step_dir = _resolve_under(project_root, args.generated_step_dir)
        try:
            if args.dry_run:
                print("[DRY-RUN] v2 step generation plan:")
                print(f"[DRY-RUN]   config={v2_config_path}")
                print(f"[DRY-RUN]   limits={v2_limits_path}")
                print(f"[DRY-RUN]   step_dir={step_dir}")
                print(f"[DRY-RUN]   target={args.generated_step_count}")
                # For dry-run we still choose existing files to build A/B command preview.
                steps = _collect_step_files(
                    base_dir=project_root,
                    step_paths=args.step,
                    steps_dir=args.steps_dir,
                    step_globs=args.step_glob if args.step_glob else ["*.step", "*.stp"],
                    max_files=args.max_files,
                )
            else:
                steps, generation_summary = _generate_steps_with_v2(
                    project_root=project_root,
                    v2_config_path=v2_config_path,
                    v2_limits_path=v2_limits_path,
                    target_count=args.generated_step_count,
                    max_attempts=args.max_generate_attempts,
                    step_output_dir=step_dir,
                    step_prefix=args.generated_step_prefix,
                    trial_id_start=args.trial_id_start,
                )
                (run_dir / "generated_steps_summary.json").write_text(
                    json.dumps(generation_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"[INFO] generated STEP files: {len(steps)}")
                for p in steps:
                    print(f"[INFO]   - {p}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] failed during v2 STEP generation: {exc}")
            return 2
    else:
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
        EngineSpec("baseline", "src.mesh_gen.main", config_base),
        EngineSpec("experimental", "src.mesh_gen_experimental.main", config_exp),
    ]

    plan = {
        "project_root": str(project_root),
        "vexis_root": str(vexis_root),
        "python_exe": python_exe,
        "steps": [str(p) for p in steps],
        "engines": [
            {"name": e.name, "module": e.module, "config": str(e.config_path)}
            for e in engines
        ],
        "timeout_sec": args.timeout_sec,
        "dry_run": bool(args.dry_run),
        "generate_steps_v2": bool(args.generate_steps_v2),
        "generation_summary": generation_summary,
    }
    (run_dir / "run_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[INFO] A/B run dir: {run_dir}")
    print(f"[INFO] STEP files: {len(steps)}")
    for step in steps:
        print(f"[INFO]   - {step}")

    if args.dry_run:
        for step in steps:
            for engine in engines:
                output_mesh = meshes_dir / engine.name / f"{step.stem}_{engine.name}.vtk"
                cmd = [
                    python_exe,
                    "-m",
                    engine.module,
                    str(engine.config_path),
                    str(step),
                    "-o",
                    str(output_mesh),
                ]
                print("[DRY-RUN]", " ".join(shlex.quote(c) for c in cmd))
        print("[INFO] dry-run finished (no engine execution)")
        return 0

    results: list[RunResult] = []
    for step in steps:
        for engine in engines:
            output_mesh = meshes_dir / engine.name / f"{step.stem}_{engine.name}.vtk"
            log_path = logs_dir / engine.name / f"{step.stem}.log"
            output_mesh.parent.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] run start: engine={engine.name} step={step.name}")
            result = _run_engine_once(
                vexis_root=vexis_root,
                python_exe=python_exe,
                engine=engine,
                step_file=step,
                output_mesh=output_mesh,
                log_path=log_path,
                timeout_sec=args.timeout_sec,
            )
            results.append(result)
            print(
                "[INFO] run end: engine=%s step=%s success=%s rc=%d singular=%d reported=%d time=%.2fs"
                % (
                    result.engine,
                    Path(result.step_file).name,
                    result.success,
                    result.returncode,
                    result.singular_warning_count,
                    result.singular_total_hits_reported,
                    result.duration_sec,
                )
            )

    rows = [asdict(r) for r in results]
    _write_csv(run_dir / "ab_results.csv", rows)

    agg = _aggregate(results)
    (run_dir / "ab_summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    comparison_rows = list(agg.get("comparisons", []))
    if comparison_rows:
        _write_csv(run_dir / "ab_comparison.csv", comparison_rows)

    print(f"[INFO] result csv: {run_dir / 'ab_results.csv'}")
    print(f"[INFO] summary json: {run_dir / 'ab_summary.json'}")
    if comparison_rows:
        print(f"[INFO] comparison csv: {run_dir / 'ab_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
