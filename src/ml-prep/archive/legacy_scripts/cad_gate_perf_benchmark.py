"""
Benchmark CAD gate ML inference and FreeCAD CAD check performance.

Measures:
  - decision latency
  - process RSS memory (including child processes)

No CAE execution.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


def _bootstrap_proto4_codex(project_root: Path) -> None:
    pkg_dir = project_root / "src" / "proto4-codex"
    sys.path.insert(0, str(pkg_dir))
    spec = importlib.util.spec_from_file_location(
        "proto4_codex",
        str(pkg_dir / "__init__.py"),
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Failed to bootstrap proto4_codex package")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["proto4_codex"] = mod
    spec.loader.exec_module(mod)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ML gate and FreeCAD check (no CAE)"
    )
    parser.add_argument("--config", default="config/optimizer_config.yaml")
    parser.add_argument("--limits", default="config/proto4_limitations.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ml-samples", type=int, default=1000)
    parser.add_argument("--cad-samples", type=int, default=200)
    parser.add_argument("--warmup-ml", type=int, default=30)
    parser.add_argument("--warmup-cad", type=int, default=10)
    parser.add_argument(
        "--sampler",
        choices=("lhs", "uniform"),
        default="lhs",
        help="Sampling strategy for parameter space",
    )
    parser.add_argument(
        "--expand-factor",
        type=float,
        default=1.0,
        help="Multiply parameter half-range around center",
    )
    parser.add_argument(
        "--memory-interval-ms",
        type=int,
        default=20,
        help="RSS sampling interval in milliseconds",
    )
    parser.add_argument("--freecad-bin", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * p
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    low = sorted_values[lower]
    high = sorted_values[upper]
    weight = position - lower
    return low * (1.0 - weight) + high * weight


class ProcessMemorySampler:
    def __init__(self, interval_sec: float) -> None:
        self._backend = "psutil"
        self._psutil = None
        self._proc = None
        self._kernel32 = None
        self._psapi = None
        self._counters_type = None
        self._ctypes = None
        self._wintypes = None
        self._rss_valid = True
        try:
            import psutil

            self._psutil = psutil
            self._proc = psutil.Process(os.getpid())
        except Exception:
            # Fallback: current-process RSS via Windows API
            import ctypes
            from ctypes import wintypes

            self._backend = "winapi"
            self._ctypes = ctypes
            self._wintypes = wintypes
            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._psapi = ctypes.WinDLL("psapi", use_last_error=True)
            self._kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            self._psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
                wintypes.DWORD,
            ]
            self._psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            self._counters_type = PROCESS_MEMORY_COUNTERS_EX
        self._interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_rss = 0
        self.end_rss = 0
        self.peak_rss = 0

    def _current_rss(self) -> int:
        if self._backend == "psutil" and self._proc is not None:
            rss = self._proc.memory_info().rss
            for child in self._proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except Exception:
                    pass
            return rss
        counters = self._counters_type()
        counters.cb = self._ctypes.sizeof(self._counters_type)
        proc_handle = self._kernel32.GetCurrentProcess()
        ok = self._psapi.GetProcessMemoryInfo(
            proc_handle,
            self._ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            self._rss_valid = False
            return 0
        return int(counters.WorkingSetSize)

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._current_rss()
            if rss > self.peak_rss:
                self.peak_rss = rss
            time.sleep(self._interval_sec)

    def start(self) -> None:
        self.start_rss = self._current_rss()
        self.peak_rss = self.start_rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.end_rss = self._current_rss()
        if self.end_rss > self.peak_rss:
            self.peak_rss = self.end_rss


def _make_uniform_points(
    bounds: dict[str, tuple[float, float]],
    n: int,
    rng: random.Random,
) -> list[dict[str, float]]:
    keys = list(bounds.keys())
    points: list[dict[str, float]] = []
    for _ in range(n):
        row: dict[str, float] = {}
        for key in keys:
            lo, hi = bounds[key]
            row[key] = rng.uniform(lo, hi)
        points.append(row)
    return points


def _make_lhs_points(
    bounds: dict[str, tuple[float, float]],
    n: int,
    rng: random.Random,
) -> list[dict[str, float]]:
    keys = list(bounds.keys())
    unit_cols: dict[str, list[float]] = {}
    for key in keys:
        slots = list(range(n))
        rng.shuffle(slots)
        values = []
        for slot in slots:
            values.append((slot + rng.random()) / n)
        unit_cols[key] = values

    points: list[dict[str, float]] = []
    for index in range(n):
        row: dict[str, float] = {}
        for key in keys:
            lo, hi = bounds[key]
            u = unit_cols[key][index]
            row[key] = lo + (hi - lo) * u
        points.append(row)
    return points


def _latency_stats(latencies_ms: list[float]) -> dict[str, float | None]:
    if not latencies_ms:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": mean(latencies_ms),
        "p50": _percentile(latencies_ms, 0.50),
        "p95": _percentile(latencies_ms, 0.95),
        "max": max(latencies_ms),
    }


def _rss_stats(sampler: ProcessMemorySampler) -> dict[str, float]:
    return {
        "start_mb": sampler.start_rss / (1024 * 1024),
        "end_mb": sampler.end_rss / (1024 * 1024),
        "peak_mb": sampler.peak_rss / (1024 * 1024),
        "peak_delta_mb": (sampler.peak_rss - sampler.start_rss) / (1024 * 1024),
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    _bootstrap_proto4_codex(project_root)

    if args.freecad_bin:
        os.environ["FREECAD_BIN"] = args.freecad_bin

    from proto4_codex.cad_gate import CadGate
    from proto4_codex.config import load_config
    from proto4_codex.geometry_adapter import GeometryAdapter, GeometryError
    from proto4_codex.types import DesignPoint

    cfg = load_config(project_root / args.config, project_root / args.limits)
    rng = random.Random(args.seed)

    bounds: dict[str, tuple[float, float]] = {}
    for key, spec in cfg.freecad.constraints.items():
        lo = float(spec.get("min", 1.0))
        hi = float(spec.get("max", 1.0))
        if lo > hi:
            lo, hi = hi, lo
        if args.expand_factor != 1.0:
            center = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            half = half * args.expand_factor
            lo = max(1e-6, center - half)
            hi = center + half
        bounds[key] = (lo, hi)

    total_points = max(args.ml_samples + args.warmup_ml, args.cad_samples + args.warmup_cad)
    if args.sampler == "lhs":
        all_points = _make_lhs_points(bounds, total_points, rng)
    else:
        all_points = _make_uniform_points(bounds, total_points, rng)

    cad_gate = CadGate(cfg.cad_gate)
    geometry_adapter = GeometryAdapter(cfg.freecad, project_root)

    interval_sec = max(0.001, args.memory_interval_ms / 1000.0)

    # ML benchmark
    for index in range(args.warmup_ml):
        point = DesignPoint(trial_id=10_000_000 + index, params=all_points[index])
        cad_gate.predict(point)

    ml_sampler = ProcessMemorySampler(interval_sec=interval_sec)
    ml_latencies: list[float] = []
    ml_confidences: list[float] = []
    ml_sampler.start()
    ml_t0 = time.perf_counter()
    for index in range(args.ml_samples):
        point = DesignPoint(trial_id=11_000_000 + index, params=all_points[index])
        t0 = time.perf_counter()
        result = cad_gate.predict(point)
        ml_latencies.append((time.perf_counter() - t0) * 1000.0)
        if result.confidence is not None:
            ml_confidences.append(float(result.confidence))
    ml_total_sec = time.perf_counter() - ml_t0
    ml_sampler.stop()

    # FreeCAD benchmark
    for index in range(args.warmup_cad):
        point = DesignPoint(trial_id=20_000_000 + index, params=all_points[index])
        try:
            geometry_adapter.generate_step(point)
        except Exception:
            pass
        finally:
            try:
                geometry_adapter.cleanup(point)
            except Exception:
                pass

    cad_sampler = ProcessMemorySampler(interval_sec=interval_sec)
    cad_latencies: list[float] = []
    cad_success = 0
    cad_fail = 0
    cad_fail_examples: list[str] = []
    cad_sampler.start()
    cad_t0 = time.perf_counter()
    for index in range(args.cad_samples):
        point = DesignPoint(trial_id=21_000_000 + index, params=all_points[index])
        t0 = time.perf_counter()
        try:
            step_path = geometry_adapter.generate_step(point)
            ok = step_path.exists() and step_path.stat().st_size > 0
            if ok:
                cad_success += 1
            else:
                cad_fail += 1
        except GeometryError as exc:
            cad_fail += 1
            if len(cad_fail_examples) < 5:
                cad_fail_examples.append(str(exc))
        except Exception as exc:
            cad_fail += 1
            if len(cad_fail_examples) < 5:
                cad_fail_examples.append(str(exc))
        finally:
            cad_latencies.append((time.perf_counter() - t0) * 1000.0)
            try:
                geometry_adapter.cleanup(point)
            except Exception:
                pass
    cad_total_sec = time.perf_counter() - cad_t0
    cad_sampler.stop()

    payload = {
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "seed": args.seed,
            "ml_samples": args.ml_samples,
            "cad_samples": args.cad_samples,
            "warmup_ml": args.warmup_ml,
            "warmup_cad": args.warmup_cad,
            "sampler": args.sampler,
            "expand_factor": args.expand_factor,
            "memory_interval_ms": args.memory_interval_ms,
            "gate_threshold": cfg.cad_gate.threshold,
            "freecad_bin": os.environ.get("FREECAD_BIN"),
        },
        "ml_inference": {
            "samples": args.ml_samples,
            "total_sec": ml_total_sec,
            "throughput_per_sec": (args.ml_samples / ml_total_sec) if ml_total_sec > 0 else None,
            "latency_ms": _latency_stats(ml_latencies),
            "rss_mb": _rss_stats(ml_sampler),
            "rss_backend": ml_sampler._backend,
            "rss_valid": ml_sampler._rss_valid,
            "confidence": {
                "min": min(ml_confidences) if ml_confidences else None,
                "max": max(ml_confidences) if ml_confidences else None,
                "mean": mean(ml_confidences) if ml_confidences else None,
            },
        },
        "freecad_check": {
            "samples": args.cad_samples,
            "success": cad_success,
            "fail": cad_fail,
            "success_rate": (cad_success / args.cad_samples) if args.cad_samples else None,
            "total_sec": cad_total_sec,
            "throughput_per_sec": (args.cad_samples / cad_total_sec) if cad_total_sec > 0 else None,
            "latency_ms": _latency_stats(cad_latencies),
            "rss_mb": _rss_stats(cad_sampler),
            "rss_backend": cad_sampler._backend,
            "rss_valid": cad_sampler._rss_valid,
            "fail_examples": cad_fail_examples,
        },
    }

    out_dir = project_root / "output" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / f"cad_gate_perf_{_now_tag()}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
