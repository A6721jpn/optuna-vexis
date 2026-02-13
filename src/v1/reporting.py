"""
v1.0 Markdown Reporting

Generate a post-run Markdown report with:
  - CAE completion summary
  - Iteration table (features + objective values)
  - Optuna configuration summary
  - Optimization history chart
  - Pareto chart for 2-objective studies
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import optuna

from .config import V1Config

logger = logging.getLogger(__name__)

FAILURE_STAGE_ATTR = "v1_0_failure_stage"
FAILURE_REASON_ATTR = "v1_0_failure_reason"
LEGACY_FAILURE_STAGE_ATTR = "proto4_failure_stage"
LEGACY_FAILURE_REASON_ATTR = "proto4_failure_reason"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _trial_attr(
    trial: optuna.trial.FrozenTrial,
    primary_key: str,
    legacy_key: str,
) -> Any:
    if primary_key in trial.user_attrs:
        return trial.user_attrs.get(primary_key)
    return trial.user_attrs.get(legacy_key)


def _load_trial_records(result_dir: Path) -> dict[int, dict[str, Any]]:
    trial_records: dict[int, dict[str, Any]] = {}
    trials_dir = result_dir / "trials"
    if not trials_dir.exists():
        return trial_records

    for trial_file in sorted(trials_dir.glob("trial_*/trial_info.json")):
        try:
            data = json.loads(trial_file.read_text(encoding="utf-8"))
            trial_id = int(data.get("trial_id"))
        except Exception:
            continue
        trial_records[trial_id] = data
    return trial_records


def _objective_labels(cfg: V1Config, study: optuna.study.Study) -> list[str]:
    if cfg.optimization.objective_type != "multi":
        return ["rmse"] if len(study.directions) == 1 else [
            f"objective_{i + 1}" for i in range(len(study.directions))
        ]

    labels: list[str] = []
    if cfg.objective.include_rmse_in_multi:
        labels.append("rmse")

    objective_names = cfg.objective.multi_objectives or list(cfg.objective.features.keys())
    for name in objective_names:
        if cfg.objective.multi_objectives_use_error:
            labels.append(f"{name}_error")
        else:
            labels.append(name)

    if len(labels) != len(study.directions):
        return [f"objective_{i + 1}" for i in range(len(study.directions))]
    return labels


def _plot_optimization_history(
    study: optuna.study.Study,
    path: Path,
    objective_label: str,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Matplotlib unavailable; fallback to SVG history plot: %s", exc)
        xs: list[int] = []
        ys: list[float] = []
        for t in sorted(study.trials, key=lambda x: x.number):
            if t.state.name != "COMPLETE":
                continue
            if t.values:
                y = t.values[0]
            else:
                y = t.value
            if y is None:
                continue
            xs.append(t.number)
            ys.append(float(y))
        if not xs:
            return None
        best_so_far: list[float] = []
        cur = float("inf")
        for y in ys:
            cur = y if y < cur else cur
            best_so_far.append(cur)
        svg_path = path.with_suffix(".svg")
        _write_history_svg(svg_path, xs, ys, best_so_far, objective_label=objective_label)
        return svg_path

    xs: list[int] = []
    ys: list[float] = []
    for t in sorted(study.trials, key=lambda x: x.number):
        if t.state.name != "COMPLETE":
            continue
        if t.values:
            y = t.values[0]
        else:
            y = t.value
        if y is None:
            continue
        xs.append(t.number)
        ys.append(float(y))

    if not xs:
        return None

    best_so_far: list[float] = []
    cur = float("inf")
    for y in ys:
        cur = y if y < cur else cur
        best_so_far.append(cur)

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, "o", color="#4C72B0", alpha=0.7, label="Trial value")
    plt.plot(xs, best_so_far, "-", color="#C44E52", linewidth=2.0, label="Best so far")
    plt.title(f"Objective Progress Over Iterations ({objective_label})")
    plt.xlabel("Trial")
    plt.ylabel(objective_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def _plot_pareto_front_2d(
    study: optuna.study.Study,
    path: Path,
    objective_labels: list[str],
) -> Optional[Path]:
    if len(study.directions) != 2:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Matplotlib unavailable; fallback to SVG pareto plot: %s", exc)
        all_points: list[tuple[float, float]] = []
        for t in study.trials:
            if t.state.name != "COMPLETE" or not t.values or len(t.values) < 2:
                continue
            all_points.append((float(t.values[0]), float(t.values[1])))
        if not all_points:
            return None
        pareto_points: list[tuple[float, float]] = []
        for t in study.best_trials:
            if not t.values or len(t.values) < 2:
                continue
            pareto_points.append((float(t.values[0]), float(t.values[1])))
        svg_path = path.with_suffix(".svg")
        _write_pareto_svg(
            svg_path,
            all_points,
            pareto_points,
            x_label=objective_labels[0],
            y_label=objective_labels[1],
        )
        return svg_path

    all_points: list[tuple[float, float]] = []
    for t in study.trials:
        if t.state.name != "COMPLETE" or not t.values or len(t.values) < 2:
            continue
        all_points.append((float(t.values[0]), float(t.values[1])))
    if not all_points:
        return None

    pareto_points: list[tuple[float, float]] = []
    for t in study.best_trials:
        if not t.values or len(t.values) < 2:
            continue
        pareto_points.append((float(t.values[0]), float(t.values[1])))

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.scatter(
        [p[0] for p in all_points],
        [p[1] for p in all_points],
        alpha=0.35,
        color="#4C72B0",
        label="All complete trials",
    )
    if pareto_points:
        ax.scatter(
            [p[0] for p in pareto_points],
            [p[1] for p in pareto_points],
            color="#C44E52",
            label="Pareto front",
        )

    ax.set_title("Pareto Front (2 objectives)")
    ax.set_xlabel(objective_labels[0])
    ax.set_ylabel(objective_labels[1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def _range_with_padding(values: list[float]) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        pad = 1.0 if math.isclose(lo, 0.0) else abs(lo) * 0.1
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" '
        'font-family="sans-serif" font-size="16" fill="#222">'
        f"{title}</text>",
    ]


def _write_history_svg(
    path: Path,
    xs: list[int],
    ys: list[float],
    best_so_far: list[float],
    *,
    objective_label: str,
) -> None:
    width, height = 1000, 620
    m_left, m_right, m_top, m_bottom = 70, 30, 50, 70
    plot_w = width - m_left - m_right
    plot_h = height - m_top - m_bottom
    xlo, xhi = _range_with_padding([float(x) for x in xs])
    ylo, yhi = _range_with_padding(ys)

    def sx(v: float) -> float:
        return m_left + (v - xlo) / (xhi - xlo) * plot_w

    def sy(v: float) -> float:
        return m_top + (1.0 - (v - ylo) / (yhi - ylo)) * plot_h

    lines = _svg_header(
        width,
        height,
        f"Objective Progress Over Iterations ({objective_label})",
    )
    lines.append(
        f'<line x1="{m_left}" y1="{m_top + plot_h}" x2="{m_left + plot_w}" '
        f'y2="{m_top + plot_h}" stroke="#222" stroke-width="1.2"/>'
    )
    lines.append(
        f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top + plot_h}" '
        'stroke="#222" stroke-width="1.2"/>'
    )

    trial_pts = " ".join(f"{sx(float(x)):.2f},{sy(float(y)):.2f}" for x, y in zip(xs, ys))
    best_pts = " ".join(f"{sx(float(x)):.2f},{sy(float(y)):.2f}" for x, y in zip(xs, best_so_far))
    lines.append(
        f'<polyline points="{trial_pts}" fill="none" stroke="#4C72B0" '
        'stroke-width="1.8" opacity="0.9"/>'
    )
    lines.append(
        f'<polyline points="{best_pts}" fill="none" stroke="#C44E52" '
        'stroke-width="2.2" opacity="0.95"/>'
    )
    for x, y in zip(xs, ys):
        lines.append(
            f'<circle cx="{sx(float(x)):.2f}" cy="{sy(float(y)):.2f}" r="2.8" fill="#4C72B0"/>'
        )

    lines.append(
        f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" '
        'font-family="sans-serif" font-size="12" fill="#333">Trial</text>'
    )
    lines.append(
        f'<text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" '
        'text-anchor="middle" font-family="sans-serif" font-size="12" fill="#333">'
        f"{objective_label}</text>"
    )
    lines.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_pareto_svg(
    path: Path,
    all_points: list[tuple[float, float]],
    pareto_points: list[tuple[float, float]],
    *,
    x_label: str,
    y_label: str,
) -> None:
    width, height = 900, 620
    m_left, m_right, m_top, m_bottom = 70, 30, 50, 70
    plot_w = width - m_left - m_right
    plot_h = height - m_top - m_bottom
    x_vals = [p[0] for p in all_points]
    y_vals = [p[1] for p in all_points]
    xlo, xhi = _range_with_padding(x_vals)
    ylo, yhi = _range_with_padding(y_vals)

    def sx(v: float) -> float:
        return m_left + (v - xlo) / (xhi - xlo) * plot_w

    def sy(v: float) -> float:
        return m_top + (1.0 - (v - ylo) / (yhi - ylo)) * plot_h

    lines = _svg_header(width, height, f"Pareto Front ({x_label} vs {y_label})")
    lines.append(
        f'<line x1="{m_left}" y1="{m_top + plot_h}" x2="{m_left + plot_w}" '
        f'y2="{m_top + plot_h}" stroke="#222" stroke-width="1.2"/>'
    )
    lines.append(
        f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top + plot_h}" '
        'stroke="#222" stroke-width="1.2"/>'
    )

    for x, y in all_points:
        lines.append(
            f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.0" fill="#4C72B0" opacity="0.45"/>'
        )
    for x, y in pareto_points:
        lines.append(
            f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.8" fill="#C44E52" opacity="0.95"/>'
        )

    lines.append(
        f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="12" fill="#333">{x_label}</text>'
    )
    lines.append(
        f'<text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" '
        'text-anchor="middle" font-family="sans-serif" font-size="12" fill="#333">'
        f"{y_label}</text>"
    )
    lines.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _collect_table_columns(
    cfg: V1Config,
    study: optuna.study.Study,
    trial_records: dict[int, dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    param_names = [b.name for b in cfg.bounds]
    for trial in study.trials:
        for name in trial.params:
            if name not in param_names:
                param_names.append(name)
    metric_names: list[str] = []
    for rec in trial_records.values():
        for key in sorted((rec.get("objective_values") or {}).keys()):
            if key not in metric_names:
                metric_names.append(key)
    objective_cols = _objective_labels(cfg, study)
    metric_names = [m for m in metric_names if m not in objective_cols]
    return param_names, objective_cols, metric_names


def _build_iteration_table(
    cfg: V1Config,
    study: optuna.study.Study,
    trial_records: dict[int, dict[str, Any]],
) -> str:
    param_names, objective_cols, metric_names = _collect_table_columns(cfg, study, trial_records)
    headers = [
        "trial_id",
        "outcome",
        *objective_cols,
        *metric_names,
        "failure_reason",
        "wall_clock_sec",
        *param_names,
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for trial in sorted(study.trials, key=lambda t: t.number):
        rec = trial_records.get(trial.number, {})
        values = list(trial.values) if trial.values else (
            [trial.value] if trial.value is not None else []
        )
        value_cells = [_fmt(values[i]) if i < len(values) else "-" for i in range(len(objective_cols))]
        metrics = rec.get("objective_values") or {}
        metric_cells = [_fmt(metrics.get(name)) for name in metric_names]
        params = trial.params or (rec.get("design_point") or {}).get("params", {})
        param_cells = [_fmt(params.get(name)) for name in param_names]
        row = [
            str(trial.number),
            rec.get("outcome", "-"),
            *value_cells,
            *metric_cells,
            _fmt(_trial_attr(trial, FAILURE_REASON_ATTR, LEGACY_FAILURE_REASON_ATTR)),
            _fmt(rec.get("wall_clock_sec")),
            *param_cells,
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_markdown_report(
    *,
    result_dir: Path,
    study: optuna.study.Study,
    cfg: V1Config,
    optimizer_config_path: str,
    limits_config_path: str,
    start_time: datetime,
    end_time: datetime,
    actual_sampler_name: str,
    rejection_stats: Optional[dict[str, int]] = None,
    version_info: Optional[dict[str, Any]] = None,
) -> Path:
    """Generate a post-run Markdown report and return its path."""
    report_path = result_dir / "report_v1_0.md"
    assets_dir = result_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    trial_records = _load_trial_records(result_dir)
    total_trials = len(study.trials)
    cae_success = sum(
        1
        for t in study.trials
        if (trial_records.get(t.number, {}).get("outcome") == "cae_success")
    )

    objective_labels = _objective_labels(cfg, study)
    primary_label = objective_labels[0] if objective_labels else "objective_1"

    history_img = assets_dir / "optimization_history.png"
    history_path = _plot_optimization_history(
        study,
        history_img,
        objective_label=primary_label,
    )

    pareto_img = assets_dir / "pareto_front_2d.png"
    pareto_labels = (
        objective_labels[:2]
        if len(objective_labels) >= 2
        else ["objective_1", "objective_2"]
    )
    pareto_path = _plot_pareto_front_2d(study, pareto_img, pareto_labels)

    stage_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for t in study.trials:
        stage = _trial_attr(t, FAILURE_STAGE_ATTR, LEGACY_FAILURE_STAGE_ATTR)
        reason = _trial_attr(t, FAILURE_REASON_ATTR, LEGACY_FAILURE_REASON_ATTR)
        if stage:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    best_value_text = "-"
    try:
        best_value_text = _fmt(study.best_value)
    except Exception:
        pass

    lines: list[str] = []
    lines.append("# Production v1.0 最適化レポート")
    lines.append("")
    lines.append(f"- 生成時刻: {datetime.now().isoformat()}")
    lines.append(f"- 実行開始: {start_time.isoformat()}")
    lines.append(f"- 実行終了: {end_time.isoformat()}")
    lines.append("")

    lines.append("## 1. 実行サマリ")
    lines.append("")
    lines.append(f"- CAE成功（完遂）数 / イテレーション総数: **{cae_success} / {total_trials}**")
    lines.append(f"- Best objective value: `{best_value_text}`")
    if rejection_stats:
        repaired = rejection_stats.get("repaired")
        repaired_text = f", repaired={repaired}" if repaired is not None else ""
        lines.append(
            "- Rejection stats: "
            f"accepted={rejection_stats.get('accepted', 0)}, "
            f"rejected={rejection_stats.get('rejected', 0)}"
            f"{repaired_text}"
        )
    if stage_counts:
        lines.append(f"- 失敗ステージ集計: `{stage_counts}`")
    if reason_counts:
        lines.append(f"- 失敗理由集計: `{reason_counts}`")
    lines.append("")

    lines.append("## 2. Optuna設定")
    lines.append("")
    lines.append(f"- Optimizer config: `{optimizer_config_path}`")
    lines.append(f"- Limits config: `{limits_config_path}`")
    lines.append(f"- Sampler (設定値): `{cfg.optimization.sampler}`")
    lines.append(f"- Sampler (実際): `{actual_sampler_name}`")
    lines.append(f"- Objective type: `{cfg.optimization.objective_type}`")
    lines.append(f"- Directions: `{[d.name for d in study.directions]}`")
    lines.append(f"- Objective labels: `{objective_labels}`")
    lines.append(f"- Seed: `{cfg.optimization.seed}`")
    lines.append(f"- n_startup_trials: `{cfg.optimization.n_startup_trials}`")
    lines.append(f"- max_trials (config): `{cfg.optimization.max_trials}`")
    lines.append(f"- convergence_threshold: `{cfg.optimization.convergence_threshold}`")
    lines.append(f"- patience: `{cfg.optimization.patience}`")
    if version_info:
        lines.append(f"- Product version: `{version_info.get('line')} {version_info.get('version')}`")
        lines.append(f"- Baseline: `{version_info.get('baseline')}`")
        lines.append(f"- Git commit: `{version_info.get('git_commit')}`")
        lines.append(f"- Git branch: `{version_info.get('git_branch')}`")
        lines.append(f"- Git dirty: `{version_info.get('git_dirty')}`")
    lines.append("")

    lines.append("## 3. 各イテレーションの特徴量と評価関数結果")
    lines.append("")
    lines.append(_build_iteration_table(cfg, study, trial_records))
    lines.append("")

    lines.append("## 4. 目標関数の最適化時系列進捗グラフ")
    lines.append("")
    if history_path:
        rel = history_path.relative_to(result_dir).as_posix()
        lines.append(f"![Optimization History]({rel})")
    else:
        lines.append("- グラフを生成できませんでした。")
    lines.append("")

    lines.append("## 5. パレート図（2目的時のみ）")
    lines.append("")
    if pareto_path:
        rel = pareto_path.relative_to(result_dir).as_posix()
        lines.append(f"![Pareto Front]({rel})")
    elif len(study.directions) == 2:
        lines.append("- 2目的ですが、表示に必要なデータ不足のため生成できませんでした。")
    else:
        lines.append("- 今回は2目的最適化ではないため対象外です。")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return report_path
