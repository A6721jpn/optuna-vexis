"""
v2 FreeCAD Engine

Core FreeCAD operations extracted from proto3-hybrid/hybrid_solver.py.
Provides headless FreeCAD execution for constraint updates and STEP export.
"""

from __future__ import annotations

import ast
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Lazy FreeCAD import with environment setup
_FreeCAD = None


def _get_freecad():
    """Lazy-load FreeCAD module with environment fallback."""
    global _FreeCAD
    if _FreeCAD is not None:
        return _FreeCAD

    try:
        import FreeCAD  # type: ignore
        _FreeCAD = FreeCAD
        return _FreeCAD
    except ImportError:
        pass

    # Try to find FreeCAD in conda environments
    conda_prefix = os.environ.get(
        "CONDA_PREFIX",
        r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad",
    )
    conda_prefixes = [p for p in os.environ.get("CONDA_PREFIXES", "").split(";") if p]
    env_candidates = []
    if conda_prefix:
        env_candidates.append(Path(conda_prefix))
    for name in ("fcad", "fcad-codex", "b123d"):
        env_candidates.append(Path(r"C:\Users\aokuni\AppData\Local\miniforge3\envs") / name)
    for p in conda_prefixes:
        env_candidates.append(Path(p))

    for env_path in env_candidates:
        freecad_bin = env_path / "Library" / "bin"
        freecad_lib = env_path / "Library" / "lib"
        if not freecad_bin.exists() and not freecad_lib.exists():
            continue
        os.environ["PATH"] = str(freecad_bin) + os.pathsep + os.environ.get("PATH", "")
        sys.path.insert(0, str(freecad_bin))
        if freecad_lib.exists():
            sys.path.insert(0, str(freecad_lib))
        try:
            import FreeCAD  # type: ignore
            _FreeCAD = FreeCAD
            logger.info("FreeCAD loaded from %s", env_path)
            return _FreeCAD
        except ImportError:
            continue

    # Final fallback: explicit FreeCAD install path
    fallback_bin = Path(
        os.environ.get(
            "FREECAD_BIN",
            r"C:\Program Files\FreeCAD 1.0\bin",
        )
    )
    if fallback_bin.exists():
        os.environ["PATH"] = str(fallback_bin) + os.pathsep + os.environ.get("PATH", "")
        sys.path.insert(0, str(fallback_bin))
        try:
            import FreeCAD  # type: ignore
            _FreeCAD = FreeCAD
            logger.info("FreeCAD loaded from %s", fallback_bin)
            return _FreeCAD
        except ImportError:
            pass

    raise ImportError("FreeCAD not found in any conda environment or fallback path")


@dataclass
class ConstraintSpec:
    """Specification for a sketch constraint."""
    index: int
    name: str
    ctype: str  # "Distance", "DistanceX", "DistanceY", "Angle"
    base_value: float
    sketch: Optional[Any] = None
    angle_unit: Optional[str] = None  # "rad" or "deg"
    physical_step: Optional[float] = None
    min_ratio: Optional[float] = None
    max_ratio: Optional[float] = None


@dataclass
class RelativeRuleEvaluation:
    rule_id: str
    lhs: float
    rhs: float
    op: str
    violation: float
    passed: bool


@dataclass
class RelativeRuleCheck:
    passed: bool
    evaluations: List[RelativeRuleEvaluation]
    objective: float
    max_violation: float


class FreecadEngine:
    """Execute FreeCAD operations for constraint updates and STEP export."""

    def __init__(
        self,
        fcstd_path: Path,
        sketch_name: Optional[str] = None,
        surface_name: Optional[str] = None,
        surface_label: Optional[str] = None,
    ) -> None:
        self._fcstd_path = Path(fcstd_path)
        self._sketch_name = sketch_name
        self._surface_name = surface_name or "Face"
        self._surface_label = surface_label or "SURFACE"
        self._doc = None
        self._sketch = None
        self._surface = None
        self._specs: List[ConstraintSpec] = []

    def open(self) -> None:
        """Open FreeCAD document and find sketch/surface."""
        FreeCAD = _get_freecad()
        
        # Disable auto-remove redundants to prevent constraints from invalidating
        p = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/Sketcher")
        p.SetBool("AutoRemoveRedundants", True)
        p.SetBool("AutoRecompute", True)

        if not self._fcstd_path.exists():
            raise FileNotFoundError(f"FCStd not found: {self._fcstd_path}")
        
        # Ensure we always open a fresh document or handle existing correctly
        self._doc = FreeCAD.openDocument(str(self._fcstd_path))
        
        self._sketch = self._find_sketch(self._doc, self._sketch_name)
        self._surface = self._find_surface(self._doc, self._surface_name, self._surface_label)
        
        logger.info("Opened FreeCAD document: %s", self._fcstd_path)

    def close(self) -> None:
        """Close the document."""
        if self._doc:
            FreeCAD = _get_freecad()
            FreeCAD.closeDocument(self._doc.Name)
            self._doc = None
            self._sketch = None
            self._surface = None

    def set_constraints(self, specs: List[ConstraintSpec]) -> None:
        """Set the constraint specifications to use."""
        self._specs = specs
        # Update sketch reference in specs as the document might have been re-opened
        for spec in specs:
            spec.sketch = self._sketch

    def apply_ratios(
        self,
        params: Dict[str, float],
        *,
        relative_rules: Optional[List[dict[str, Any]]] = None,
        relative_repair: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Apply parameter ratios and optionally enforce relative constraints."""
        if self._sketch is None:
            raise RuntimeError("Sketch not loaded")

        ratio_params = self._normalize_ratio_params(params)
        if not self._apply_ratio_params(
            ratio_params,
            save_error_doc=True,
            check_surface=True,
        ):
            return False

        rules = self._normalize_relative_rules(relative_rules)
        if not rules:
            return True

        check = self._evaluate_relative_rules(rules)
        if check.passed:
            return True

        logger.info(
            "Relative-constraint violation detected: max_violation=%.6g, objective=%.6g",
            check.max_violation,
            check.objective,
        )

        actions = {str(rule.get("on_violation", "repair_then_reject")).strip().lower() for rule in rules}
        if "repair_then_reject" not in actions:
            logger.warning("Relative constraints failed and repair is disabled by rule action")
            return False

        repaired_params = self._repair_relative_constraints(
            ratio_params=ratio_params,
            rules=rules,
            repair_cfg=relative_repair or {},
            baseline_check=check,
        )
        if repaired_params is None:
            logger.warning("Relative-constraint repair failed")
            return False

        # Ensure the final document state reflects the repaired candidate.
        if not self._apply_ratio_params(
            repaired_params,
            save_error_doc=True,
            check_surface=True,
        ):
            return False

        final_check = self._evaluate_relative_rules(rules)
        if not final_check.passed:
            logger.warning(
                "Relative-constraint repair ended infeasible: max_violation=%.6g",
                final_check.max_violation,
            )
            return False

        return True

    def _normalize_ratio_params(self, params: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for spec in self._specs:
            raw = params.get(spec.name, 1.0)
            try:
                ratio = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(ratio):
                continue
            out[spec.name] = self._clip_and_quantize_ratio(spec, ratio)
        return out

    def _apply_ratio_params(
        self,
        ratio_params: Dict[str, float],
        *,
        save_error_doc: bool,
        check_surface: bool,
    ) -> bool:
        if self._sketch is None:
            return False

        for spec in self._specs:
            if spec.name not in ratio_params:
                continue

            ratio = ratio_params[spec.name]
            raw_physical_value = spec.base_value * ratio
            quantized_physical_value = self._quantize_physical_value(
                raw_physical_value,
                spec.physical_step,
            )
            sketch_value = self._clamp_candidate(
                spec.ctype,
                quantized_physical_value,
                spec.angle_unit,
            )

            if sketch_value is None:
                logger.warning("Invalid value for %s: ratio=%.6f", spec.name, ratio)
                return False

            try:
                self._set_constraint_value(
                    spec.sketch,
                    spec.index,
                    sketch_value,
                    spec.ctype,
                    spec.angle_unit,
                    constraint_name=spec.name,
                )
            except Exception as exc:
                self._log_constraint_set_error(spec, sketch_value, exc)
                return False

        self._doc.recompute()
        if not self._check_recompute(self._doc):
            logger.debug("Recompute failed")
            if save_error_doc:
                self._save_error_doc()
            return False
        if check_surface and not self._check_surface(self._surface):
            logger.debug("Surface validation failed")
            return False
        return True

    def _log_constraint_set_error(self, spec: ConstraintSpec, sketch_value: float, exc: Exception) -> None:
        try:
            c = spec.sketch.Constraints[spec.index]
            logger.error(
                "Failed to set constraint %s (idx=%d, val=%.4f). "
                "State: Value=%.4f, Driving=%s, Active=%s, Type=%s. Error: %s",
                spec.name,
                spec.index,
                sketch_value,
                c.Value,
                getattr(c, "Driving", "n/a"),
                getattr(c, "IsActive", "n/a"),
                c.Type,
                exc,
            )
        except Exception:
            logger.error(
                "Failed to set constraint %s (idx=%d). Could not retrieve state. Error: %s",
                spec.name,
                spec.index,
                exc,
            )

    def _save_error_doc(self) -> None:
        try:
            error_dir = self._fcstd_path.parent.parent / "output" / "error_docs"
            error_dir.mkdir(parents=True, exist_ok=True)
            import time

            timestamp = int(time.time() * 1000)
            error_path = error_dir / f"error_{timestamp}.FCStd"
            self._doc.saveAs(str(error_path))
            logger.info("Saved error document to %s", error_path)
        except Exception as exc:
            logger.warning("Failed to save error document: %s", exc)

    def _clip_and_quantize_ratio(self, spec: ConstraintSpec, ratio: float) -> float:
        low = spec.min_ratio
        high = spec.max_ratio
        if low is not None and high is not None and low > high:
            low, high = high, low
        if low is not None and ratio < low:
            ratio = low
        if high is not None and ratio > high:
            ratio = high

        if spec.physical_step is not None and spec.base_value not in (0.0, -0.0):
            ratio_step = abs(spec.physical_step / spec.base_value)
            if math.isfinite(ratio_step) and ratio_step > 0.0:
                ratio = round(ratio / ratio_step) * ratio_step
                ratio = self._round_by_step(ratio, ratio_step)
                if low is not None and ratio < low:
                    ratio = low
                if high is not None and ratio > high:
                    ratio = high
        return ratio

    @staticmethod
    def _quantize_physical_value(value: float, step: Optional[float]) -> float:
        if step is None:
            return value
        if not math.isfinite(step) or step <= 0:
            return value
        q = round(value / step) * step
        step_str = f"{step:.12f}".rstrip("0")
        digits = len(step_str.split(".")[1]) if "." in step_str else 0
        return round(q, digits)

    @staticmethod
    def _round_by_step(value: float, step: float) -> float:
        step_str = f"{step:.12f}".rstrip("0")
        digits = len(step_str.split(".")[1]) if "." in step_str else 0
        return round(value, digits)

    def _normalize_relative_rules(
        self,
        raw_rules: Optional[List[dict[str, Any]]],
    ) -> List[dict[str, Any]]:
        if not raw_rules:
            return []

        out: List[dict[str, Any]] = []
        valid_ops = {">=", "<=", ">", "<", "=="}
        for i, raw_rule in enumerate(raw_rules):
            if not isinstance(raw_rule, dict):
                continue
            lhs = str(raw_rule.get("lhs", raw_rule.get("expr", ""))).strip()
            rhs = str(raw_rule.get("rhs", "0.0")).strip() or "0.0"
            op = str(raw_rule.get("op", ">=")).strip()
            if not lhs:
                continue
            if op not in valid_ops:
                logger.warning(
                    "Skip relative rule with unsupported op '%s' (id=%s)",
                    op,
                    raw_rule.get("id", f"rule_{i+1}"),
                )
                continue

            try:
                lhs_ast = self._compile_expr(lhs)
                rhs_ast = self._compile_expr(rhs)
            except Exception as exc:
                logger.warning(
                    "Skip invalid relative rule expression (id=%s): %s",
                    raw_rule.get("id", f"rule_{i+1}"),
                    exc,
                )
                continue

            drivers = raw_rule.get("repair_drivers", [])
            if not isinstance(drivers, list):
                drivers = []
            on_violation = str(raw_rule.get("on_violation", "repair_then_reject")).strip().lower()
            rule = {
                "id": str(raw_rule.get("id", f"rule_{i+1}")).strip() or f"rule_{i+1}",
                "lhs": lhs,
                "rhs": rhs,
                "op": op,
                "tolerance": max(0.0, float(raw_rule.get("tolerance", 1.0e-4))),
                "weight": max(1.0e-12, float(raw_rule.get("weight", 1.0))),
                "on_violation": on_violation,
                "repair_drivers": [str(x) for x in drivers if str(x).strip()],
                "_lhs_ast": lhs_ast,
                "_rhs_ast": rhs_ast,
            }
            out.append(rule)
        return out

    @staticmethod
    def _canonical_constraint_name(name: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")
        return normalized.upper()

    def _constraint_value_map(self) -> Dict[str, float]:
        if self._sketch is None:
            return {}
        values: Dict[str, float] = {}
        for i in range(self._sketch.ConstraintCount):
            c = self._sketch.Constraints[i]
            name = str(getattr(c, "Name", "") or "").strip()
            if not name:
                continue
            try:
                value = float(c.Value)
            except Exception:
                continue
            if not math.isfinite(value):
                continue
            values[name] = value
        return values

    @staticmethod
    def _compile_expr(expression: str) -> ast.AST:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Expression):
                continue
            if isinstance(node, (ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd)):
                continue
            if isinstance(node, ast.BinOp):
                if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
                    raise ValueError(f"Unsupported operator: {node.op.__class__.__name__}")
                continue
            if isinstance(node, ast.UnaryOp):
                if not isinstance(node.op, (ast.USub, ast.UAdd)):
                    raise ValueError(f"Unsupported unary operator: {node.op.__class__.__name__}")
                continue
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Only simple function names are allowed")
                if node.func.id not in {"abs", "min", "max"}:
                    raise ValueError(f"Unsupported function: {node.func.id}")
                if node.keywords:
                    raise ValueError("Keyword arguments are not supported")
                continue
            if isinstance(node, ast.Name):
                continue
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    continue
                raise ValueError("Only numeric literals are allowed")
            raise ValueError(f"Unsupported expression node: {node.__class__.__name__}")
        return tree

    def _evaluate_expr(self, tree: ast.AST, values: Dict[str, float]) -> float:
        builtins = {"abs": abs, "min": min, "max": max}

        def _eval(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant):
                return float(node.value)
            if isinstance(node, ast.Name):
                direct = values.get(node.id)
                if direct is not None:
                    return float(direct)
                upper = values.get(node.id.upper())
                if upper is not None:
                    return float(upper)
                canonical = values.get(self._canonical_constraint_name(node.id))
                if canonical is not None:
                    return float(canonical)
                raise KeyError(f"Unknown variable '{node.id}'")
            if isinstance(node, ast.BinOp):
                lhs = _eval(node.left)
                rhs = _eval(node.right)
                if isinstance(node.op, ast.Add):
                    return lhs + rhs
                if isinstance(node.op, ast.Sub):
                    return lhs - rhs
                if isinstance(node.op, ast.Mult):
                    return lhs * rhs
                if isinstance(node.op, ast.Div):
                    return lhs / rhs
                if isinstance(node.op, ast.Pow):
                    return lhs ** rhs
                if isinstance(node.op, ast.Mod):
                    return lhs % rhs
                raise ValueError(f"Unsupported operator: {node.op.__class__.__name__}")
            if isinstance(node, ast.UnaryOp):
                v = _eval(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return +v
                if isinstance(node.op, ast.USub):
                    return -v
                raise ValueError(f"Unsupported unary operator: {node.op.__class__.__name__}")
            if isinstance(node, ast.Call):
                fn_name = node.func.id  # validated by _compile_expr
                fn = builtins[fn_name]
                args = [_eval(arg) for arg in node.args]
                return float(fn(*args))
            raise ValueError(f"Unsupported expression node: {node.__class__.__name__}")

        value = float(_eval(tree))
        if not math.isfinite(value):
            raise ValueError("Expression result is not finite")
        return value

    @staticmethod
    def _rule_violation(lhs: float, rhs: float, op: str, tolerance: float) -> float:
        if op == ">=":
            return max(0.0, rhs - lhs - tolerance)
        if op == "<=":
            return max(0.0, lhs - rhs - tolerance)
        if op == ">":
            return max(0.0, rhs - lhs + tolerance)
        if op == "<":
            return max(0.0, lhs - rhs + tolerance)
        if op == "==":
            return max(0.0, abs(lhs - rhs) - tolerance)
        return float("inf")

    def _evaluate_relative_rules(self, rules: List[dict[str, Any]]) -> RelativeRuleCheck:
        values = self._constraint_value_map()
        eval_values: Dict[str, float] = {"PI": math.pi, "E": math.e}
        for name, value in values.items():
            eval_values[name] = value
            eval_values[name.upper()] = value
            eval_values[self._canonical_constraint_name(name)] = value

        evaluations: List[RelativeRuleEvaluation] = []
        objective = 0.0
        max_violation = 0.0
        all_passed = True

        for rule in rules:
            try:
                lhs = self._evaluate_expr(rule["_lhs_ast"], eval_values)
                rhs = self._evaluate_expr(rule["_rhs_ast"], eval_values)
                violation = self._rule_violation(lhs, rhs, rule["op"], rule["tolerance"])
                passed = violation <= 0.0
            except Exception as exc:
                logger.warning(
                    "Relative rule evaluation failed (id=%s): %s",
                    rule["id"],
                    exc,
                )
                lhs = float("nan")
                rhs = float("nan")
                violation = 1.0e6
                passed = False

            objective += float(rule["weight"]) * (violation ** 2)
            if violation > max_violation:
                max_violation = violation
            all_passed = all_passed and passed
            evaluations.append(
                RelativeRuleEvaluation(
                    rule_id=str(rule["id"]),
                    lhs=lhs,
                    rhs=rhs,
                    op=str(rule["op"]),
                    violation=violation,
                    passed=passed,
                )
            )

        return RelativeRuleCheck(
            passed=all_passed,
            evaluations=evaluations,
            objective=objective,
            max_violation=max_violation,
        )

    def _repair_relative_constraints(
        self,
        *,
        ratio_params: Dict[str, float],
        rules: List[dict[str, Any]],
        repair_cfg: dict[str, Any],
        baseline_check: RelativeRuleCheck,
    ) -> Optional[Dict[str, float]]:
        if not bool(repair_cfg.get("enabled", True)):
            return None

        spec_by_key = {self._canonical_constraint_name(spec.name): spec for spec in self._specs}
        driver_keys: set[str] = set()
        for rule in rules:
            for raw_name in rule.get("repair_drivers", []):
                key = self._canonical_constraint_name(raw_name)
                if key in spec_by_key:
                    driver_keys.add(key)
        if not driver_keys:
            driver_keys = set(spec_by_key.keys())
        driver_specs = [spec_by_key[k] for k in sorted(driver_keys)]
        if not driver_specs:
            return None

        max_iters = max(1, int(repair_cfg.get("max_iters", 20)))
        max_evals = max(1, int(repair_cfg.get("max_evals", 80)))
        step_decay = float(repair_cfg.get("step_decay", 0.5))
        if not (0.0 < step_decay < 1.0):
            step_decay = 0.5
        initial_step_scale = max(1.0, float(repair_cfg.get("initial_step_scale", 6.0)))
        min_step_ratio_cfg = max(1.0e-8, float(repair_cfg.get("min_step_ratio", 1.0e-4)))
        regularization_lambda = max(0.0, float(repair_cfg.get("regularization_lambda", 1.0e-2)))

        base_params = dict(ratio_params)
        best_params = dict(ratio_params)
        eval_count = 1
        best_check = baseline_check

        min_steps: Dict[str, float] = {}
        steps: Dict[str, float] = {}
        scales: Dict[str, float] = {}
        for spec in driver_specs:
            name = spec.name
            span = 1.0
            if spec.min_ratio is not None and spec.max_ratio is not None:
                lo = min(spec.min_ratio, spec.max_ratio)
                hi = max(spec.min_ratio, spec.max_ratio)
                span = max(hi - lo, min_step_ratio_cfg)
            base_step = min_step_ratio_cfg
            if spec.physical_step is not None and spec.base_value not in (0.0, -0.0):
                ratio_step = abs(spec.physical_step / spec.base_value)
                if math.isfinite(ratio_step) and ratio_step > 0.0:
                    base_step = max(base_step, ratio_step)
            min_steps[name] = base_step
            steps[name] = max(base_step, min(span, base_step * initial_step_scale))
            scales[name] = max(span, abs(base_params.get(name, 1.0)), 1.0)

        cache: Dict[tuple[tuple[str, float], ...], tuple[float, RelativeRuleCheck, Dict[str, float]]] = {}

        def _evaluate(candidate: Dict[str, float]) -> tuple[float, RelativeRuleCheck]:
            key = tuple(sorted((name, round(candidate[name], 12)) for name in candidate))
            cached = cache.get(key)
            if cached is not None:
                return cached[0], cached[1]

            merged = dict(ratio_params)
            merged.update(candidate)
            merged = self._normalize_ratio_params(merged)
            if not self._apply_ratio_params(merged, save_error_doc=False, check_surface=True):
                failed = RelativeRuleCheck(
                    passed=False,
                    evaluations=[],
                    objective=1.0e9,
                    max_violation=1.0e6,
                )
                cache[key] = (1.0e9, failed, merged)
                return 1.0e9, failed

            check = self._evaluate_relative_rules(rules)
            regularization = 0.0
            for spec in driver_specs:
                name = spec.name
                delta = merged.get(name, base_params.get(name, 1.0)) - base_params.get(name, 1.0)
                scale = scales[name]
                regularization += (delta / scale) ** 2
            objective = check.objective + (regularization_lambda * regularization)
            cache[key] = (objective, check, merged)
            return objective, check

        best_objective = best_check.objective
        if regularization_lambda > 0.0:
            best_objective += 0.0

        for _ in range(max_iters):
            if best_check.passed:
                break
            if eval_count >= max_evals:
                break

            improved = False
            for spec in driver_specs:
                name = spec.name
                current = best_params.get(name, base_params.get(name, 1.0))
                local_best_params = None
                local_best_check = None
                local_best_objective = best_objective

                for direction in (1.0, -1.0):
                    candidate = dict(best_params)
                    candidate[name] = self._clip_and_quantize_ratio(
                        spec,
                        current + (steps[name] * direction),
                    )
                    if math.isclose(candidate[name], current, rel_tol=0.0, abs_tol=1.0e-14):
                        continue
                    eval_count += 1
                    objective, check = _evaluate(candidate)
                    if objective + 1.0e-12 < local_best_objective:
                        local_best_objective = objective
                        local_best_check = check
                        local_best_params = candidate
                    if eval_count >= max_evals:
                        break

                if local_best_params is not None and local_best_check is not None:
                    best_params = local_best_params
                    best_check = local_best_check
                    best_objective = local_best_objective
                    improved = True
                    if best_check.passed:
                        break
                if eval_count >= max_evals:
                    break

            if best_check.passed:
                break

            if not improved:
                any_reduced = False
                for spec in driver_specs:
                    name = spec.name
                    next_step = steps[name] * step_decay
                    if next_step >= min_steps[name]:
                        if next_step < steps[name]:
                            any_reduced = True
                        steps[name] = next_step
                    else:
                        steps[name] = min_steps[name]
                if not any_reduced:
                    break

        if best_check.passed:
            logger.info(
                "Relative-constraint repair succeeded: evals=%d, max_violation=%.6g",
                eval_count,
                best_check.max_violation,
            )
            return best_params

        logger.warning(
            "Relative-constraint repair exhausted: evals=%d, max_violation=%.6g",
            eval_count,
            best_check.max_violation,
        )
        return None

    def export_step(self, output_path: Path) -> Path:
        """Export current geometry to STEP file.
        
        Args:
            output_path: Target STEP file path.
            
        Returns:
            Path to exported STEP file.
        """
        if self._surface is None:
            raise RuntimeError("No surface to export")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import Part  # type: ignore
        Part.export([self._surface], str(output_path))
        logger.info("Exported STEP: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Private helpers (from proto3-hybrid)
    # ------------------------------------------------------------------

    def _find_sketch(self, doc, sketch_hint: Optional[str] = None):
        """Find sketch object in document."""
        sketches = [obj for obj in doc.Objects if obj.TypeId == "Sketcher::SketchObject"]
        if not sketches:
            raise ValueError("No Sketcher::SketchObject found in document")
        if sketch_hint:
            for obj in sketches:
                if obj.Name == sketch_hint or obj.Label == sketch_hint:
                    return obj
            hint_lower = sketch_hint.lower()
            for obj in sketches:
                if obj.Name.lower() == hint_lower or obj.Label.lower() == hint_lower:
                    return obj
            raise ValueError(f"Sketch not found: {sketch_hint}")
        return sketches[0]

    def _find_surface(self, doc, name: Optional[str], label: Optional[str]):
        """Find surface object in document."""
        if name:
            obj = doc.getObject(name)
            if obj:
                return obj
        if label:
            for obj in doc.Objects:
                if obj.Label == label:
                    return obj
        return None

    def _check_surface(self, obj) -> bool:
        """Validate surface geometry."""
        if obj is None:
            return False
        shape = getattr(obj, "Shape", None)
        if shape is None:
            return False
        if hasattr(shape, "isNull") and shape.isNull():
            return False
        try:
            if hasattr(shape, "isValid") and not shape.isValid():
                return False
        except Exception:
            return False
        try:
            if hasattr(shape, "check"):
                problems = shape.check(True)
                if problems:
                    return False
        except Exception:
            return False
        try:
            area = getattr(shape, "Area", None)
            if area is not None and area <= 0:
                return False
        except Exception:
            return False
        return True

    def _check_recompute(self, doc) -> bool:
        """Check if recompute was successful and log details if not."""
        try:
            objects = list(getattr(doc, "Objects", []))
        except Exception:
            objects = []
            
        success = True
        for obj in objects:
            try:
                state = getattr(obj, "State", [])
                # State is a list of strings in recent FreeCAD versions
                if any(flag in state for flag in ("Invalid", "RecomputeError")):
                    logger.warning(
                        "Object '%s' (%s) error state: %s", 
                        obj.Name, obj.Label, state
                    )
                    success = False
            except Exception:
                continue
        return success

    def _set_constraint_value(
        self,
        sketch,
        index: int,
        value: float,
        ctype: str,
        angle_unit: Optional[str],
        constraint_name: Optional[str] = None,
    ) -> None:
        """Set constraint value in sketch using setExpression for robustness."""
        FreeCAD = _get_freecad()
        
        # Robustly find index by name if provided
        if constraint_name:
            found_idx = -1
            for i, c in enumerate(sketch.Constraints):
                if c.Name == constraint_name:
                    found_idx = i
                    break
            if found_idx != -1:
                index = found_idx
            else:
                logger.warning(
                    "Constraint '%s' not found by name, using index %d",
                    constraint_name, index
                )
        
        # Construct expression string
        expression = f"{value}"
        if ctype == "Angle":
            # Convert radians to degrees if necessary because we append " deg"
            if angle_unit == "rad":
                val_deg = math.degrees(value)
                expression = f"{val_deg} deg"
            else:
                expression = f"{value} deg"
        else:
            # Assuming params are standard length units (mm).
            expression = f"{value} mm"

        try:
            # Use setExpression which is often more robust than setDatum
            # Path is usually 'Constraints[i]'
            path = f"Constraints[{index}]"
            sketch.setExpression(path, expression)
            
            # Also need to ensure the value is actually applied? 
            # setExpression usually sets the expression but value update happens on recompute.
            # However, we can also try to force set the datum value if expression is empty?
            # No, if we set expression, it overrides datum.
            
        except Exception as exc:
            # Fallback or re-raise with detail
            raise RuntimeError(f"setExpression('{path}', '{expression}') failed: {exc}") from exc



    def _clamp_candidate(self, ctype: str, value: float, angle_unit: Optional[str]) -> Optional[float]:
        """Validate and clamp candidate value."""
        if not math.isfinite(value):
            return None
        if ctype in {"Distance", "DistanceX", "DistanceY"}:
            if value <= 0:
                return None
            return value
        if ctype == "Angle":
            if angle_unit == "rad":
                if value <= 0 or value >= math.pi:
                    return None
                return value
            if value <= 0 or value >= 180:
                return None
            return value
        return value

    def _convert_to_sketch_value(self, ctype: str, value: float, angle_unit: Optional[str]) -> Optional[float]:
        """Convert value to sketch-compatible format."""
        if not math.isfinite(value):
            return None
        if ctype in {"Distance", "DistanceX", "DistanceY"}:
            if value <= 0:
                return None
            return value
        if ctype == "Angle":
            if angle_unit == "rad":
                if value <= 0 or value >= math.pi:
                    return None
                return value
            if value <= 0 or value >= 180:
                return None
            return value
        return value
