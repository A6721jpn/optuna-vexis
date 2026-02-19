"""
v2-claude Feasibility Oracle

FreeCAD-First feasibility manifold learning module.

Replaces the static ML gate (cad_gate.py) with a dynamic system:
  1. FreeCAD lightweight checks (~0.5s) as ground-truth oracle
  2. Interaction-aware surrogate (GradientBoosting with pairwise features)
  3. Online retraining from accumulated FreeCAD results
  4. 3-tier feasibility judgment (high-confidence / uncertain / reject)

The surrogate learns parameter interactions that the axis-aligned
hypercube bounds cannot capture, enabling broader exploration of the
true feasible manifold.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import BoundsSpec, FeasibilitySpec
from .geometry_adapter import GeometryAdapter

logger = logging.getLogger(__name__)


@dataclass
class FeasibilityRecord:
    """Single feasibility observation."""

    params: dict[str, float]
    feasible: bool
    source: str  # "freecad", "discovery", "optimization"
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)


class FeasibilityDB:
    """In-memory feasibility database with optional disk persistence."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._records: list[FeasibilityRecord] = []
        self._db_path = Path(db_path) if db_path else None
        if self._db_path and self._db_path.exists():
            self._load()

    def add(self, record: FeasibilityRecord) -> None:
        self._records.append(record)

    def add_batch(self, records: list[FeasibilityRecord]) -> None:
        self._records.extend(records)

    @property
    def records(self) -> list[FeasibilityRecord]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    @property
    def feasible_count(self) -> int:
        return sum(1 for r in self._records if r.feasible)

    @property
    def infeasible_count(self) -> int:
        return sum(1 for r in self._records if not r.feasible)

    def save(self) -> None:
        if self._db_path is None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "params": r.params,
                "feasible": r.feasible,
                "source": r.source,
                "timestamp": r.timestamp,
                "details": r.details,
            }
            for r in self._records
        ]
        self._db_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("Feasibility DB saved: %d records → %s", len(data), self._db_path)

    def _load(self) -> None:
        try:
            raw = json.loads(self._db_path.read_text(encoding="utf-8"))
            for item in raw:
                self._records.append(
                    FeasibilityRecord(
                        params=item["params"],
                        feasible=item["feasible"],
                        source=item.get("source", "loaded"),
                        timestamp=item.get("timestamp", 0.0),
                        details=item.get("details", {}),
                    )
                )
            logger.info("Feasibility DB loaded: %d records from %s", len(self._records), self._db_path)
        except Exception as exc:
            logger.warning("Failed to load feasibility DB from %s: %s", self._db_path, exc)


class InteractionSurrogate:
    """GradientBoosting classifier with pairwise interaction features.

    Input: 20 raw parameters → 20 + C(20,2) = 210 features
    Output: feasibility probability in [0, 1]

    The pairwise interaction terms allow the model to capture
    confounding effects like "CROWN-D-L × SHOULDER-ANGLE-OUT"
    that determine whether a parameter combination is geometrically feasible.
    """

    def __init__(
        self,
        param_names: list[str],
        use_interactions: bool = True,
    ) -> None:
        self._param_names = list(param_names)
        self._use_interactions = use_interactions
        self._model: Any = None
        self._n_raw = len(param_names)
        self._n_features = self._n_raw
        if use_interactions:
            self._n_features += self._n_raw * (self._n_raw - 1) // 2
        self._trained = False
        self._train_count = 0

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def train_count(self) -> int:
        return self._train_count

    def _build_features(self, raw: np.ndarray) -> np.ndarray:
        """Build feature matrix including pairwise interactions.

        Args:
            raw: (N, n_params) array of raw parameter values.

        Returns:
            (N, n_features) array with raw + interaction columns.
        """
        if not self._use_interactions:
            return raw

        n_samples = raw.shape[0]
        n_pairs = self._n_raw * (self._n_raw - 1) // 2
        interactions = np.empty((n_samples, n_pairs), dtype=np.float64)

        idx = 0
        for i in range(self._n_raw):
            for j in range(i + 1, self._n_raw):
                interactions[:, idx] = raw[:, i] * raw[:, j]
                idx += 1

        return np.hstack([raw, interactions])

    def _params_to_raw(self, params: dict[str, float]) -> np.ndarray:
        """Convert a single params dict to a raw feature row."""
        return np.array(
            [params.get(name, 1.0) for name in self._param_names],
            dtype=np.float64,
        )

    def train(self, records: list[FeasibilityRecord]) -> None:
        """Train or retrain the surrogate from feasibility records."""
        if len(records) < 10:
            logger.debug("Not enough records (%d) for surrogate training", len(records))
            return

        from sklearn.ensemble import GradientBoostingClassifier

        raw_list = [self._params_to_raw(r.params) for r in records]
        labels = [1 if r.feasible else 0 for r in records]

        raw = np.array(raw_list, dtype=np.float64)
        X = self._build_features(raw)
        y = np.array(labels, dtype=np.int32)

        # Check class balance
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            logger.warning(
                "Surrogate training skipped: single-class data (pos=%d, neg=%d)",
                n_pos, n_neg,
            )
            return

        self._model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self._model.fit(X, y)
        self._trained = True
        self._train_count = len(records)
        logger.info(
            "Surrogate trained: %d samples (%d feasible, %d infeasible), %d features",
            len(records), n_pos, n_neg, X.shape[1],
        )

    def predict_score(self, params: dict[str, float]) -> float:
        """Predict feasibility probability for a single point.

        Returns:
            Score in [0, 1] where higher = more likely feasible.
            Returns 0.5 if model is not trained.
        """
        if not self._trained or self._model is None:
            return 0.5

        raw = self._params_to_raw(params).reshape(1, -1)
        X = self._build_features(raw)
        proba = self._model.predict_proba(X)[0]
        # proba is [P(infeasible), P(feasible)] — return P(feasible)
        return float(proba[1]) if len(proba) > 1 else float(proba[0])

    def predict_scores_batch(self, params_list: list[dict[str, float]]) -> np.ndarray:
        """Predict feasibility scores for multiple points."""
        if not self._trained or self._model is None:
            return np.full(len(params_list), 0.5)

        raw = np.array(
            [self._params_to_raw(p) for p in params_list],
            dtype=np.float64,
        )
        X = self._build_features(raw)
        proba = self._model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]


class FeasibilityOracle:
    """FreeCAD-First feasibility oracle with learned surrogate.

    Provides:
      - Ground-truth feasibility checks via FreeCAD subprocess
      - Surrogate-based fast predictions with interaction awareness
      - 3-tier evaluation (high-confidence / uncertain / reject)
      - Phase 0 discovery for initial feasible region mapping
      - Online surrogate retraining during optimization
    """

    def __init__(
        self,
        geometry_adapter: GeometryAdapter,
        bounds: list[BoundsSpec],
        feasibility_spec: FeasibilitySpec,
    ) -> None:
        self._geom = geometry_adapter
        self._bounds = bounds
        self._spec = feasibility_spec
        self._param_names = [b.name for b in bounds]
        self._db = FeasibilityDB(feasibility_spec.db_path)
        self._surrogate = InteractionSurrogate(
            self._param_names,
            use_interactions=feasibility_spec.interaction_features,
        )
        self._checks_since_retrain = 0

        # Auto-train surrogate from loaded DB records
        if len(self._db) > 0:
            logger.info(
                "Auto-training surrogate from %d loaded DB records",
                len(self._db),
            )
            self._surrogate.train(self._db.records)

    @property
    def db(self) -> FeasibilityDB:
        return self._db

    @property
    def surrogate(self) -> InteractionSurrogate:
        return self._surrogate

    # ------------------------------------------------------------------
    # FreeCAD ground-truth check
    # ------------------------------------------------------------------

    def check_freecad(
        self,
        params: dict[str, float],
        source: str = "optimization",
    ) -> FeasibilityRecord:
        """Run FreeCAD lightweight feasibility check.

        Args:
            params: {constraint_name: ratio_value} dict.
            source: Label for the record source.

        Returns:
            FeasibilityRecord with ground-truth result.
        """
        t0 = time.time()
        result = self._geom.check_feasibility(params)
        elapsed = time.time() - t0

        feasible = result.get("feasible", False)
        record = FeasibilityRecord(
            params=dict(params),
            feasible=feasible,
            source=source,
            details={
                "check_time_sec": round(elapsed, 3),
                "raw_result": result,
            },
        )
        self._db.add(record)
        self._checks_since_retrain += 1

        logger.debug(
            "FreeCAD check: feasible=%s (%.2fs, source=%s)",
            feasible, elapsed, source,
        )
        return record

    # ------------------------------------------------------------------
    # Surrogate prediction
    # ------------------------------------------------------------------

    def predict_surrogate(self, params: dict[str, float]) -> float:
        """Return surrogate feasibility score in [0, 1]."""
        return self._surrogate.predict_score(params)

    # ------------------------------------------------------------------
    # 3-tier evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        params: dict[str, float],
        trial_id: int = -1,
    ) -> tuple[bool, float, str]:
        """3-tier feasibility evaluation.

        Returns:
            (is_feasible, score, tier) where tier is one of:
            - "high_confidence": surrogate score >= tier_high, no FreeCAD check
            - "uncertain_pass": surrogate uncertain, FreeCAD confirmed feasible
            - "uncertain_fail": surrogate uncertain, FreeCAD confirmed infeasible
            - "rejected": surrogate score < tier_uncertain
            - "exploration_bypass_pass": rejected but bypassed, FreeCAD confirmed
            - "exploration_bypass_fail": rejected but bypassed, FreeCAD denied
            - "untrained": surrogate not trained, fell through to FreeCAD
        """
        score = self._surrogate.predict_score(params)

        if not self._surrogate.is_trained:
            # No surrogate yet — always use FreeCAD
            record = self.check_freecad(params, source="optimization_untrained")
            self._maybe_retrain()
            return record.feasible, score, "untrained"

        if score >= self._spec.tier_high_confidence:
            # High confidence: trust surrogate, skip FreeCAD
            return True, score, "high_confidence"

        if score >= self._spec.tier_uncertain:
            # Uncertain: verify with FreeCAD
            record = self.check_freecad(params, source="optimization_uncertain")
            self._maybe_retrain()
            tier = "uncertain_pass" if record.feasible else "uncertain_fail"
            return record.feasible, score, tier

        # Rejected by surrogate
        # Exploration bypass: small chance to verify with FreeCAD anyway
        if random.random() < self._spec.exploration_ratio:
            record = self.check_freecad(params, source="exploration_bypass")
            self._maybe_retrain()
            tier = "exploration_bypass_pass" if record.feasible else "exploration_bypass_fail"
            return record.feasible, score, tier

        return False, score, "rejected"

    # ------------------------------------------------------------------
    # Phase 0: Feasibility Discovery
    # ------------------------------------------------------------------

    def run_discovery(self, n_points: Optional[int] = None) -> dict[str, Any]:
        """Phase 0: Sample widely and check FreeCAD feasibility.

        Generates Latin Hypercube samples across the parameter space
        (with margins beyond config bounds) and runs FreeCAD --check-only
        for each point. Results populate the FeasibilityDB and are used
        to train the initial surrogate.

        Args:
            n_points: Number of points to sample. Defaults to spec.discovery_points.

        Returns:
            Summary dict with counts and timing.
        """
        n = n_points or self._spec.discovery_points
        if n <= 0:
            logger.info("Phase 0 discovery skipped (n_points=0)")
            return {"n_points": 0, "feasible": 0, "infeasible": 0}

        # Tier split counts (match _generate_lhs_samples)
        n_a = max(1, int(n * 0.30))
        n_b = max(1, int(n * 0.40))
        n_c = max(1, n - n_a - n_b)

        logger.info(
            "Phase 0: Feasibility discovery starting (%d points: tier_A=%d, tier_B=%d, tier_C=%d)",
            n, n_a, n_b, n_c,
        )
        t0 = time.time()

        # Generate samples with expanded bounds
        margin = self._spec.discovery_bounds_margin
        samples = self._generate_lhs_samples(n, margin)

        feasible_count = 0
        infeasible_count = 0
        for i, params in enumerate(samples):
            record = self.check_freecad(params, source="discovery")
            if record.feasible:
                feasible_count += 1
            else:
                infeasible_count += 1

            if (i + 1) % 20 == 0 or (i + 1) == n:
                rate = feasible_count / (i + 1) * 100
                logger.info(
                    "Phase 0 progress: %d/%d checked (feasible=%d, infeasible=%d, rate=%.1f%%)",
                    i + 1, n, feasible_count, infeasible_count, rate,
                )

        elapsed = time.time() - t0

        # Train initial surrogate
        self._surrogate.train(self._db.records)
        self._checks_since_retrain = 0
        self._db.save()

        summary = {
            "n_points": n,
            "feasible": feasible_count,
            "infeasible": infeasible_count,
            "feasible_rate": feasible_count / n if n > 0 else 0.0,
            "elapsed_sec": round(elapsed, 1),
            "surrogate_trained": self._surrogate.is_trained,
            "tier_split": {"A_few_dim": n_a, "B_mid_dim": n_b, "C_full_dim": n_c},
        }
        logger.info(
            "Phase 0 complete: %d points in %.1fs (%.1f%% feasible). Surrogate trained=%s",
            n, elapsed, summary["feasible_rate"] * 100, summary["surrogate_trained"],
        )
        return summary

    def _generate_lhs_samples(
        self,
        n: int,
        margin: float,
    ) -> list[dict[str, float]]:
        """Generate progressive-perturbation samples for discovery.

        Naive LHS across all 20 dimensions produces 0% feasibility due to
        parameter interaction effects.  Instead we use a 3-tier strategy:

          Tier A (~30%): perturb 1–3 dimensions, others at baseline (ratio=1.0)
                         → high feasibility rate, maps near-baseline manifold
          Tier B (~40%): perturb 5–10 dimensions
                         → medium feasibility, explores pairwise interactions
          Tier C (~30%): perturb all dimensions
                         → low feasibility, maps the infeasible boundary

        This guarantees both positive and negative examples for surrogate
        training even in high-dimensional parameter spaces with strong
        confounding interactions.
        """
        n_dims = len(self._bounds)
        rng = np.random.default_rng(seed=42)

        # Build ratio-space bounds
        ratio_lo = np.empty(n_dims, dtype=np.float64)
        ratio_hi = np.empty(n_dims, dtype=np.float64)
        for i, b in enumerate(self._bounds):
            rlo = b.min / b.base_value if b.base_value != 0 else b.min
            rhi = b.max / b.base_value if b.base_value != 0 else b.max
            span = rhi - rlo
            ratio_lo[i] = rlo - margin * span
            ratio_hi[i] = rhi + margin * span
            if ratio_lo[i] <= 0:
                ratio_lo[i] = rlo * 0.5

        # Tier split counts
        n_a = max(1, int(n * 0.30))   # few-dim perturbation
        n_b = max(1, int(n * 0.40))   # mid-dim perturbation
        n_c = max(1, n - n_a - n_b)   # full-dim perturbation

        baseline = np.ones(n_dims, dtype=np.float64)
        result: list[dict[str, float]] = []

        # --- Tier A: perturb 1–3 dims ---
        for _ in range(n_a):
            k = rng.integers(1, min(4, n_dims + 1))
            dims = rng.choice(n_dims, size=k, replace=False)
            sample = baseline.copy()
            for d in dims:
                sample[d] = rng.uniform(ratio_lo[d], ratio_hi[d])
            result.append(self._array_to_params(sample))

        # --- Tier B: perturb 5–10 dims ---
        for _ in range(n_b):
            k = rng.integers(5, min(11, n_dims + 1))
            dims = rng.choice(n_dims, size=k, replace=False)
            sample = baseline.copy()
            for d in dims:
                sample[d] = rng.uniform(ratio_lo[d], ratio_hi[d])
            result.append(self._array_to_params(sample))

        # --- Tier C: perturb all dims (classic LHS) ---
        lhs_unit = np.empty((n_c, n_dims), dtype=np.float64)
        for j in range(n_dims):
            perm = rng.permutation(n_c)
            for i in range(n_c):
                lhs_unit[i, j] = (perm[i] + rng.random()) / n_c
        lhs_scaled = ratio_lo + lhs_unit * (ratio_hi - ratio_lo)
        for i in range(n_c):
            result.append(self._array_to_params(lhs_scaled[i]))

        # Shuffle to avoid systematic ordering bias
        rng.shuffle(result)
        return result

    def _array_to_params(self, arr: np.ndarray) -> dict[str, float]:
        """Convert a parameter array to a named dict."""
        return {
            self._param_names[j]: float(arr[j])
            for j in range(len(self._param_names))
        }

    # ------------------------------------------------------------------
    # Retraining
    # ------------------------------------------------------------------

    def _maybe_retrain(self) -> None:
        """Retrain surrogate if enough new data has accumulated."""
        if self._checks_since_retrain >= self._spec.surrogate_retrain_interval:
            self.retrain()

    def retrain(self) -> None:
        """Force retrain the surrogate from all accumulated data."""
        self._surrogate.train(self._db.records)
        self._checks_since_retrain = 0
        self._db.save()
        logger.info(
            "Surrogate retrained with %d total records",
            len(self._db),
        )
