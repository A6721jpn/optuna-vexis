"""
Proto1 Optimizer Module

Optunaを使用した最適化エンジンラッパー
- TPESampler: デフォルト、効率的な探索
- GPSampler: ガウス過程ベイズ最適化
- NSGAIISampler: 多目的最適化
"""

import logging
from pathlib import Path
from typing import Callable, Optional, Any

import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.study import Study
from optuna.trial import Trial

# GPSamplerは実験的機能
try:
    from optuna.samplers import GPSampler
    HAS_GP_SAMPLER = True
except ImportError:
    HAS_GP_SAMPLER = False

try:
    from optuna.samplers import NSGAIISampler
    HAS_NSGAII = True
except ImportError:
    HAS_NSGAII = False


logger = logging.getLogger(__name__)


class Optimizer:
    """
    Optunaラッパークラス
    
    設定に基づいてサンプラーを選択し、最適化ループを制御
    """
    
    def __init__(
        self,
        dimensions: list[dict],
        config: dict,
        storage_path: Optional[str | Path] = None
    ):
        """
        Args:
            dimensions: 寸法定義リスト（dimensions.yamlから）
            config: 最適化設定（optimizer_config.yamlのoptimizationセクション）
            storage_path: Optunaストレージパス（省略時はインメモリ）
        """
        self.dimensions = dimensions
        self.config = config
        self._study: Optional[Study] = None
        self._current_trial: Optional[Trial] = None
        
        # サンプラー選択
        sampler_name = config.get("sampler", "TPE").upper()
        self._sampler = self._create_sampler(sampler_name)
        
        # ストレージ設定
        if storage_path:
            storage_path = Path(storage_path)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage = f"sqlite:///{storage_path}"
        else:
            self._storage = None
        
        # スタディ方向（単一目的 vs 多目的）
        objective_type = config.get("objective_type", "single")
        if objective_type == "multi":
            # 多目的: RMSE + 特徴量誤差
            self._directions = ["minimize", "minimize"]  # 拡張可能
        else:
            self._directions = ["minimize"]
        
        logger.info(f"Optimizer初期化: sampler={sampler_name}, directions={self._directions}")
    
    def _create_sampler(self, name: str) -> Any:
        """サンプラーを作成"""
        seed = self.config.get("seed")
        
        if name == "TPE":
            return TPESampler(seed=seed)
        elif name == "GP":
            if HAS_GP_SAMPLER:
                return GPSampler(seed=seed)
            else:
                logger.warning("GPSamplerが利用不可。TPESamplerを使用します。")
                return TPESampler(seed=seed)
        elif name == "NSGAII":
            if HAS_NSGAII:
                return NSGAIISampler(seed=seed)
            else:
                logger.warning("NSGAIISamplerが利用不可。TPESamplerを使用します。")
                return TPESampler(seed=seed)
        elif name == "RANDOM":
            return RandomSampler(seed=seed)
        else:
            logger.warning(f"未知のサンプラー: {name}。TPESamplerを使用します。")
            return TPESampler(seed=seed)
    
    def create_study(self, study_name: str = "proto1_optimization") -> Study:
        """
        Optunaスタディを作成
        
        Args:
            study_name: スタディ名
        
        Returns:
            作成されたStudy
        """
        if len(self._directions) == 1:
            self._study = optuna.create_study(
                study_name=study_name,
                sampler=self._sampler,
                direction=self._directions[0],
                storage=self._storage,
                load_if_exists=True
            )
        else:
            self._study = optuna.create_study(
                study_name=study_name,
                sampler=self._sampler,
                directions=self._directions,
                storage=self._storage,
                load_if_exists=True
            )
        
        logger.info(f"スタディ作成: {study_name}")
        return self._study
    
    def suggest_params(self, trial: Trial) -> dict[str, float]:
        """
        次の試行パラメータを提案
        
        Args:
            trial: Optunaトライアル
        
        Returns:
            {寸法名: 値} の辞書
        """
        self._current_trial = trial
        params = {}
        
        for dim in self.dimensions:
            name = dim["name"]
            dim_type = dim.get("type", "float")
            min_val = dim["min"]
            max_val = dim["max"]
            
            if dim_type == "float":
                step = dim.get("step")
                if step:
                    params[name] = trial.suggest_float(name, min_val, max_val, step=step)
                else:
                    params[name] = trial.suggest_float(name, min_val, max_val)
            elif dim_type == "int":
                params[name] = trial.suggest_int(name, int(min_val), int(max_val))
            elif dim_type == "categorical":
                choices = dim.get("choices", [min_val, max_val])
                params[name] = trial.suggest_categorical(name, choices)
            else:
                logger.warning(f"未対応の寸法タイプ: {dim_type}")
        
        logger.info(f"Trial {trial.number}: 提案パラメータ = {params}")
        return params
    
    def run_optimization(
        self,
        objective_func: Callable[[dict[str, float]], float | tuple[float, ...]],
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        callbacks: Optional[list] = None
    ) -> Study:
        """
        最適化を実行
        
        Args:
            objective_func: 目的関数（パラメータ辞書 -> 目的関数値）
            n_trials: 試行回数（省略時は設定ファイルから）
            timeout: タイムアウト秒
            callbacks: コールバックリスト
        
        Returns:
            最適化済みStudy
        """
        if self._study is None:
            self.create_study()
        
        if n_trials is None:
            n_trials = self.config.get("max_trials", 100)
        
        # 既存の試行数を差し引いて、残りの試行数を決定
        completed_trials = self.get_n_trials()
        remaining_trials = n_trials - completed_trials
        
        if remaining_trials <= 0:
            logger.info(f"既に {completed_trials} 試行完了しています。追加の試行は行いません（max_trials={n_trials}）")
            return self._study

        def wrapped_objective(trial: Trial) -> float | tuple[float, ...]:
            params = self.suggest_params(trial)
            return objective_func(params)
        
        logger.info(f"最適化開始: 合計試行数={n_trials}, 残り試行数={remaining_trials}")
        
        self._study.optimize(
            wrapped_objective,
            n_trials=remaining_trials,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=True
        )
        
        logger.info("最適化完了")
        return self._study
    
    def get_best_params(self) -> Optional[dict]:
        """最良パラメータを取得"""
        if self._study is None:
            return None
        
        try:
            if len(self._directions) == 1:
                return self._study.best_params
            else:
                # 多目的の場合、パレートフロントの最初の解
                best_trials = self._study.best_trials
                if best_trials:
                    return best_trials[0].params
                return None
        except ValueError:
            return None
    
    def get_best_value(self) -> Optional[float]:
        """最良目的関数値を取得"""
        if self._study is None:
            return None
        
        try:
            if len(self._directions) == 1:
                return self._study.best_value
            else:
                best_trials = self._study.best_trials
                if best_trials:
                    return best_trials[0].values[0]
                return None
        except ValueError:
            return None
    
    def get_n_trials(self) -> int:
        """完了した試行数を取得"""
        if self._study is None:
            return 0
        return len(self._study.trials)
    
    def is_converged(self, threshold: float) -> bool:
        """
        収束判定
        
        Args:
            threshold: 収束閾値
        
        Returns:
            収束していればTrue
        """
        best_value = self.get_best_value()
        if best_value is None:
            return False
        return best_value <= threshold
    
    def get_study_summary(self) -> dict:
        """スタディのサマリを取得"""
        if self._study is None:
            return {}
        
        return {
            "n_trials": self.get_n_trials(),
            "best_params": self.get_best_params(),
            "best_value": self.get_best_value(),
            "sampler": type(self._sampler).__name__
        }


class ConvergenceCallback:
    """
    収束時に最適化を停止するコールバック
    """
    
    def __init__(self, threshold: float, patience: int = 5):
        """
        Args:
            threshold: 収束閾値
            patience: 改善がない場合の許容試行数
        """
        self.threshold = threshold
        self.patience = patience
        self._best_value = float("inf")
        self._no_improvement_count = 0
    
    def __call__(self, study: Study, trial: optuna.trial.FrozenTrial) -> None:
        """コールバック実行"""
        current_value = trial.value
        
        if current_value is None:
            return
        
        # 収束判定
        if current_value <= self.threshold:
            logger.info(f"収束達成: {current_value:.6f} <= {self.threshold}")
            study.stop()
            return
        
        # 改善判定
        if current_value < self._best_value:
            self._best_value = current_value
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1
        
        # Patience超過
        if self._no_improvement_count >= self.patience:
            logger.info(f"改善なし {self.patience} 試行。最適化停止。")
            study.stop()
