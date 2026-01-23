"""
Proto2 Optimizer Module

Optunaを使用した最適化エンジンラッパー
（Proto1から流用・OGDEN係数用に拡張）
"""

import logging
import warnings
from pathlib import Path
from typing import Callable, Optional, Any

import optuna
from optuna.samplers import TPESampler, RandomSampler, QMCSampler, CmaEsSampler
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

# AutoSampler from OptunaHub
try:
    import optunahub
    _auto_sampler_module = optunahub.load_module(package="samplers/auto_sampler")
    AutoSampler = _auto_sampler_module.AutoSampler
    HAS_AUTO_SAMPLER = True
except Exception:
    HAS_AUTO_SAMPLER = False


logger = logging.getLogger(__name__)


class Optimizer:
    """
    Optunaラッパークラス（Proto2用）
    
    OGDEN係数の最適化に対応
    """
    
    def __init__(
        self,
        bounds: dict,
        config: dict,
        mode: str = "all",
        storage_path: Optional[str | Path] = None
    ):
        """
        Args:
            bounds: 係数ごとの範囲 {"c": [(min,max),...], ...}
            config: 最適化設定（optimizer_config.yamlのoptimizationセクション）
            mode: "elastic_only" | "visco_only" | "all"
            storage_path: Optunaストレージパス
        """
        self.bounds = bounds
        self.config = config
        self.mode = mode
        self._study: Optional[Study] = None
        self._current_trial: Optional[Trial] = None

        # 離散化使用時はOptunaの冗長な警告を要約して表示
        # "range is not divisible by step" という警告が大量に出るのを防ぐ
        step = self.config.get("discretization_step")
        if step is not None:
            warnings.filterwarnings(
                "ignore", 
                message=".*range is not divisible by.*",
                category=UserWarning,
                module="optuna.distributions"
            )
            # 要約ログを一回だけ出力
            logger.info(f"Discretization error for step={step} is suppressedgit push")
        
        # ストレージ設定
        if storage_path:
            storage_path = Path(storage_path)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage = f"sqlite:///{storage_path}"
        else:
            self._storage = None

        # サンプラー選択
        sampler_name = config.get("sampler", "TPE").upper()
        self._sampler = self._create_sampler(sampler_name)
        
        # 最適化方向
        self._directions = config.get("directions", ["minimize"])
        # configにdirectionsがない場合、objective_typeから推測も可能だが、
        # main.pyから明示的に渡されることを想定するか、ここでデフォルトを設定
        
        logger.info(f"Optimizer初期化: sampler={type(self._sampler).__name__}, mode={mode}, directions={self._directions}")
    
    def _create_sampler(self, name: str) -> Any:
        """サンプラーを作成"""
        seed = self.config.get("seed")
        n_startup = self.config.get("n_startup_trials", 10)
        
        # AUTO の場合は OptunaHub の AutoSampler を使用（初期探索も自動）
        if name == "AUTO":
            if HAS_AUTO_SAMPLER:
                logger.info("AutoSamplerを使用します（サンプラー・初期探索は自動選択）")
                return AutoSampler(seed=seed)
            else:
                logger.warning("AutoSamplerが利用不可。TPESamplerを使用します。")
                return TPESampler(seed=seed)
        
        # 現在の試行数を取得して、初期探索期間か判定
        current_trials = 0
        if self._storage:
            try:
                # DB内のスタディ情報を取得
                summaries = optuna.study.get_all_study_summaries(self._storage)
                
                # 指定スタディ名、もしくは最初のスタディを採用
                target_study_name = "proto2_material_optimization"
                found_study = None
                
                for s in summaries:
                    if s.study_name == target_study_name:
                        found_study = s
                        break
                
                # 指定名がない場合、DB内の唯一のスタディであればそれを使う
                if found_study is None and len(summaries) == 1:
                    found_study = summaries[0]
                
                if found_study:
                    current_trials = found_study.n_trials
                    
            except Exception as e:
                logger.warning(f"試行数確認失敗: {e}")
        
        logger.info(f"現在の試行数: {current_trials}, 初期探索設定: {n_startup}")
        
        # 初期探索期間ならSobol列を使用
        if current_trials < n_startup:
            logger.info("初期探索フェーズ: QMCSampler(Sobol)を使用します")
            return QMCSampler(scramble=True, seed=seed)
            
        # 本番サンプラー
        logger.info(f"本番探索フェーズ: {name}Samplerを使用します")
        
        if name == "TPE":
            return TPESampler(seed=seed, n_startup_trials=0) # StartupはSobolでやったので0
        elif name == "GP":
            if HAS_GP_SAMPLER:
                # 独立サンプリング警告を抑制
                return GPSampler(seed=seed, n_startup_trials=0, warn_independent_sampling=False)
            else:
                logger.warning("GPSamplerが利用不可。TPESamplerを使用します。")
                return TPESampler(seed=seed, n_startup_trials=0)
        elif name == "NSGAII":
            if HAS_NSGAII:
                return NSGAIISampler(seed=seed)
            else:
                logger.warning("NSGAIISamplerが利用不可。TPESamplerを使用します。")
                return TPESampler(seed=seed, n_startup_trials=0)
        elif name == "RANDOM":
            return RandomSampler(seed=seed)
        elif name in ["CMA-ES", "CMAES"]:
            logger.info("CmaEsSamplerを使用します（注意: 独立サンプリング警告が出る可能性があります）")
            return CmaEsSampler(seed=seed, n_startup_trials=0)
        else:
            logger.warning(f"未知のサンプラー: {name}。TPESamplerを使用します。")
            return TPESampler(seed=seed, n_startup_trials=0)
    
    def create_study(self, study_name: str = "proto2_optimization") -> Study:
        """Optunaスタディを作成"""
        self._study = optuna.create_study(
            study_name=study_name,
            sampler=self._sampler,
            directions=self._directions,
            storage=self._storage,
            load_if_exists=True
        )
        
        logger.info(f"スタディ作成: {study_name}")
        return self._study
    
    def suggest_ogden_params(self, trial: Trial) -> dict:
        """
        OGDEN係数を提案
        
        Args:
            trial: Optunaトライアル
        
        Returns:
            {"c": [...], "m": [...], "t": [...], "g": [...]}
        """
        self._current_trial = trial
        params = {}
        
        # 対象係数を決定
        if self.mode == "elastic_only":
            target_keys = ["c", "m"]
        elif self.mode == "visco_only":
            target_keys = ["t", "g"]
        else:
            target_keys = ["c", "m", "t", "g"]
        
        for key in ["c", "m", "t", "g"]:
            key_bounds = self.bounds.get(key, [])
            values = []
            
            for i, bound in enumerate(key_bounds):
                if key in target_keys and bound is not None:
                    # 最適化対象
                    low, high = bound
                    val = trial.suggest_float(f"{key}_{i}", low, high)
                    values.append(val)
                else:
                    # 固定値（boundsに格納されていないので元の値を使用）
                    # boundsがNoneの場合は0.0
                    values.append(0.0)
            
            params[key] = values
        
        return params
    
    def run_optimization(
        self,
        objective_func: Callable[[dict], float],
        base_params: dict,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        callbacks: Optional[list] = None
    ) -> Study:
        """
        最適化を実行
        
        Args:
            objective_func: 目的関数（パラメータ辞書 -> 目的関数値）
            base_params: 固定値用のベースパラメータ
            n_trials: 試行回数
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
        
        # 対象係数を決定
        if self.mode == "elastic_only":
            target_keys = ["c", "m"]
        elif self.mode == "visco_only":
            target_keys = ["t", "g"]
        else:
            target_keys = ["c", "m", "t", "g"]
        
        def wrapped_objective(trial: Trial) -> float:
            # OGDEN係数を提案
            suggested = {}
            
            # 離散化ステップ（設定から取得、デフォルトはNone=連続値）
            step = self.config.get("discretization_step")
            
            for key in ["c", "m", "t", "g"]:
                key_bounds = self.bounds.get(key, [])
                base_values = base_params.get(key, [])
                values = []
                
                for i, bound in enumerate(key_bounds):
                    if key in target_keys and bound is not None:
                        low, high = bound
                        if step is not None:
                            # 離散化: stepの倍数に丸める
                            val = trial.suggest_float(f"{key}_{i}", low, high, step=step)
                        else:
                            val = trial.suggest_float(f"{key}_{i}", low, high)
                        values.append(val)
                    else:
                        # 固定値（ベースパラメータから取得）
                        if i < len(base_values):
                            values.append(base_values[i])
                        else:
                            values.append(0.0)
                
                suggested[key] = values
            
            return objective_func(suggested)
        
        logger.info(f"最適化開始: 合計試行数={n_trials}, 残り試行数={remaining_trials}")
        
        
        self._study.optimize(
            wrapped_objective,
            n_trials=remaining_trials,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=False
        )
        
        logger.info("最適化完了")
        return self._study
    
    def get_best_params(self, base_params: dict) -> Optional[dict]:
        """
        最良パラメータを取得
        
        Args:
            base_params: 固定値用のベースパラメータ
        
        Returns:
            完全なパラメータ辞書
        """
        if self._study is None:
            return None
        
        try:
            best_trial_params = self._study.best_params
        except ValueError:
            return None
        except RuntimeError:
            # 多目的最適化の場合、best_paramsは定義されない
            # パレートフロントの最初の解を返すか、Noneを返す
            # ここでは「パレート解の1つ」として、0番目のトライアルを返す（簡易的な対応）
            try:
                best_trials = self._study.best_trials
                if best_trials:
                    best_trial_params = best_trials[0].params
                else:
                    return None
            except Exception:
                return None
        
        # 対象係数を決定
        if self.mode == "elastic_only":
            target_keys = ["c", "m"]
        elif self.mode == "visco_only":
            target_keys = ["t", "g"]
        else:
            target_keys = ["c", "m", "t", "g"]
        
        # パラメータを再構築
        result = {}
        for key in ["c", "m", "t", "g"]:
            key_bounds = self.bounds.get(key, [])
            base_values = base_params.get(key, [])
            values = []
            
            for i, bound in enumerate(key_bounds):
                param_name = f"{key}_{i}"
                if key in target_keys and bound is not None and param_name in best_trial_params:
                    values.append(best_trial_params[param_name])
                else:
                    if i < len(base_values):
                        values.append(base_values[i])
                    else:
                        values.append(0.0)
            
            result[key] = values
        
        return result
    
    def get_best_value(self) -> Optional[float]:
        """最良目的関数値を取得"""
        if self._study is None:
            return None
        
        try:
            return self._study.best_value
        except ValueError:
            return None
        except RuntimeError:
            # 多目的最適化の場合
            try:
                best_trials = self._study.best_trials
                if best_trials:
                    # 最初の解の値を返す（タプル）
                    return best_trials[0].values
                else:
                    return None
            except Exception:
                return None
    
    def get_n_trials(self) -> int:
        """完了した試行数を取得"""
        if self._study is None:
            return 0
        return len(self._study.trials)
    
    def is_converged(self, threshold: float) -> bool:
        """収束判定"""
        best_value = self.get_best_value()
        if best_value is None:
            return False
        return best_value <= threshold
    
    def get_study_summary(self, base_params: dict) -> dict:
        """スタディのサマリを取得"""
        if self._study is None:
            return {}
        
        return {
            "n_trials": self.get_n_trials(),
            "best_params": self.get_best_params(base_params),
            "best_value": self.get_best_value(),
            "sampler": type(self._sampler).__name__,
            "optimization_mode": self.mode
        }


class ConvergenceCallback:
    """収束時に最適化を停止するコールバック"""
    
    def __init__(self, threshold: float, patience: int = 20):
        """
        Args:
            threshold: この値以下で収束とみなす
            patience: 改善がない試行回数がこれを超えると停止（デフォルト20）
        """
        self.threshold = threshold
        self.patience = patience
        self._best_value = float("inf")
        self._no_improvement_count = 0
    
    def __call__(self, study: Study, trial: optuna.trial.FrozenTrial) -> None:
        # 多目的最適化対応
        try:
            current_value = trial.value
        except RuntimeError:
            # 多目的の場合、trial.valueはエラーになる
            # 収束判定は第一目的関数（RMSE）のみで行うことにする
            if trial.values:
                current_value = trial.values[0]
            else:
                return

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
