"""
Proto1 Main Module

VEXIS CAE自動最適化のエントリポイント

Usage:
    python -m src.proto1.main --config config/optimizer_config.yaml
"""

import argparse
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .utils import (
    setup_logger,
    load_yaml,
    save_json,
    get_project_root,
    ensure_dir,
    TrialLogger,
    generate_trial_id
)
from .step_editor import StepEditor, edit_step_file
from .vexis_runner import VexisRunner
from .result_loader import ResultLoader
from .objective import ObjectiveCalculator, calculate_rmse
from .optimizer import Optimizer, ConvergenceCallback


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description="Proto1: VEXIS CAE自動最適化プロトタイプ"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/optimizer_config.yaml",
        help="最適化設定ファイルのパス"
    )
    parser.add_argument(
        "--dimensions", "-d",
        type=str,
        default="config/dimensions.yaml",
        help="寸法定義ファイルのパス"
    )
    parser.add_argument(
        "--max-trials", "-n",
        type=int,
        default=None,
        help="最大試行回数（設定ファイルを上書き）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="CAE実行をスキップ（テスト用）"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細ログ出力"
    )
    return parser.parse_args()


def load_configs(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    """設定ファイルを読込"""
    project_root = get_project_root()
    
    config_path = project_root / args.config
    dim_path = project_root / args.dimensions
    
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    if not dim_path.exists():
        raise FileNotFoundError(f"寸法ファイルが見つかりません: {dim_path}")
    
    config = load_yaml(config_path)
    dim_config = load_yaml(dim_path)
    
    return config, dim_config.get("dimensions", [])


def find_input_step(config: dict) -> Path:
    """入力STEPファイルを探す"""
    project_root = get_project_root()
    input_dir = project_root / config.get("paths", {}).get("input_dir", "input")
    
    # STEPファイルを検索
    step_files = list(input_dir.glob("*.step")) + list(input_dir.glob("*.stp"))
    
    if not step_files:
        raise FileNotFoundError(f"入力STEPファイルが見つかりません: {input_dir}")
    
    # 最初のファイルを使用
    return step_files[0]


def main() -> int:
    """メイン関数"""
    args = parse_args()
    project_root = get_project_root()
    
    # ロガーセットアップ
    log_level = "DEBUG" if args.verbose else "INFO"
    log_dir = project_root / "output" / "logs"
    logger = setup_logger("proto1", str(log_dir), level=log_level)
    
    logger.info("=" * 60)
    logger.info("Proto1 最適化開始")
    logger.info("=" * 60)
    
    try:
        # 設定読込
        config, dimensions = load_configs(args)
        opt_config = config.get("optimization", {})
        obj_config = config.get("objective", {})
        paths_config = config.get("paths", {})
        
        logger.info(f"設定読込完了: {len(dimensions)}個の寸法変数")
        
        # 最大試行回数
        max_trials = args.max_trials or opt_config.get("max_trials", 100)
        convergence_threshold = opt_config.get("convergence_threshold", 0.01)
        
        # 入力ファイル
        input_step = find_input_step(config)
        logger.info(f"入力STEP: {input_step}")
        
        # ターゲットカーブ読込
        target_path = project_root / paths_config.get("target_curve", "tgt/target_curve.csv")
        result_loader = ResultLoader()
        target_curve = result_loader.load_target(target_path)
        
        # ターゲット特徴量抽出
        feature_config = obj_config.get("features", {})
        target_features = result_loader.extract_features(target_curve, feature_config)
        logger.info(f"ターゲット特徴量: {target_features}")
        
        # 目的関数計算器
        obj_calculator = ObjectiveCalculator(target_curve, target_features, obj_config)
        
        # VEXISランナー
        vexis_path = project_root / paths_config.get("vexis_path", "vexis")
        vexis_runner = VexisRunner(vexis_path)
        
        # 出力ディレクトリ
        result_dir = project_root / paths_config.get("result_dir", "output")
        temp_dir = project_root / "temp"
        ensure_dir(result_dir)
        ensure_dir(temp_dir)
        
        # Optimizer作成
        optimizer = Optimizer(
            dimensions=dimensions,
            config=opt_config,
            storage_path=result_dir / "optuna_study.db"
        )
        optimizer.create_study()
        
        # 収束コールバック
        callbacks = [ConvergenceCallback(convergence_threshold)]
        
        # 試行カウンタ（既存の試行数から開始）
        trial_count = optimizer.get_n_trials()
        
        def objective_function(params: dict[str, float]) -> float:
            """目的関数（Optunaから呼び出される）"""
            nonlocal trial_count
            trial_count += 1
            
            trial_id = generate_trial_id()
            trial_logger = TrialLogger(trial_id, str(log_dir))
            trial_logger.log_parameters(params)
            
            logger.info(f"--- Trial {trial_count} ---")
            logger.info(f"パラメータ: {params}")
            
            try:
                # 1. STEPファイル編集
                job_name = f"trial_{trial_id}"
                edited_step = temp_dir / f"{job_name}.step"
                
                if not args.dry_run:
                    edit_step_file(input_step, edited_step, params, dimensions)
                    logger.info(f"STEP編集完了: {edited_step}")
                else:
                    # ドライラン: 元ファイルをコピー
                    import shutil
                    shutil.copy(input_step, edited_step)
                    logger.info(f"[DRY-RUN] STEPコピー: {edited_step}")
                
                # 2. CAE解析実行
                if not args.dry_run:
                    result_csv = vexis_runner.run_analysis(
                        edited_step,
                        job_name,
                        log_path=trial_logger.get_log_path("vexis")
                    )
                    
                    if result_csv is None:
                        logger.error("CAE解析失敗")
                        trial_logger.log_error("CAE解析失敗")
                        return float("inf")
                else:
                    # ドライラン: ダミー結果
                    logger.info("[DRY-RUN] CAE解析スキップ")
                    # ターゲットにノイズを加えたダミー結果を生成
                    import numpy as np
                    import pandas as pd
                    noise = np.random.normal(0, 0.1, len(target_curve))
                    dummy_result = target_curve.copy()
                    dummy_result["force"] = dummy_result["force"] + noise
                    
                    result_csv = temp_dir / f"{job_name}_result.csv"
                    dummy_result.to_csv(result_csv, index=False)
                
                # 3. 結果読込
                result_curve = result_loader.load_curve(result_csv)
                result_features = result_loader.extract_features(result_curve, feature_config)
                logger.info(f"結果特徴量: {result_features}")
                
                # 4. 目的関数計算
                objectives = obj_calculator.evaluate(result_curve, result_features)
                trial_logger.log_objectives(objectives)
                
                # 重み付きスコアまたはRMSEを返す
                obj_type = obj_config.get("type", "rmse")
                if obj_type == "multi":
                    return objectives.get("weighted_score", objectives["rmse"])
                else:
                    return objectives["rmse"]
                
            except Exception as e:
                logger.error(f"Trial {trial_count} エラー: {e}")
                trial_logger.log_error(str(e))
                return float("inf")
        
        # 最適化実行
        logger.info(f"最適化開始: max_trials={max_trials}, threshold={convergence_threshold}")
        
        optimizer.run_optimization(
            objective_func=objective_function,
            n_trials=max_trials,
            callbacks=callbacks
        )
        
        # 結果サマリ
        summary = optimizer.get_study_summary()
        summary["start_time"] = datetime.now().isoformat()
        summary["convergence_achieved"] = optimizer.is_converged(convergence_threshold)
        
        # サマリ保存
        save_json(summary, result_dir / "summary.json")
        
        logger.info("=" * 60)
        logger.info("最適化完了")
        logger.info(f"試行回数: {summary['n_trials']}")
        logger.info(f"最良パラメータ: {summary['best_params']}")
        logger.info(f"最良目的関数値: {summary['best_value']}")
        logger.info(f"収束達成: {summary['convergence_achieved']}")
        logger.info("=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"ファイルエラー: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("ユーザーによる中断 (Ctrl+C)")
        if 'vexis_runner' in dir():
            vexis_runner.request_stop()
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.exception(f"予期しないエラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
