"""
Proto2 Main Module

OGDEN材料モデル係数自動最適化のエントリポイント

Usage:
    # 全係数最適化（デフォルト）
    python -m src.proto2.main --config config/optimizer_config.yaml

    # 弾性係数のみ（c, m）
    python -m src.proto2.main --config config/optimizer_config.yaml --mode elastic_only

    # 粘弾性係数のみ（t, g）
    python -m src.proto2.main --config config/optimizer_config.yaml --mode visco_only
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import optuna

from .utils import (
    setup_logger,
    load_yaml,
    save_json,
    get_project_root,
    ensure_dir,
    TrialLogger,
    generate_trial_id,
    format_params_for_log
)
from .material_editor import MaterialEditor, save_optimized_material
from .curve_processor import CurveProcessor
from .vexis_runner import VexisRunner
from .result_loader import ResultLoader
from .result_loader import ResultLoader
from .objective import ObjectiveCalculator, FatalOptimizationError
from .optimizer import Optimizer, ConvergenceCallback
from .visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description="Proto2: OGDEN材料モデル係数自動最適化"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/optimizer_config.yaml",
        help="最適化設定ファイルのパス"
    )
    parser.add_argument(
        "--limits", "-l",
        type=str,
        default="config/proto2_limitations.yaml",
        help="係数範囲設定ファイルのパス"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["elastic_only", "visco_only", "all"],
        default=None,
        help="最適化モード（設定ファイルを上書き）"
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="既存のStudyを継続する（指定しない場合は新規作成）"
    )
    return parser.parse_args()


def load_configs(args: argparse.Namespace) -> tuple[dict, dict]:
    """設定ファイルを読込"""
    project_root = get_project_root()
    
    config_path = project_root / args.config
    limits_path = project_root / args.limits
    
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    if not limits_path.exists():
        raise FileNotFoundError(f"制限ファイルが見つかりません: {limits_path}")
    
    config = load_yaml(config_path)
    limits = load_yaml(limits_path)
    
    return config, limits


def find_input_step(limits: dict, project_root: Path) -> Path:
    """入力STEPファイルを取得"""
    input_step_name = limits.get("input_step", "example_1.stp")
    input_dir = project_root / "input"
    
    # 指定されたファイルを探す
    step_path = input_dir / input_step_name
    if step_path.exists():
        return step_path
    
    # 見つからない場合は最初のSTEPファイルを使用
    step_files = list(input_dir.glob("*.step")) + list(input_dir.glob("*.stp"))
    
    if not step_files:
        raise FileNotFoundError(f"入力STEPファイルが見つかりません: {input_dir}")
    
    return step_files[0]


def main() -> int:
    """メイン関数"""
    args = parse_args()
    project_root = get_project_root()
    
    # ロガーセットアップ
    log_level = "DEBUG" if args.verbose else "INFO"
    log_dir = project_root / "output" / "logs"
    logger = setup_logger("proto2", str(log_dir), level=log_level)
    
    # Optunaのデフォルトログを抑制（カスタムログを見やすくするため）
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)
    
    logger.info("=" * 60)
    logger.info("Proto2 材料モデル係数最適化開始")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # 設定読込
        config, limits = load_configs(args)
        opt_config = config.get("optimization", {})
        obj_config = config.get("objective", {})
        paths_config = config.get("paths", {})
        
        # 最適化モード（コマンドラインオプション優先）
        opt_mode = args.mode or limits.get("optimization_mode", "all")
        logger.info(f"最適化モード: {opt_mode}")
        
        # 最大試行回数
        max_trials = args.max_trials or opt_config.get("max_trials", 100)
        convergence_threshold = opt_config.get("convergence_threshold", 0.01)
        
        # 係数範囲設定
        range_percent = limits.get("range_percent", 60)
        base_material_name = limits.get("base_material", "Ogden_Rubber_v1")
        min_nonzero = limits.get("constraints", {}).get("min_nonzero", 0.001)
        
        # CAE解析範囲
        cae_range = limits.get("cae_stroke_range", {})
        stroke_min = cae_range.get("min", 0.0)
        stroke_max = cae_range.get("max", 0.5)
        
        logger.info(f"基準材料: {base_material_name}")
        logger.info(f"係数範囲: ±{range_percent}%")
        logger.info(f"CAE範囲: [{stroke_min}, {stroke_max}] mm")
        
        # パス設定
        vexis_path = project_root / paths_config.get("vexis_path", "vexis")
        result_dir = project_root / paths_config.get("result_dir", "output")
        ensure_dir(result_dir)
        
        # Optunaストレージパス
        storage_path = result_dir / "optuna_study_proto2.db"
        
        # 新規実行の場合は既存DBをバックアップ
        if not args.resume and storage_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = result_dir / f"optuna_study_proto2_backup_{timestamp}.db"
            try:
                storage_path.rename(backup_path)
                logger.info(f"既存の学習DBをバックアップしました: {backup_path.name}")
                logger.info("新規Studyとして開始します")
            except Exception as e:
                logger.warning(f"DBバックアップ失敗: {e}")
        elif args.resume and storage_path.exists():
             logger.info(f"既存の学習DBから継続します: {storage_path.name}")

        # MaterialEditor初期化
        material_editor = MaterialEditor(
            vexis_path / "config" / "material.yaml",
            vexis_path / "config" / "config.yaml"
        )
        
        # 基準OGDEN係数を取得
        base_params = material_editor.get_ogden_params(base_material_name)
        logger.info(f"基準OGDEN係数:\n{format_params_for_log(base_params)}") #ここ手で変更した
        
        # 係数範囲を計算
        bounds = material_editor.calculate_bounds(
            base_params, range_percent, min_nonzero, opt_mode
        )
        
        # ターゲットカーブ読込・処理
        target_path = project_root / paths_config.get("target_curve", "input/example_2_tgt.csv")
        curve_processor = CurveProcessor()
        target_curve = curve_processor.process_target_curve(
            target_path,
            (stroke_min, stroke_max),
            use_polynomial=False,  # 多項式近似は将来実装
            num_points=100
        )
        
        # 特徴量抽出（multiモードの場合のみ）
        result_loader = ResultLoader()
        obj_type = obj_config.get("type", "rmse")
        use_features = obj_type == "multi"
        feature_config = obj_config.get("features", {}) if use_features else {}
        target_features = result_loader.extract_features(target_curve, feature_config) if use_features else {}
        if use_features:
            logger.info(f"ターゲット特徴量: {target_features}")
        else:
            logger.info("目的関数: RMSEのみ（特徴量抽出OFF）")
        
        # 目的関数計算器
        obj_calculator = ObjectiveCalculator(target_curve, target_features, obj_config)
        
        # VEXISランナー
        vexis_runner = VexisRunner(vexis_path)
        
        # 入力STEPファイル
        input_step = find_input_step(limits, project_root)
        logger.info(f"入力STEP: {input_step}")
        
        # STEPファイルをVEXISのinputにコピー
        job_base_name = "proto2_trial"
        vexis_runner.setup_input_step(input_step, job_base_name)
        
        # Optimizer作成
        optimizer = Optimizer(
            bounds=bounds,
            config=opt_config,
            mode=opt_mode,
            storage_path=storage_path
        )
        optimizer.create_study("proto2_material_optimization")
        
        # 収束コールバック
        callbacks = [ConvergenceCallback(convergence_threshold)]
        
        # 試行カウンタ
        trial_count = 0
        
        def objective_function(params: dict) -> float:
            """目的関数（Optunaから呼び出される）"""
            nonlocal trial_count
            
            trial_id = generate_trial_id()
            trial_logger = TrialLogger(trial_id, str(log_dir))
            trial_logger.log_parameters(params)
            
            logger.info(f"\n\n{'='*20} Trial {trial_count} {'='*20}\n")
            logger.info(f"OGDEN係数 (Input):\n{format_params_for_log(params)}")
            
            current_trial_num = trial_count  # 現在の試行番号を保存
            trial_count += 1
            
            try:
                # 1. 試行用材料をmaterial.yamlに追記
                material_name = material_editor.add_trial_material(
                    current_trial_num, params, base_material_name
                )
                
                # 2. config.yamlのmaterial_nameを更新
                material_editor.update_config_material(material_name)
                
                # 3. CAE解析実行
                if not args.dry_run:
                    result_csv = vexis_runner.run_analysis(
                        job_base_name,
                        log_path=trial_logger.get_log_path("vexis")
                    )
                    
                    if result_csv is None:
                        logger.error("CAE解析失敗")
                        trial_logger.log_error("CAE解析失敗")
                        return float("inf")
                else:
                    # ドライラン: ダミー結果
                    logger.info("[DRY-RUN] CAE解析スキップ")
                    noise = np.random.normal(0, 0.02, len(target_curve))
                    dummy_result = target_curve.copy()
                    dummy_result["force"] = dummy_result["force"] + noise
                    
                    result_csv = project_root / "temp" / f"{job_base_name}_result.csv"
                    ensure_dir(result_csv.parent)
                    dummy_result.to_csv(result_csv, index=False)
                
                # 4. 結果読込
                result_curve = result_loader.load_curve(result_csv)
                
                # CAE範囲に切り出し
                result_curve_trimmed = curve_processor.extract_range(
                    result_curve, stroke_min, stroke_max
                )
                
                # 特徴量抽出（multiモードの場合のみ）
                if use_features:
                    result_features = result_loader.extract_features(result_curve_trimmed, feature_config)
                    logger.info(f"結果特徴量: {result_features}")
                else:
                    result_features = {}
                
                # 5. 目的関数計算
                objectives = obj_calculator.evaluate(result_curve_trimmed, result_features)
                trial_logger.log_objectives(objectives)
                
                rmse = objectives["rmse"]
                logger.info(f"Trial {current_trial_num} Finished. RMSE: {rmse:.6f}")
                
                return rmse
                
            except FatalOptimizationError as e:
                logger.critical(f"致命的なエラーが発生しました: {e}")
                vexis_runner.request_stop()
                sys.exit(1)
            except Exception as e:
                logger.error(f"Trial {trial_count} エラー: {e}")
                trial_logger.log_error(str(e))
                return float("inf")
        
        # 最適化実行
        logger.info(f"最適化開始: max_trials={max_trials}, threshold={convergence_threshold}")
        
        optimizer.run_optimization(
            objective_func=objective_function,
            base_params=base_params,
            n_trials=max_trials,
            callbacks=callbacks
        )
        
        # --- 可視化処理 ---
        logger.info("可視化処理を実行中...")
        visualizer = Visualizer(result_dir / "plots")
        study = optimizer._study
        
        if study:
            # 1. 最適化履歴プロット
            visualizer.plot_optimization_history(study)
            
            # 2. パレートフロント（多目的の場合のみ）
            visualizer.plot_pareto_front(study)
        
        # 結果サマリ
        end_time = datetime.now()
        best_params = optimizer.get_best_params(base_params)
        best_value = optimizer.get_best_value()
        
        summary = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "optimization_mode": opt_mode,
            "base_material": base_material_name,
            "range_percent": range_percent,
            "cae_stroke_range": {"min": stroke_min, "max": stroke_max},
            "total_trials": optimizer.get_n_trials(),
            "best_params": best_params,
            "best_rmse": best_value,
            "convergence_achieved": optimizer.is_converged(convergence_threshold)
        }
        
        # サマリ保存
        save_json(summary, result_dir / "summary_proto2.json")
        
        # 最適化結果をYAML出力
        if best_params:
            base_material_data = material_editor.load_base_material(base_material_name)
            save_optimized_material(
                result_dir / "optimized_material.yaml",
                best_params,
                base_material_data,
                summary
            )
            
            # 最良パラメータで再解析（プロット用）
            logger.info("最良パラメータで確認解析を実行中...")
            try:
                # 試行用材料追加
                material_name = material_editor.add_trial_material(
                    9999, best_params, base_material_name # ID 9999 for verification
                )
                material_editor.update_config_material(material_name)
                
                result_csv = None
                if not args.dry_run:
                    # STEPファイルをセットアップ（ジョブ名が変わるため必須）
                    vexis_runner.setup_input_step(input_step, "proto2_best_verification")
                    
                    result_csv = vexis_runner.run_analysis(
                        "proto2_best_verification",
                        log_path=result_dir / "logs" / "best_verification.log"
                    )
                else:
                    # ドライラン: ダミー結果（ターゲットに少しノイズ）
                    noise_level = best_value if best_value else 0.05
                    noise = np.random.normal(0, noise_level, len(target_curve))
                    dummy_result = target_curve.copy()
                    dummy_result["force"] = dummy_result["force"] + noise
                    
                    result_csv = project_root / "temp" / "proto2_best_verification_result.csv"
                    ensure_dir(result_csv.parent)
                    dummy_result.to_csv(result_csv, index=False)

                if result_csv:
                    best_curve = result_loader.load_curve(result_csv)
                    # プロット
                    visualizer.plot_best_result_comparison(
                        target_curve, best_curve
                    )
            except Exception as e:
                logger.error(f"確認解析エラー: {e}")
        
        # クリーンアップ
        logger.info("クリーンアップ中...")
        material_editor.cleanup_trial_materials()
        material_editor.restore_original_config()
        
        logger.info("=" * 60)
        logger.info("Proto2 最適化完了")
        logger.info(f"試行回数: {summary['total_trials']}")
        logger.info(f"最良RMSE: {summary['best_rmse']:.6f}")
        logger.info(f"収束達成: {summary['convergence_achieved']}")
        if best_params:
            logger.info("最良OGDEN係数:")
            logger.info(format_params_for_log(best_params))
        logger.info("=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"ファイルエラー: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("ユーザーによる中断 (Ctrl+C)")
        return 130
    except Exception as e:
        logger.exception(f"予期しないエラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
