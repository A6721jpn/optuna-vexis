"""
Proto2 Visualizer Module

最適化結果の可視化
- 最適化履歴（目的関数値の推移）
- パレートフロント（多目的最適化時）
- カーブ比較（最良試行 vs ターゲット）
"""

import logging
import os
from pathlib import Path
from typing import Optional, Any

# Qtプラットフォームエラーを回避
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import matplotlib
matplotlib.use('Agg')  # 非GUIバックエンド（Qt不要）
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from optuna.study import Study
from optuna.trial import FrozenTrial

logger = logging.getLogger(__name__)


class Visualizer:
    """
    最適化結果をグラフ化するクラス
    """
    
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 日本語フォント対応は環境依存が強いため、デフォルトフォントで英語ラベル推奨
        # plt.rcParams['font.family'] = 'sans-serif'
    
    def plot_optimization_history(self, study: Study, filename: str = "optimization_history.png") -> None:
        """
        最適化履歴（Best Valueの推移）をプロット
        """
        if len(study.trials) == 0:
            return
            
        trials = [t for t in study.trials if t.state.name == "COMPLETE"]
        if not trials:
            return
            
        trial_numbers = [t.number for t in trials]
        # 多目的最適化対応: 最初の目的関数(RMSE)を使用
        try:
            values = [t.value for t in trials]
        except RuntimeError:
            values = [t.values[0] for t in trials if t.values]

        if not values:
            return
        
        # Best Valueの推移を計算
        best_values = []
        current_best = float("inf")
        for v in values:
            if v < current_best:
                current_best = v
            best_values.append(current_best)
        
        plt.figure(figsize=(10, 6))
        plt.plot(trial_numbers, values, 'o', color='lightblue', alpha=0.5, label='Trial Value')
        plt.plot(trial_numbers, best_values, '-', color='red', linewidth=2, label='Best Value')
        
        plt.title('Optimization History')
        plt.xlabel('Number of Trials')
        plt.ylabel('Objective Value (RMSE)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"最適化履歴グラフを保存: {output_path}")

    def plot_pareto_front(
        self, 
        study: Study, 
        target_names: tuple[str, str] = ("Objective 1", "Objective 2"),
        filename: str = "pareto_front.png"
    ) -> None:
        """
        パレートフロント（多目的最適化）をプロット
        """
        # 単一目的の場合はスキップ
        if len(study.directions) < 2:
            return

        trials = [t for t in study.best_trials]
        if not trials:
            return

        values = [t.values for t in trials]
        if not values:
            return
            
        x_vals = [v[0] for v in values]
        y_vals = [v[1] for v in values]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x_vals, y_vals, c='red', label='Pareto Front')
        
        # 全試行も薄くプロット
        all_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
        if all_trials:
            all_values = [t.values for t in all_trials if t.values]
            if all_values:
                all_x = [v[0] for v in all_values]
                all_y = [v[1] for v in all_values]
                plt.scatter(all_x, all_y, c='blue', alpha=0.1, label='All Trials')
        
        plt.title('Pareto Front')
        plt.xlabel(target_names[0])
        plt.ylabel(target_names[1])
        plt.grid(True)
        plt.legend()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"パレートフロントグラフを保存: {output_path}")

    def plot_best_result_comparison(
        self,
        target_curve: pd.DataFrame,
        best_result_curve: pd.DataFrame,
        filename: str = "best_result_comparison.png"
    ) -> None:
        """
        ターゲットカーブと最良結果カーブの比較プロット
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(
            target_curve["displacement"], 
            target_curve["force"], 
            '--', color='black', linewidth=2, label='Target'
        )
        
        plt.plot(
            best_result_curve["displacement"], 
            best_result_curve["force"], 
            '-', color='red', linewidth=2, label='Optimized'
        )
        
        plt.title('Result Comparison')
        plt.xlabel('Displacement [mm]')
        plt.ylabel('Force [N]')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"結果比較グラフを保存: {output_path}")
