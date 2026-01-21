"""
Proto2 ユーティリティモジュール

ロギング、ファイル操作、共通関数を提供
（Proto1から流用・拡張）
"""

import logging
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def setup_logger(
    name: str,
    log_dir: str,
    level: str = "INFO",
    console: bool = True
) -> logging.Logger:
    """
    ロガーをセットアップ
    
    Args:
        name: ロガー名
        log_dir: ログ出力ディレクトリ
        level: ログレベル (DEBUG, INFO, WARNING, ERROR)
        console: コンソール出力の有無
    
    Returns:
        設定済みLogger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    
    # ログディレクトリ作成
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # ファイルハンドラ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # コンソールハンドラ
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        console_formatter = logging.Formatter(
            "[%(levelname)s] %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_project_root() -> Path:
    """プロジェクトルートディレクトリを取得"""
    # src/proto2/utils.py -> プロジェクトルート
    return Path(__file__).parent.parent.parent


def ensure_dir(path: str | Path) -> Path:
    """ディレクトリが存在しなければ作成"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file_safe(src: str | Path, dst: str | Path) -> Path:
    """
    ファイルを安全にコピー
    
    既存ファイルがあれば上書き
    """
    src, dst = Path(src), Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst


def save_json(data: dict, path: str | Path, indent: int = 2) -> None:
    """辞書をJSONファイルに保存"""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> dict:
    """JSONファイルを読込"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str | Path) -> dict:
    """YAMLファイルを読込"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str | Path) -> None:
    """辞書をYAMLファイルに保存"""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def generate_trial_id() -> str:
    """一意のトライアルIDを生成"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]


class TrialLogger:
    """
    各試行のログを管理するクラス
    """
    
    def __init__(self, trial_id: str, base_log_dir: str):
        self.trial_id = trial_id
        self.trial_dir = Path(base_log_dir) / f"trial_{trial_id}"
        ensure_dir(self.trial_dir)
        self._data: dict[str, Any] = {
            "trial_id": trial_id,
            "start_time": datetime.now().isoformat(),
            "parameters": {},
            "objectives": {},
            "status": "running"
        }
    
    def log_parameters(self, params: dict, precision: int = 6) -> None:
        """パラメータを記録（浮動小数点は丸める）"""
        # 浮動小数点の表示精度問題を回避するため丸める
        rounded_params = {}
        for key, values in params.items():
            if isinstance(values, list):
                rounded_params[key] = [round(v, precision) for v in values]
            else:
                rounded_params[key] = round(values, precision) if isinstance(values, float) else values
        self._data["parameters"] = rounded_params
        self._save()
    
    def log_objectives(self, objectives: dict) -> None:
        """目的関数値を記録"""
        self._data["objectives"] = objectives
        self._data["end_time"] = datetime.now().isoformat()
        self._data["status"] = "completed"
        self._save()
    
    def log_error(self, error: str) -> None:
        """エラーを記録"""
        self._data["error"] = error
        self._data["end_time"] = datetime.now().isoformat()
        self._data["status"] = "failed"
        self._save()
    
    def get_log_path(self, name: str) -> Path:
        """サブログファイルのパスを取得"""
        return self.trial_dir / f"{name}.log"
    
    def _save(self) -> None:
        """試行データをJSONに保存"""
        save_json(self._data, self.trial_dir / "trial_info.json")


def format_params_for_log(params: dict) -> str:
    """
    OGDEN係数を読みやすい形式でフォーマット
    
    Args:
        params: {"c": [...], "m": [...], "t": [...], "g": [...]}
    
    Returns:
        フォーマット済み文字列
    """
    lines = []
    for key in ["c", "m", "t", "g"]:
        if key in params:
            vals = params[key]
            vals_str = ", ".join(f"{v:.4f}" for v in vals)
            lines.append(f"  {key}: [{vals_str}]")
    return "\n".join(lines)
