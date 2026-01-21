"""
Proto2 VEXIS Runner Module

VEXISサブモジュールのCLIをsubprocessで呼び出すラッパー
（Proto1から流用・修正）
- STEP編集なし（固定形状）
- 材料設定はMaterialEditorが担当
"""

import logging
import os
import subprocess
import signal
import threading
import time
from pathlib import Path
from typing import Optional

from .utils import ensure_dir, copy_file_safe, get_project_root


logger = logging.getLogger(__name__)


class VexisRunner:
    """
    VEXISサブモジュールのCLIラッパー
    
    Proto2ではSTEP編集なし、固定形状を使用
    材料設定はMaterialEditorが事前に行う
    """
    
    def __init__(self, vexis_path: Optional[str | Path] = None):
        """
        Args:
            vexis_path: VEXISサブモジュールのパス（省略時は自動検出）
        """
        if vexis_path:
            self.vexis_root = Path(vexis_path)
        else:
            self.vexis_root = get_project_root() / "vexis"
        
        if not self.vexis_root.exists():
            raise FileNotFoundError(f"VEXISサブモジュールが見つかりません: {self.vexis_root}")
        
        self.vexis_input = self.vexis_root / "input"
        self.vexis_results = self.vexis_root / "results"
        self.vexis_temp = self.vexis_root / "temp"
        self.vexis_main = self.vexis_root / "main.py"
        
        # 現在実行中のプロセス
        self._current_process: Optional[subprocess.Popen] = None
        self._stop_requested = False
        
        logger.info(f"VEXIS Runner初期化: {self.vexis_root}")
    
    def request_stop(self) -> None:
        """実行中の解析を停止要求"""
        self._stop_requested = True
        if self._current_process:
            logger.info("VEXIS停止要求を送信")
            self._terminate_process(self._current_process)
    
    def _terminate_process(self, proc: subprocess.Popen, timeout: int = 10) -> None:
        """プロセスを安全に終了"""
        if proc.poll() is not None:
            return  # 既に終了
        
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGTERM)
            
            try:
                proc.wait(timeout=timeout)
                logger.info("VEXISプロセスが正常終了")
            except subprocess.TimeoutExpired:
                logger.warning("VEXISプロセスを強制終了")
                proc.kill()
                proc.wait(timeout=5)
        except Exception as e:
            logger.error(f"プロセス終了エラー: {e}")
    
    def setup_input_step(self, step_path: str | Path, job_name: str) -> Path:
        """
        入力STEPファイルをVEXISのinputディレクトリにコピー
        
        Args:
            step_path: 入力STEPファイルのパス
            job_name: ジョブ名
        
        Returns:
            コピー先のパス
        """
        step_path = Path(step_path)
        
        if not step_path.exists():
            raise FileNotFoundError(f"STEPファイルが見つかりません: {step_path}")
        
        ensure_dir(self.vexis_input)
        target_step = self.vexis_input / f"{job_name}.step"
        copy_file_safe(step_path, target_step)
        
        logger.info(f"STEPファイルをコピー: {step_path.name} -> {target_step}")
        return target_step
    
    def run_analysis(
        self,
        job_name: str,
        log_path: Optional[str | Path] = None,
        timeout: int = 1800
    ) -> Optional[Path]:
        """
        CAE解析を実行
        
        Args:
            job_name: ジョブ名（結果ファイル名に使用）
            log_path: ログ出力先（省略時は標準出力）
            timeout: タイムアウト秒数（デフォルト30分）
        
        Returns:
            成功時: 結果CSVファイルのパス
            失敗時: None
        """
        self._stop_requested = False
        
        # プロジェクトルートの.venv環境のPythonを使用
        project_root = self.vexis_root.parent
        vexis_python = project_root / ".venv" / "Scripts" / "python.exe"
        if not vexis_python.exists():
            vexis_python = project_root / ".venv" / "bin" / "python"
        
        if vexis_python.exists():
            python_cmd = str(vexis_python)
        else:
            logger.warning(".venvが見つかりません。システムPythonを使用します。")
            python_cmd = "python"
        
        cmd = [python_cmd, str(self.vexis_main)]
        
        logger.info(f"VEXIS実行: {' '.join(cmd)}")
        
        # ログファイルの準備
        log_file = None
        if log_path:
            log_path = Path(log_path)
            ensure_dir(log_path.parent)
            log_file = open(log_path, "w", encoding="utf-8")
        
        error_detected = False
        
        try:
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            
            # Qt GUIを無効化
            env = os.environ.copy()
            env["QT_QPA_PLATFORM"] = "offscreen"
            env["DISPLAY"] = ""
            env["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkOSOpenGLRenderWindow"
            env["PYVISTA_OFF_SCREEN"] = "true"
            
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.vexis_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creationflags,
                env=env
            )
            self._current_process = proc
            
            # タイムアウト監視
            def timeout_watcher():
                start_time = time.time()
                while proc.poll() is None:
                    if self._stop_requested:
                        logger.info("停止要求により中断")
                        self._terminate_process(proc)
                        return
                    if time.time() - start_time > timeout:
                        logger.error(f"VEXIS実行タイムアウト: {timeout}秒")
                        self._terminate_process(proc)
                        return
                    time.sleep(1)
            
            watcher = threading.Thread(target=timeout_watcher, daemon=True)
            watcher.start()
            
            # リアルタイムログ出力
            try:
                for line in proc.stdout:
                    if self._stop_requested:
                        break
                    
                    if log_file:
                        log_file.write(line)
                        log_file.flush()
                    
                    line_lower = line.lower()
                    if "error termination" in line_lower or "fatal error" in line_lower:
                        error_detected = True
                        logger.warning(f"VEXISエラー検出: {line.rstrip()}")
                    
                    logger.debug(line.rstrip())
            except Exception as e:
                logger.warning(f"ログ読み取りエラー: {e}")
            
            proc.wait()
            
            if self._stop_requested:
                logger.info("VEXIS実行が中断されました")
                return None
            
            if proc.returncode != 0:
                logger.error(f"VEXIS実行エラー: return code {proc.returncode}")
                return None
            
            if error_detected:
                logger.warning("VEXISでエラーが検出されましたが、プロセスは完了しました")
            
        except Exception as e:
            logger.error(f"VEXIS実行例外: {e}")
            return None
        finally:
            self._current_process = None
            if log_file:
                log_file.close()
        
        # 結果ファイルを確認
        result_csv = self.vexis_results / f"{job_name}_result.csv"
        
        for _ in range(10):
            if result_csv.exists():
                break
            time.sleep(0.5)
        
        if result_csv.exists():
            logger.info(f"解析完了: {result_csv}")
            return result_csv
        else:
            logger.error(f"結果CSVが見つかりません: {result_csv}")
            return None
    
    def cleanup_input(self, job_name: str) -> None:
        """入力ファイルをクリーンアップ"""
        target_step = self.vexis_input / f"{job_name}.step"
        if target_step.exists():
            target_step.unlink()
            logger.debug(f"入力ファイル削除: {target_step}")
    
    def get_result_path(self, job_name: str) -> Path:
        """結果CSVファイルのパスを取得"""
        return self.vexis_results / f"{job_name}_result.csv"
