"""VEXIS Integration Tests.

Tests the full VEXIS pipeline (FEBio-based CAE) with real execution.

These tests require the local VEXIS/FEBio environment and are skipped in CI.
Run manually with: pytest tests/proto4/test_vexis_integration.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Import from proto4_claude via conftest aliasing
from proto4_claude.cae_evaluator import load_curve, extract_range, calculate_rmse


# Skip all tests in this module if VEXIS is not available
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("SKIP_VEXIS") == "true",
    reason="VEXIS tests require local FEBio environment",
)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).parent.parent.parent


@pytest.fixture
def vexis_root(project_root: Path) -> Path:
    return project_root / "vexis"


@pytest.fixture
def test_step_file(project_root: Path) -> Path:
    """Return existing STEP file for testing."""
    step_file = project_root / "vexis" / "input" / "proto2_best_verification.step"
    if not step_file.exists():
        pytest.skip(f"Test STEP file not found: {step_file}")
    return step_file


@pytest.fixture
def febio_path() -> Path:
    """Check FEBio is installed and return path."""
    febio = Path("C:/Program Files/FEBioStudio/bin/febio4.exe")
    if not febio.exists():
        pytest.skip(f"FEBio not found: {febio}")
    return febio


def _check_vexis_can_run(vexis_root: Path) -> bool:
    """Check if VEXIS dependencies are available."""
    # Check FEBio
    febio = Path("C:/Program Files/FEBioStudio/bin/febio4.exe")
    if not febio.exists():
        return False
    # Check template
    template = vexis_root / "template2.feb"
    if not template.exists():
        return False
    return True



class TestVexisEnvironment:
    """Verify VEXIS environment is correctly set up."""

    def test_vexis_directory_exists(self, vexis_root: Path):
        assert vexis_root.exists(), f"VEXIS root not found: {vexis_root}"

    def test_vexis_main_exists(self, vexis_root: Path):
        main_py = vexis_root / "main.py"
        assert main_py.exists(), f"VEXIS main.py not found: {main_py}"

    def test_vexis_input_dir_exists(self, vexis_root: Path):
        input_dir = vexis_root / "input"
        assert input_dir.exists(), f"VEXIS input dir not found: {input_dir}"

    def test_vexis_results_dir_exists(self, vexis_root: Path):
        results_dir = vexis_root / "results"
        assert results_dir.exists(), f"VEXIS results dir not found: {results_dir}"


class TestVexisExecution:
    """Test actual VEXIS execution (requires FEBio)."""

    @pytest.mark.slow
    def test_vexis_single_execution(self, vexis_root: Path, test_step_file: Path, febio_path: Path, tmp_path: Path):
        """Run VEXIS on a test STEP file and verify output.
        
        Note: This test processes ALL STEP files in vexis/input/.
        It may take several minutes per file.
        """
        # Setup - copy STEP to input with unique name
        job_name = "test_vexis_integration"
        input_step = vexis_root / "input" / f"{job_name}.step"
        shutil.copy2(test_step_file, input_step)

        try:
            # Find Python interpreter
            venv_py = vexis_root.parent / ".venv" / "Scripts" / "python.exe"
            if not venv_py.exists():
                venv_py = vexis_root.parent / ".venv" / "bin" / "python"
            python_cmd = str(venv_py) if venv_py.exists() else sys.executable

            # Set environment for headless execution
            env = os.environ.copy()
            env.update({
                "QT_QPA_PLATFORM": "offscreen",
                "PYVISTA_OFF_SCREEN": "true",
                "VTK_DEFAULT_OPENGL_WINDOW": "vtkOSOpenGLRenderWindow",
            })

            # Run VEXIS
            result = subprocess.run(
                [python_cmd, str(vexis_root / "main.py")],
                cwd=str(vexis_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout
            )

            # Check for errors
            if result.returncode != 0:
                pytest.fail(f"VEXIS execution failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

            # Verify result CSV exists
            result_csv = vexis_root / "results" / f"{job_name}_result.csv"
            assert result_csv.exists(), f"Result CSV not found: {result_csv}"

            # Load and validate result
            df = load_curve(result_csv)
            assert len(df) > 0, "Result curve is empty"
            assert "displacement" in df.columns
            assert "force" in df.columns
            assert df["displacement"].min() >= 0, "Displacement should be non-negative"

        finally:
            # Cleanup
            if input_step.exists():
                input_step.unlink()


class TestCaeEvaluatorIntegration:
    """Test CaeEvaluator with real VEXIS execution."""

    @pytest.mark.slow
    def test_cae_evaluator_with_real_vexis(
        self, project_root: Path, vexis_root: Path, test_step_file: Path, febio_path: Path
    ):
        """Full CaeEvaluator test with real VEXIS."""
        from proto4_claude.cae_evaluator import CaeEvaluator
        from proto4_claude.config import CaeSpec, ObjectiveSpec
        from proto4_claude.types import CaeStatus, DesignPoint

        # Create target curve (synthetic for test)
        import numpy as np
        d = np.linspace(0.0, 0.5, 50)
        target_curve = pd.DataFrame({
            "displacement": d,
            "force": 100.0 * (d / 0.5) ** 1.5,
        })

        # Create specs
        cae_spec = CaeSpec(
            stroke_range_min=0.0,
            stroke_range_max=0.5,
            timeout_sec=600,
            max_retries=1,
        )
        obj_spec = ObjectiveSpec(
            type="rmse",
            weights={"rmse": 1.0},
            features={},
        )

        # Create evaluator
        evaluator = CaeEvaluator(
            vexis_path=vexis_root,
            cae_spec=cae_spec,
            obj_spec=obj_spec,
            target_curve=target_curve,
            target_features={},
        )

        # Create design point
        point = DesignPoint(
            trial_id=999,
            params={"test_param": 1.0},
        )

        # Run evaluation
        result = evaluator.evaluate(test_step_file, point)

        # Verify result
        assert result.status == CaeStatus.SUCCESS, f"CAE failed: {result}"
        assert result.metrics is not None
        assert "rmse" in result.metrics
        assert result.metrics["rmse"] >= 0


class TestResultCsvLoading:
    """Test loading existing VEXIS result CSVs."""

    def test_load_existing_result_csv(self, vexis_root: Path):
        """Load any existing result CSV in vexis/results/."""
        results_dir = vexis_root / "results"
        csv_files = list(results_dir.glob("*_result.csv"))

        if not csv_files:
            pytest.skip("No result CSVs found in vexis/results/")

        for csv_path in csv_files[:3]:  # Test up to 3 files
            df = load_curve(csv_path)
            assert len(df) > 0, f"Empty curve in {csv_path}"
            assert "displacement" in df.columns
            assert "force" in df.columns
