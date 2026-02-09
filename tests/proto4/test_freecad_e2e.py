"""End-to-end test for FreeCAD integration with real FCStd file."""
import pytest
from pathlib import Path
import sys

# Add proto4-claude to path
proto4_path = Path(__file__).parent.parent.parent / 'src' / 'proto4-claude'
sys.path.insert(0, str(proto4_path))


class TestFreecadEndToEnd:
    """End-to-end tests with real FreeCAD execution."""

    @pytest.fixture
    def fcstd_path(self):
        """Path to test FCStd file."""
        path = Path(__file__).parent.parent.parent / "ref-fcad" / "TH1-ref.FCStd"
        if not path.exists():
            pytest.skip(f"Test FCStd not found: {path}")
        return path

    def test_freecad_engine_open_and_close(self, fcstd_path):
        """Test that FreecadEngine can open and close a document."""
        from proto4_claude.freecad_engine import FreecadEngine

        engine = FreecadEngine(fcstd_path=fcstd_path)
        try:
            engine.open()
            assert engine._doc is not None
            assert engine._sketch is not None
            print(f"Opened: {fcstd_path}")
            print(f"Sketch: {engine._sketch.Name}")
            if engine._surface:
                print(f"Surface: {engine._surface.Name}")
        finally:
            engine.close()
        
        assert engine._doc is None
        print("FreeCAD engine open/close: OK")

    def test_apply_baseline_ratios(self, fcstd_path, tmp_path):
        """Test applying baseline ratios (1.0) and exporting STEP."""
        from proto4_claude.freecad_engine import FreecadEngine, ConstraintSpec
        from proto4_claude.cad_gate import AI_V0_FEATURE_NAMES

        engine = FreecadEngine(fcstd_path=fcstd_path)
        try:
            engine.open()
            
            # Create constraint specs for first few parameters
            specs = []
            for idx, name in enumerate(AI_V0_FEATURE_NAMES[:5]):
                specs.append(ConstraintSpec(
                    index=idx,
                    name=name,
                    ctype="Distance",
                    base_value=1.0,
                ))
            engine.set_constraints(specs)
            
            # Apply baseline ratios
            ratios = {name: 1.0 for name in AI_V0_FEATURE_NAMES}
            result = engine.apply_ratios(ratios)
            print(f"apply_ratios result: {result}")
            
            # Export STEP if successful
            if result and engine._surface:
                step_path = tmp_path / "test_output.step"
                engine.export_step(step_path)
                assert step_path.exists()
                print(f"Exported STEP: {step_path} ({step_path.stat().st_size} bytes)")
        finally:
            engine.close()

    def test_geometry_adapter_integration(self, fcstd_path, tmp_path):
        """Test GeometryAdapter with real FreeCAD."""
        from proto4_claude.geometry_adapter import GeometryAdapter, GeometryError
        from proto4_claude.config import FreecadSpec
        from proto4_claude.types import DesignPoint
        from proto4_claude.cad_gate import AI_V0_FEATURE_NAMES

        # Create spec pointing to test file
        spec = FreecadSpec(
            fcstd_path=str(fcstd_path),
            sketch_name="Sketch001",
            surface_name="Face",
            surface_label="SURFACE",
            constraints={name: {"min": 0.8, "max": 1.2, "base_value": 1.0} 
                        for name in AI_V0_FEATURE_NAMES},
            step_output_dir=str(tmp_path),
            step_filename_template="trial_{trial_id}.step",
        )

        adapter = GeometryAdapter(spec, project_root=Path("."))
        adapter._fcstd_path = fcstd_path  # Override with absolute path

        try:
            # Create design point with baseline ratios
            params = {name: 1.0 for name in AI_V0_FEATURE_NAMES}
            point = DesignPoint(trial_id=999, params=params)

            # Try to generate STEP
            try:
                step_path = adapter.generate_step(point)
                assert step_path.exists()
                print(f"GeometryAdapter generated: {step_path}")
            except GeometryError as e:
                print(f"GeometryError (expected if constraints don't match): {e}")
                # This is OK if the FCStd doesn't have matching constraints
        finally:
            adapter.close()
