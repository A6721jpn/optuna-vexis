"""Full STEP export test with constraint modification."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# FreeCAD import fallback
try:
    import FreeCAD
    import Part
except ImportError:
    for name in ("fcad", "fcad-codex", "b123d"):
        env_path = Path(r"C:\Users\aokuni\AppData\Local\miniforge3\envs") / name
        freecad_bin = env_path / "Library" / "bin"
        freecad_lib = env_path / "Library" / "lib"
        if not freecad_bin.exists():
            continue
        os.environ["PATH"] = str(freecad_bin) + os.pathsep + os.environ.get("PATH", "")
        sys.path.insert(0, str(freecad_bin))
        if freecad_lib.exists():
            sys.path.insert(0, str(freecad_lib))
        try:
            import FreeCAD
            import Part
            print(f"FreeCAD loaded from {env_path}")
            break
        except ImportError:
            continue
    else:
        print("ERROR: FreeCAD not found")
        sys.exit(1)


# ai-v0 feature names (20-dim)
FEATURE_NAMES = [
    "CROWN-D-L", "CROWN-D-H", "CROWN-W", "PUSHER-D-H", "PUSHER-D-L",
    "TIP-D", "STROKE-OUT", "STROKE-CENTER", "FOOT-W", "FOOT-MID",
    "SHOULDER-ANGLE-OUT", "SHOULDER-ANGLE-IN", "TOP-T", "TOP-DROP",
    "FOOT-IN", "DIAMETER", "HEIGHT", "TIP-DROP", "SHOUDER-T", "FOOT-OUT"
]


def main():
    fcstd_path = Path(__file__).parent / "ref-fcad" / "TH1-ref.FCStd"
    output_dir = Path(__file__).parent / "input" / "step"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Opening: {fcstd_path}")
    doc = FreeCAD.openDocument(str(fcstd_path))
    
    # Find sketch and build constraint map
    sketch = None
    for obj in doc.Objects:
        if obj.TypeId == "Sketcher::SketchObject":
            sketch = obj
            break
    
    # Map constraint names to indices
    constraint_map = {}
    for i in range(sketch.ConstraintCount):
        c = sketch.Constraints[i]
        if c.Name:
            constraint_map[c.Name] = {
                "index": i,
                "type": c.Type,
                "base_value": c.Value,
            }
    
    print(f"\nConstraint map ({len(constraint_map)} named):")
    for name in FEATURE_NAMES:
        if name in constraint_map:
            c = constraint_map[name]
            print(f"  {name:20s}: idx={c['index']:2d}, base={c['base_value']:.4f}")
        else:
            print(f"  {name:20s}: NOT FOUND")
    
    # Find Part object for export
    part_obj = doc.getObject("Part__Feature")
    if not part_obj:
        # Try to find any valid Part
        for obj in doc.Objects:
            if hasattr(obj, "Shape") and not obj.Shape.isNull():
                if obj.Shape.Area > 0 and obj.Shape.Area < 1e10:
                    part_obj = obj
                    break
    
    print(f"\nPart object: {part_obj.Name if part_obj else 'NOT FOUND'}")
    
    # Test 1: Baseline export
    print("\n=== Test 1: Baseline STEP export ===")
    step_path = output_dir / "test_baseline.step"
    Part.export([part_obj], str(step_path))
    print(f"Exported: {step_path} ({step_path.stat().st_size} bytes)")
    
    # Test 2: Modified geometry (all ratios = 1.05)
    print("\n=== Test 2: Modified geometry (ratio=1.05) ===")
    original_values = {}
    for name in FEATURE_NAMES:
        if name in constraint_map:
            c = constraint_map[name]
            original_values[name] = c["base_value"]
            new_value = c["base_value"] * 1.05
            if c["type"] == "Angle":
                quantity = FreeCAD.Units.Quantity(f"{new_value} rad")
                sketch.setDatum(c["index"], quantity)
            else:
                sketch.setDatum(c["index"], float(new_value))
    
    doc.recompute()
    
    # Check recompute status
    has_error = False
    for obj in doc.Objects:
        state = getattr(obj, "State", None)
        if state and any(flag in state for flag in ("Invalid", "RecomputeError")):
            print(f"  ERROR: {obj.Name} state: {state}")
            has_error = True
    
    if not has_error:
        print("  Recompute: SUCCESS")
        step_path = output_dir / "test_modified.step"
        Part.export([part_obj], str(step_path))
        print(f"  Exported: {step_path} ({step_path.stat().st_size} bytes)")
    
    # Restore original values
    print("\n=== Restoring original values ===")
    for name, val in original_values.items():
        c = constraint_map[name]
        if c["type"] == "Angle":
            quantity = FreeCAD.Units.Quantity(f"{val} rad")
            sketch.setDatum(c["index"], quantity)
        else:
            sketch.setDatum(c["index"], float(val))
    doc.recompute()
    print("  Restored")
    
    FreeCAD.closeDocument(doc.Name)
    print("\n=== STEP export test: SUCCESS ===")


if __name__ == "__main__":
    main()
