import sys
import os

# FreeCAD setup
conda_prefix = r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad"
freecad_bin = os.path.join(conda_prefix, "Library", "bin")
freecad_lib = os.path.join(conda_prefix, "Library", "lib")
os.environ["PATH"] = freecad_bin + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, freecad_bin)
sys.path.insert(0, freecad_lib)

import FreeCAD

try:
    doc = FreeCAD.openDocument(r"C:\github_repo\optuna-for-vexis\input\model.FCStd")
    sketch = doc.getObject("Sketch")
    
    print(f"Sketch Constraint Count: {sketch.ConstraintCount}")
    
    # Check driving status for all named constraints
    print("\nNamed Constraints Status:")
    for i, c in enumerate(sketch.Constraints):
        if c.Name:
            driving = "Yes" if c.Driving else "No"
            active = "Yes" if c.IsActive else "No"
            print(f"[{i:3d}] {c.Name:30s} Driving={driving} Active={active} Type={c.Type}")

except Exception as e:
    print(f"Global error: {e}")
finally:
    pass
