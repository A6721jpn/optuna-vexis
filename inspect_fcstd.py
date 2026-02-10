import sys
import os

# Match freecad_engine.py's loading logic
conda_prefix = r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad"
freecad_bin = os.path.join(conda_prefix, "Library", "bin")
freecad_lib = os.path.join(conda_prefix, "Library", "lib")

os.environ["PATH"] = freecad_bin + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, freecad_bin)
sys.path.insert(0, freecad_lib)

import FreeCAD

doc = FreeCAD.openDocument(r"C:\github_repo\optuna-for-vexis\input\model.FCStd")

# Find sketch
sketch = None
for obj in doc.Objects:
    if obj.TypeId == "Sketcher::SketchObject":
        sketch = obj
        break

if sketch is None:
    print("No sketch found!")
else:
    print(f"Sketch: Name={sketch.Name}, Label={sketch.Label}")
    print(f"Constraint count: {sketch.ConstraintCount}")
    print()
    print("Constraints:")
    for i in range(sketch.ConstraintCount):
        c = sketch.Constraints[i]
        name = c.Name if c.Name else "(unnamed)"
        ctype = c.Type
        val = c.Value if hasattr(c, 'Value') else "N/A"
        print(f"  [{i:3d}] Name={name:30s} Type={ctype:15s} Value={val}")

FreeCAD.closeDocument(doc.Name)
