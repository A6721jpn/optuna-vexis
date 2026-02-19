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
    
    # Inspect one constraint to see available attributes
    if sketch.ConstraintCount > 0:
        c = sketch.Constraints[32] # FOOT-MID
        print(f"Constraint 32 (FOOT-MID) attributes:")
        print(dir(c))
        
        # Check specific attributes usually related to driving status
        # Note: In Python API, it might not be directly on the Constraint object 
        # but handled via sketch methods like toggleDriving(index) or similar? 
        # Actually in 0.19+, constraint object has no 'Driving' attribute usually.
        # But let's check.
        
except Exception as e:
    print(f"Global error: {e}")
finally:
    pass
