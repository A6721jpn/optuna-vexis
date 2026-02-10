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
    # doc.recompute() # Try recompute first?
    sketch = doc.getObject("Sketch")
    
    print(f"Sketch Constraint Count: {sketch.ConstraintCount}")

    # Test indices reported in failure log
    target_indices = [32, 35, 37, 46, 47] 
    
    print("\nTesting setDatum on problem indices...")
    for idx in target_indices:
        try:
            c = sketch.Constraints[idx]
            print(f"[{idx}] Name={c.Name} Type={c.Type} Val={c.Value}")
            
            # Try setting same value
            sketch.setDatum(idx, c.Value)
            print(f"  -> Success: setDatum({idx})")
        except Exception as e:
            print(f"  -> Failed: setDatum({idx}) - {e}")

except Exception as e:
    print(f"Global error: {e}")
finally:
    pass
