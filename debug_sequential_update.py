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

def set_datum_by_name(sketch, name, value):
    # Find index
    idx = -1
    for i, c in enumerate(sketch.Constraints):
        if c.Name == name:
            idx = i
            break
    
    if idx == -1:
        print(f"FAILED: Constraint '{name}' not found")
        return False
    
    try:
        # Check type
        ctype = sketch.Constraints[idx].Type
        print(f"Setting '{name}' (idx={idx}, type={ctype}) to {value}...")
        
        sketch.setDatum(idx, float(value))
        print(f"  -> Success")
        return True
    except Exception as e:
        print(f"  -> FAILED: {e}")
        # Inspect the constraint state
        try:
            c = sketch.Constraints[idx]
            print(f"     Debug: Driving={c.Driving}, Active={c.Active}, Value={c.Value}")
        except:
            pass
        return False

try:
    # Open document
    doc = FreeCAD.openDocument(r"C:\github_repo\optuna-for-vexis\input\model.FCStd")
    # doc.recompute()
    sketch = doc.getObject("Sketch")
    
    # Simulation of apply_ratios
    # We apply the same updates as in the pipeline (roughly)
    # The pipeline applies update in the order of constraints definition in yaml
    
    constraints_order = [
        "CROWN-D-L",
        "CROWN-D-H",
        "CROWN-W",
        "PUSHER-D-H",
        "PUSHER-D-L",
        "TIP-D",
        "STROKE-OUT",
        "STROKE-CENTER",
        "FOOT-W",
        "FOOT-MID",  # <-- Fails here often
        "SHOULDER-ANGLE-OUT",
        "SHOULDER-ANGLE-IN",
        "TOP-T",
        "TOP-DROP",
        "FOOT-IN",
        "DIAMETER",
        "HEIGHT",
        "TIP-DROP",
        "SHOUDER-T",
        "FOOT-OUT"
    ]
    
    # Just try setting them all to their current value (ratio 1.0)
    # Or slight modification to trigger solver
    
    for name in constraints_order:
        # Get current value first? Or just set dummy?
        # Let's try to set them to base_value * 1.0 (from yaml, but we don't have yaml here)
        # We'll read current value and set it back, essentially a no-op but calls setDatum
        
        # Find constraint to get value
        val = 0.0
        found = False
        for c in sketch.Constraints:
            if c.Name == name:
                val = c.Value
                found = True
                break
        
        if not found:
            print(f"Skipping {name} (not found)")
            continue
            
        # Call setDatum
        if not set_datum_by_name(sketch, name, val):
            print("Stopping due to failure")
            break

except Exception as e:
    print(f"Global error: {e}")
finally:
    pass
