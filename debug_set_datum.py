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
    print("Opened document")

    # Find sketch
    sketch = None
    for obj in doc.Objects:
        if obj.TypeId == "Sketcher::SketchObject":
            sketch = obj
            break
            
    if not sketch:
        print("Sketch not found")
        sys.exit(1)

    print(f"Sketch: {sketch.Name}")
    
    # Try to set datum for CROWN-D-L
    # From previous inspection, CROWN-D-L is at index 23
    target_name = "CROWN-D-L"
    target_idx = -1
    for i, c in enumerate(sketch.Constraints):
        if c.Name == target_name:
            target_idx = i
            print(f"Found {target_name} at index {i}, Type={c.Type}, Value={c.Value}")
            break
            
    if target_idx == -1:
        print(f"{target_name} not found")
        sys.exit(1)
        
    # Experiment 1: setDatum using list index
    print(f"Attempting setDatum({target_idx}, 1.0)...")
    try:
        sketch.setDatum(target_idx, 1.0)
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")

    # Experiment 2: Check getDatum
    # Python API might not have getDatum exposed directly or it works differently
    
    # Analyze Constraints to count how many have Datums
    print("\nCounting constraints with datums...")
    datum_count = 0
    crown_datum_idx = -1
    
    # List of types that typically have datums
    # Distance, DistanceX, DistanceY, Radius, Diameter, Angle, etc.
    # Coincident, Horizontal, Vertical usually don't (unless they have value?)
    
    # But inspecting values in previous step:
    # [  1] Name=(unnamed) Type=Horizontal Value=0.0
    # Value is 0.0, implies it might have a datum? 
    # Usually Horizontal/Vertical are just constraints without value unless it's "Horiz Distance" which is DistanceX.
    
    # Let's try to setDatum on index 0..ConstraintCount to see which ones accept it
    print("\nScanning setDatum on first 30 indices...")
    for i in range(30):
        try:
            # retrieve current value first to avoid changing geometry too much
            # val = sketch.getDatum(i) # if this exists
            # Just try setting to current value if possible, or dummy
            
            # We don't know current value easily without accessing Constraints[i].Value
            # But we don't know if 'i' matches Constraints[i]
            
            sketch.setDatum(i, 1.0) # Just try setting 1.0
            print(f"Index {i}: Success (ConstraintType=?)")
            # Revert? No need, we close without saving
        except Exception as e:
            print(f"Index {i}: Failed ({e})")

except Exception as e:
    print(f"Global error: {e}")
finally:
    # Don't save
    pass
