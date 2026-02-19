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
    
    target_name = "CROWN-D-L"
    target_c = None
    target_i = -1
    
    for i, c in enumerate(sketch.Constraints):
        if c.Name == target_name:
            target_c = c
            target_i = i
            break
            
    if target_c:
        print(f"Found {target_name} at index {target_i}")
        print(f"Current Value: {target_c.Value}")
        
        print("Attempting to set Value property directly...")
        try:
            # Note: In some FreeCAD versions, Constraints list returns copies or values, 
            # so modifying 'c' might not affect the sketch. 
            # We usually need to modify the list and assign it back, or use setDatum.
            
            # Test if 'c' is a reference or copy
            original_val = target_c.Value
            target_c.Value = 1.234
            print(f"Set Value to 1.234. New Value in object: {target_c.Value}")
            
            # Check if sketch actually updated
            print("Checking sketch.Constraints[i].Value...")
            if sketch.Constraints[target_i].Value == 1.234:
                print("Success! Direct property modification works.")
            else:
                print(f"Failed. Sketch value is still {sketch.Constraints[target_i].Value}")
                print("Direct property modification on Constraint object does NOT verify back to sketch.")
                
        except Exception as e:
            print(f"Exception setting Value: {e}")
            
except Exception as e:
    print(f"Global error: {e}")
finally:
    pass
