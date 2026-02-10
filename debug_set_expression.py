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
    
    target_idx = 32 # FOOT-MID
    new_val = 1.0832
    
    print(f"Target: Index {target_idx}, New Value {new_val}")
    
    # Method 1: setExpression with index
    # Note: Expression uses 0-based index? usually yes.
    # Syntax: setExpression(PropertyPath, Expression)
    # For constraints, path is usually 'Constraints[i]' 
    
    print("\nAttempting setExpression('Constraints[32]', '1.0832')...")
    try:
        sketch.setExpression(f'Constraints[{target_idx}]', str(new_val))
        print("Success: setExpression returned without error")
        
        # Verify
        doc.recompute()
        c = sketch.Constraints[target_idx]
        print(f"Verified Value: {c.Value}")
        
    except Exception as e:
        print(f"Failed setExpression: {e}")

    # Method 2: named constraints directly in expression?
    # Some versions support aliases. 'Constraints.FOOT_MID' ?
    # Let's check if aliases are defined.
    
    print("\nChecking aliases...")
    try:
        # Constraints usually don't have direct aliases unless mapped in ExpressionEngine
        pass
    except:
        pass

except Exception as e:
    print(f"Global error: {e}")
finally:
    pass
