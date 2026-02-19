import sys
import os
from pathlib import Path
import logging

# Setup FreeCAD path
freecad_path = Path(r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad\Library\bin")
sys.path.append(str(freecad_path))
os.environ["PATH"] += f";{freecad_path}"

import FreeCAD
import Part
import Sketcher

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_expression")

def main():
    fcstd_path = Path("input/model.FCStd").resolve()
    logger.info(f"Opening {fcstd_path}")
    
    doc = FreeCAD.open(str(fcstd_path))
    sketch = doc.getObject("Sketch")
    
    # Disable auto-remove (match current engine)
    p = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/Sketcher")
    p.SetBool("AutoRemoveRedundants", True)
    p.SetBool("AutoRecompute", True)

    logger.info(f"Constraint count: {len(sketch.Constraints)}")

    # Target constraints (from config)
    # We don't have the mapping here easily, but we can iterate all named constraints
    # or just try to set ALL constraints to their current value
    
    failures = 0
    
    for i, c in enumerate(sketch.Constraints):
        if not c.Name:
            continue
            
        original_val = c.Value
        
        # Determine unit
        # Guess based on value size or name? 
        # Actually we can check c.Type? No, Type is enum.
        # Let's assume Distance unless name contains Angle?
        # Or just use the logic from engine:
        # if "Angle" in c.Name or value is around 90/180/45...
        
        c_type = getattr(c, "Type", "Unknown")
        logger.info(f"[{i}] {c.Name} (Type: {c_type}, {type(c_type)}): {original_val}")

        # Engine logic:
        # angle_constraints = ["SHOULDER-ANGLE-OUT", "SHOULDER-ANGLE-IN"]
        is_angle = "ANGLE" in c.Name
        
        unit = "deg" if is_angle else "mm"
        expression = f"{original_val} {unit}"
        
        logger.info(f"[{i}] {c.Name}: {original_val} -> '{expression}'")
        
        try:
            sketch.setExpression(f"Constraints[{i}]", expression)
            doc.recompute()
            
            # Check recompute
            if "Invalid" in getattr(sketch, "State", []) or "RecomputeError" in getattr(sketch, "State", []):
                logger.error(f"  -> FAILED Recompute! State: {sketch.State}")
                failures += 1
            else:
                logger.info(f"  -> OK. New val: {sketch.Constraints[i].Value}")
                
        except Exception as e:
            logger.error(f"  -> EXCEPTION: {e}")
            failures += 1
            
    if failures == 0:
        logger.info("ALL GREEN: setExpression with current values is safe.")
    else:
        logger.error(f"FAILED: {failures} constraints broke the model.")

if __name__ == "__main__":
    main()
