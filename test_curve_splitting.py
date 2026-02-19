
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(".").resolve()))

from src.proto2.curve_processor import CurveProcessor
from src.proto2.objective import ObjectiveCalculator, FatalOptimizationError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_curve_splitting")

def create_cycle_data(max_disp=10.0, points=20, has_unload=True):
    # Loading
    x_load = np.linspace(0, max_disp, points)
    y_load = x_load * 2.0  # Linear stiffness 2.0
    
    if not has_unload:
        return pd.DataFrame({"displacement": x_load, "force": y_load})
    
    # Unloading
    x_unload = np.linspace(max_disp, 0, points)
    y_unload = x_unload * 1.5 # Hysteresis, lower force
    
    # Concatenate
    x = np.concatenate([x_load, x_unload])
    y = np.concatenate([y_load, y_unload])
    
    return pd.DataFrame({"displacement": x, "force": y})

def test_splitting():
    logger.info("--- Testing Split Logic ---")
    processor = CurveProcessor()
    
    # Data with Loading and Unloading
    df_cycle = create_cycle_data(has_unload=True)
    load, unload = processor.split_cycle(df_cycle)
    
    assert load is not None
    assert unload is not None
    logger.info(f"Full Cycle Split: Load={len(load)}, Unload={len(unload)} - OK")
    
    # Data with Loading only
    df_load = create_cycle_data(has_unload=False)
    load, unload = processor.split_cycle(df_load)
    
    assert load is not None
    assert unload is None
    logger.info(f"Load Only Split: Load={len(load)}, Unload={unload} - OK")

def test_cases():
    logger.info("\n--- Testing 4 Cases ---")
    
    target_feats = {}
    config = {"weights": {"rmse": 1.0}}
    
    # Prepare Data
    df_LU = create_cycle_data(has_unload=True) # Loading + Unloading
    df_L  = create_cycle_data(has_unload=False) # Loading only
    
    # --- Case 1: Target(L+U) vs Result(L+U) ---
    logger.info("Case 1: Target(L+U) vs Result(L+U)")
    calc = ObjectiveCalculator(df_LU, target_feats, config)
    res = calc.evaluate(df_LU, {}) # Same data, RMSE should be 0
    logger.info(f"RMSE: {res['rmse']} (Expected 0.0)")
    assert res['rmse'] < 1e-6
    if "rmse_unloading" in res:
        logger.info("Unloading evaluated - OK")
    else:
        logger.error("Unloading NOT evaluated!")

    # --- Case 2: Target(L) vs Result(L+U) ---
    logger.info("Case 2: Target(L) vs Result(L+U)")
    calc = ObjectiveCalculator(df_L, target_feats, config)
    # Result has U, but Target is L only. Should ignore U.
    # Since Result L matches Target L (same gen function), RMSE should be 0.
    res = calc.evaluate(df_LU, {}) 
    logger.info(f"RMSE: {res['rmse']} (Expected 0.0)")
    assert res['rmse'] < 1e-6
    
    # --- Case 3: Target(L+U) vs Result(L) ---
    logger.info("Case 3: Target(L+U) vs Result(L)")
    calc = ObjectiveCalculator(df_LU, target_feats, config)
    # Result is L only. Target has U. Should ignore Target U.
    res = calc.evaluate(df_L, {})
    logger.info(f"RMSE: {res['rmse']} (Expected 0.0)")
    assert res['rmse'] < 1e-6

    # --- Case 4: Result No Load ---
    logger.info("Case 4: Result No Loading (Fatal Error)")
    # Create invalid result (e.g. only 1 point)
    df_invalid = pd.DataFrame({"displacement": [0.0], "force": [0.0]})
    try:
        calc.evaluate(df_invalid, {})
        logger.error("Failed to raise FatalOptimizationError")
    except FatalOptimizationError:
        logger.info("Raised FatalOptimizationError as expected - OK")
    except Exception as e:
        logger.error(f"Raised wrong exception: {type(e)}")

if __name__ == "__main__":
    test_splitting()
    test_cases()
