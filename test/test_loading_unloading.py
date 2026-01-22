
import sys
import os
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.proto2.curve_processor import CurveProcessor
from src.proto2.objective import ObjectiveCalculator, FatalOptimizationError

class TestLoadingUnloading(unittest.TestCase):
    def setUp(self):
        self.processor = CurveProcessor()
        
        # Create a hysteresis loop data: 0 -> 10 -> 0 (x), Force follows
        x_load = np.linspace(0, 1.0, 11)
        y_load = x_load * 10 # Linear loading
        
        x_unload = np.linspace(1.0, 0, 11)
        y_unload = x_unload * 5 + 5 # Hysteresis
        
        # Combined (join smoothly at 1.0)
        # x_load ends at 1.0, x_unload starts at 1.0
        # effectively using x_load + x_unload[1:] to share the peak point
        self.df_loop = pd.DataFrame({
            "displacement": np.concatenate([x_load, x_unload[1:]]),
            "force": np.concatenate([y_load, y_unload[1:]])
        })
        
        # Loading only
        self.df_load = pd.DataFrame({
            "displacement": x_load,
            "force": y_load
        })
        
        # Unloading only data (invalid as full curve but for testing)
        # Note: CurveProcessor expects start from 0 usually, but split_cycle checks idxmax
        
    def test_split_cycle(self):
        load, unload = self.processor.split_cycle(self.df_loop)
        
        self.assertIsNotNone(load)
        self.assertIsNotNone(unload)
        self.assertEqual(len(load), 11) # 0 to 1.0 inclusive
        self.assertEqual(len(unload), 11) # 1.0 to 0 inclusive
        
        # Verify overlap point
        self.assertEqual(load.iloc[-1]["displacement"], 1.0)
        self.assertEqual(unload.iloc[0]["displacement"], 1.0)
        
    def test_split_cycle_load_only(self):
        load, unload = self.processor.split_cycle(self.df_load)
        self.assertIsNotNone(load)
        self.assertIsNone(unload)
        self.assertEqual(len(load), 11)

    def test_case1_both_exist(self):
        """Case 1: Target(L+U), Result(L+U)"""
        calc = ObjectiveCalculator(self.df_loop, {}, {})
        
        # Result is identical to target -> RMSE should be 0
        objectives = calc.evaluate(self.df_loop, {})
        self.assertAlmostEqual(objectives["rmse"], 0.0)
        self.assertAlmostEqual(objectives["rmse_loading"], 0.0)
        self.assertAlmostEqual(objectives["rmse_unloading"], 0.0)
        
        # Result slightly different
        df_res = self.df_loop.copy()
        df_res["force"] += 1.0
        
        objectives = calc.evaluate(df_res, {})
        self.assertAlmostEqual(objectives["rmse"], 1.0) # Avg of 1.0 and 1.0 is 1.0

    def test_case2_target_load_only(self):
        """Case 2: Target(L), Result(L+U) -> Evaluate Loading only"""
        calc = ObjectiveCalculator(self.df_load, {}, {})
        
        # Result has loop, Target has load only
        objectives = calc.evaluate(self.df_loop, {})
        
        # Should match loading part (perfect match)
        self.assertAlmostEqual(objectives["rmse"], 0.0)
        self.assertIn("rmse_loading", objectives)
        self.assertNotIn("rmse_unloading", objectives) # Unloading ignored

    def test_case3_result_load_only(self):
        """Case 3: Target(L+U), Result(L) -> Evaluate Loading only"""
        calc = ObjectiveCalculator(self.df_loop, {}, {})
        
        # Result has load only
        objectives = calc.evaluate(self.df_load, {})
        
        # Should match loading part
        self.assertAlmostEqual(objectives["rmse"], 0.0)
        self.assertIn("rmse_loading", objectives)
        self.assertNotIn("rmse_unloading", objectives)

    def test_case4_fatal_error(self):
        """Case 4: Result has no loading (or invalid)"""
        calc = ObjectiveCalculator(self.df_loop, {}, {})
        
        # Empty result
        empty_df = pd.DataFrame({"displacement": [], "force": []})
        with self.assertRaises(FatalOptimizationError):
            calc.evaluate(empty_df, {})
            
        # Too few points
        short_df = pd.DataFrame({"displacement": [0.0], "force": [0.0]})
        with self.assertRaises(FatalOptimizationError):
            calc.evaluate(short_df, {})

if __name__ == "__main__":
    unittest.main()
