"""
E2E Test Runner for Proto4
"""
import sys
import subprocess
import shutil
from pathlib import Path

def run_e2e_test():
    # Setup paths
    current_dir = Path(__file__).parent.resolve()
    project_root = current_dir.parent.parent.parent
    
    config_path = current_dir / "config" / "optimizer_config_test.yaml"
    limits_path = current_dir / "config" / "limitations_test.yaml"
    
    # Ensure clean slate for test output
    test_output_dir = project_root / "output" / "trials_test"
    test_db_path = test_output_dir / "optuna_study_proto4.db"
    
    print(f"--- Starting E2E Test ---")
    print(f"Config: {config_path}")
    print(f"Limits: {limits_path}")
    
    if test_db_path.exists():
        print(f"Cleaning old test DB: {test_db_path}")
        try:
            test_db_path.unlink()
        except PermissionError:
            print("Warning: Could not delete DB file. Ensure no other process is using it.")

    if test_output_dir.exists():
        shutil.rmtree(test_output_dir, ignore_errors=True)
    
    # Construct command to run the main pipeline with test configs
    # We use the same python interpreter as this script
    python_exe = sys.executable
    
    # Important: run_proto4.py is in project root.
    # But wait, run_proto4.py might hardcode config paths or use arguments.
    # Let's check run_proto4.py arguments. It accepts --config and --limits.
    
    run_script = project_root / "run_proto4.py"
    
    cmd = [
        python_exe,
        str(run_script),
        "--config", str(config_path.relative_to(project_root)),
        "--limits", str(limits_path.relative_to(project_root)),
        "--verbose"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        # Run process
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False # We handle return code manually
        )
        
        # Output handling
        print("--- STDOUT ---")
        print(result.stdout)
        
        if result.stderr:
            print("--- STDERR ---")
            print(result.stderr)
            
        if result.returncode == 0:
            print("\n[SUCCESS] Pipeline finished successfully.")
            # Verify artifacts
            if (test_output_dir / "trial_0" / "trial_info.json").exists():
                 print("[Pass] Trial 0 data generated.")
            else:
                 print("[Fail] Trial 0 data NOT found.")
                 return 1
                 
            if (test_output_dir / "trial_1" / "trial_info.json").exists():
                 print("[Pass] Trial 1 data generated.")
            else:
                 print("[Fail] Trial 1 data NOT found.")
                 return 1
                 
            return 0
        else:
            print(f"\n[FAILURE] Pipeline failed with return code {result.returncode}")
            return result.returncode

    except Exception as e:
        print(f"Test runner error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_e2e_test())
