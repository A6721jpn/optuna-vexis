import sys
import os

# Add Library/bin to PATH for DLL loading
conda_prefix = r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad"
lib_bin = os.path.join(conda_prefix, "Library", "bin")
os.environ["PATH"] = lib_bin + os.pathsep + os.environ["PATH"]

print(f"Added {lib_bin} to PATH")

try:
    import numpy
    print(f"NumPy imported successfully: {numpy.__version__}")
except ImportError as e:
    print(f"NumPy import failed: {e}")
except Exception as e:
    print(f"NumPy import crashed: {e}")
