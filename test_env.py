import sys
print(f"Python: {sys.version}")
try:
    import optuna
    print(f"Optuna: {optuna.__version__}")
except ImportError as e:
    print(f"Optuna not found: {e}")

try:
    import pandas
    print(f"Pandas: {pandas.__version__}")
except ImportError as e:
    print(f"Pandas not found: {e}")

try:
    import scipy
    print(f"Scipy: {scipy.__version__}")
except ImportError as e:
    print(f"Scipy not found: {e}")

try:
    import FreeCAD
    print(f"FreeCAD: {FreeCAD.__file__}")
except ImportError as e:
    print(f"FreeCAD not found: {e}")
