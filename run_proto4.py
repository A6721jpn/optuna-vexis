import sys
import os
import traceback

# Open log file immediately to catch early errors
with open("debug_run.log", "w", buffering=1) as log:
    def log_print(msg):
        print(msg)
        log.write(str(msg) + "\n")
        log.flush()

    try:
        log_print("Starting run_proto4.py...")
        log_print(f"Python executable: {sys.executable}")
        log_print(f"Python version: {sys.version}")
        
        import importlib.util
        from pathlib import Path
        
        project_root = Path(__file__).parent.resolve()
        log_print(f"Project root: {project_root}")
        
        # FIX: Add Library/bin to PATH for numpy/scipy DLLs
        conda_prefix = r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad"
        lib_bin = os.path.join(conda_prefix, "Library", "bin")
        os.environ["PATH"] = lib_bin + os.pathsep + os.environ["PATH"]
        log_print(f"Added {lib_bin} to PATH")
        
        sys.path.append(str(project_root / "src"))
        log_print(f"Added {project_root / 'src'} to sys.path")
        
        pkg_dir = project_root / "src" / "proto4-claude"
        log_print(f"Package dir: {pkg_dir}")
        
        if not pkg_dir.exists():
            raise FileNotFoundError(f"Package directory {pkg_dir} not found")
            
        # Create the package module
        log_print("Creating proto4_claude package...")
        spec = importlib.util.spec_from_file_location(
            "proto4_claude",
            str(pkg_dir / "__init__.py"),
            submodule_search_locations=[str(pkg_dir)]
        )
        if spec is None:
            raise ImportError("Failed to create spec for proto4_claude")
            
        mod = importlib.util.module_from_spec(spec)
        sys.modules["proto4_claude"] = mod
        log_print("Executing proto4_claude.__init__...")
        spec.loader.exec_module(mod)
        
        # Register submodules
        log_print("Registering submodules...")
        for py_file in pkg_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            sub_name = f"proto4_claude.{py_file.stem}"
            log_print(f"Loading spec for {sub_name} ({py_file.name})...")
            sub_spec = importlib.util.spec_from_file_location(sub_name, str(py_file))
            if sub_spec is None:
                 log_print(f"Warning: Failed to create spec for {sub_name}")
                 continue
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sys.modules[sub_name] = sub_mod
            log_print(f"Executing module {sub_name}...")
            sub_spec.loader.exec_module(sub_mod)
            log_print(f"Finished module {sub_name}")
            
        log_print("Submodules registered. Importing main...")
        
        from proto4_claude.runner import main
        log_print("Calling main()...")
        
        ret = main()
        log_print(f"main() returned {ret}")
        sys.exit(ret)
        
    except Exception:
        log_print("CRITICAL ERROR:")
        log.write(traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)
