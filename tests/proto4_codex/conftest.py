import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PKG_DIR = PROJECT_ROOT / "src" / "proto4-codex"

if "proto4_codex" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "proto4_codex",
        str(PKG_DIR / "__init__.py"),
        submodule_search_locations=[str(PKG_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["proto4_codex"] = mod
    spec.loader.exec_module(mod)

    for py_file in PKG_DIR.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        sub_name = f"proto4_codex.{py_file.stem}"
        if sub_name not in sys.modules:
            sub_spec = importlib.util.spec_from_file_location(sub_name, str(py_file))
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sys.modules[sub_name] = sub_mod
            sub_spec.loader.exec_module(sub_mod)

from proto4_codex.config import *  # noqa: F401,F403
from proto4_codex.freecad_engine import FreecadEngine  # noqa: F401
