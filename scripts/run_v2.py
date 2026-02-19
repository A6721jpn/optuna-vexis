"""Entrypoint for Production v2 package (src/v2)."""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = PROJECT_ROOT / "src" / "v2"


def main() -> int:
    try:
        spec = importlib.util.spec_from_file_location(
            "v2",
            str(PKG_DIR / "__init__.py"),
            submodule_search_locations=[str(PKG_DIR)],
        )
        if spec is None:
            raise ImportError("Failed to create spec for v2 package")

        mod = importlib.util.module_from_spec(spec)
        sys.modules["v2"] = mod
        spec.loader.exec_module(mod)

        for py_file in PKG_DIR.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            sub_name = f"v2.{py_file.stem}"
            if sub_name in sys.modules:
                continue
            sub_spec = importlib.util.spec_from_file_location(sub_name, str(py_file))
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sys.modules[sub_name] = sub_mod
            sub_spec.loader.exec_module(sub_mod)

        from v2.runner import main as runner_main

        return runner_main()
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
