"""Entrypoint for Production v2-claude package (src/v2-claude)."""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = PROJECT_ROOT / "src" / "v2-claude"
# Python module name uses underscore (v2_claude) since hyphens are invalid.
MOD_NAME = "v2_claude"


def main() -> int:
    try:
        spec = importlib.util.spec_from_file_location(
            MOD_NAME,
            str(PKG_DIR / "__init__.py"),
            submodule_search_locations=[str(PKG_DIR)],
        )
        if spec is None:
            raise ImportError(f"Failed to create spec for {MOD_NAME} package")

        mod = importlib.util.module_from_spec(spec)
        sys.modules[MOD_NAME] = mod
        spec.loader.exec_module(mod)

        for py_file in PKG_DIR.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            sub_name = f"{MOD_NAME}.{py_file.stem}"
            if sub_name in sys.modules:
                continue
            sub_spec = importlib.util.spec_from_file_location(sub_name, str(py_file))
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sys.modules[sub_name] = sub_mod
            sub_spec.loader.exec_module(sub_mod)

        runner = sys.modules[f"{MOD_NAME}.runner"]
        return runner.main()
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
