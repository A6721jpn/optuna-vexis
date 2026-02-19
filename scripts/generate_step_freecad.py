"""
FreeCAD STEP generator for harness.

Uses proto4-codex FreecadEngine to open a model and export STEP.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from proto4_codex_alias import FreecadEngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fcstd", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fcstd = Path(args.fcstd)
    output = Path(args.output)

    engine = FreecadEngine(fcstd_path=fcstd)
    try:
        engine.open()
        engine.export_step(output)
    finally:
        engine.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
