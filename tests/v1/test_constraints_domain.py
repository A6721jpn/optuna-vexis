from __future__ import annotations

import importlib.util
import textwrap
import sys
from pathlib import Path

import pytest


def _ensure_v1_package_loaded() -> None:
    if "v1" in sys.modules:
        return
    project_root = Path(__file__).resolve().parents[2]
    pkg_dir = project_root / "src" / "v1"
    spec = importlib.util.spec_from_file_location(
        "v1",
        str(pkg_dir / "__init__.py"),
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load v1 package for tests")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v1"] = mod
    spec.loader.exec_module(mod)


_ensure_v1_package_loaded()

from v1.config import BoundsSpec, FreecadSpec, V1Config, load_config  # noqa: E402
from v1.runner import _convert_physical_bounds_to_ratio  # noqa: E402


def test_load_config_accepts_physical_constraints_domain(tmp_path: Path) -> None:
    optimizer_yaml = tmp_path / "optimizer.yaml"
    limits_yaml = tmp_path / "limits.yaml"

    optimizer_yaml.write_text(
        textwrap.dedent(
            """
            optimization:
              sampler: RANDOM
              max_trials: 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    limits_yaml.write_text(
        textwrap.dedent(
            """
            freecad:
              constraints_domain: physical
              constraints:
                HEIGHT:
                  min: 2.58
                  max: 2.62
                  base_value: 2.60
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(optimizer_yaml, limits_yaml)
    assert cfg.freecad.constraints_domain == "physical"
    assert cfg.bounds[0].name == "HEIGHT"
    assert cfg.bounds[0].min == pytest.approx(2.58)
    assert cfg.bounds[0].max == pytest.approx(2.62)
    assert cfg.bounds[0].base_value == pytest.approx(2.60)


def test_convert_physical_bounds_to_ratio_in_runner() -> None:
    cfg = V1Config(
        freecad=FreecadSpec(constraints_domain="physical"),
        bounds=[
            BoundsSpec(name="HEIGHT", min=2.58, max=2.62, base_value=2.60),
            BoundsSpec(name="SHOULDER-ANGLE-OUT", min=2.185, max=2.189, base_value=2.187),
        ],
    )

    _convert_physical_bounds_to_ratio(cfg)

    assert cfg.bounds[0].min == pytest.approx(2.58 / 2.60)
    assert cfg.bounds[0].max == pytest.approx(2.62 / 2.60)
    assert cfg.bounds[1].min == pytest.approx(2.185 / 2.187)
    assert cfg.bounds[1].max == pytest.approx(2.189 / 2.187)


def test_load_config_rejects_unknown_constraints_domain(tmp_path: Path) -> None:
    optimizer_yaml = tmp_path / "optimizer.yaml"
    limits_yaml = tmp_path / "limits.yaml"

    optimizer_yaml.write_text(
        "optimization:\n  sampler: RANDOM\n  max_trials: 1\n",
        encoding="utf-8",
    )
    limits_yaml.write_text(
        textwrap.dedent(
            """
            freecad:
              constraints_domain: absolute
              constraints:
                HEIGHT:
                  min: 2.58
                  max: 2.62
                  base_value: 2.60
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="freecad.constraints_domain"):
        load_config(optimizer_yaml, limits_yaml)
