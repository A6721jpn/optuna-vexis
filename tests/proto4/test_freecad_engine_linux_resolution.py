from __future__ import annotations

import builtins
from unittest.mock import patch


def test_get_freecad_uses_linux_default_bin(monkeypatch) -> None:
    import proto4_claude.freecad_engine as fe

    monkeypatch.delenv("FREECAD_BIN", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", "")
    monkeypatch.setenv("CONDA_PREFIXES", "")
    fe._FreeCAD = None

    class DummyFreeCAD:
        pass

    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "FreeCAD":
            return DummyFreeCAD
        return original_import(name, *args, **kwargs)

    with patch("proto4_claude.freecad_engine.os.name", "posix"):
        with patch("proto4_claude.freecad_engine.Path.exists", return_value=True):
            with patch("builtins.__import__", side_effect=_fake_import):
                mod = fe._get_freecad()

    assert mod is DummyFreeCAD
