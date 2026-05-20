# pylint: disable=missing-docstring
"""Coverage for the make_lazy_module helper."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from zetta_utils.common.lazy import make_lazy_module


def _make_pkg(tmp_path: Path, name: str) -> Path:
    """Create a real importable package with sub.py + helper.py for tests."""
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "sub.py").write_text("VALUE = 42\n")
    (pkg / "helper.py").write_text("def helper_fn():\n    return 'helper'\n")
    return pkg


def test_subpackage_attribute_resolves(tmp_path, monkeypatch):
    _make_pkg(tmp_path, "lazy_pkg_a")
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg_mod = __import__("lazy_pkg_a")
    pkg_globals: dict = pkg_mod.__dict__
    getattr_, _dir = make_lazy_module(pkg_mod.__name__, pkg_globals, subpackages=("sub",))
    setattr(pkg_mod, "__getattr__", getattr_)
    try:
        assert pkg_mod.sub.VALUE == 42
    finally:
        sys.modules.pop("lazy_pkg_a.sub", None)
        sys.modules.pop("lazy_pkg_a", None)


def test_named_reexport_resolves(tmp_path, monkeypatch):
    _make_pkg(tmp_path, "lazy_pkg_b")
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg_mod = __import__("lazy_pkg_b")
    getattr_, _dir = make_lazy_module(
        pkg_mod.__name__,
        pkg_mod.__dict__,
        reexports_by_module={".helper": ("helper_fn",)},
    )
    setattr(pkg_mod, "__getattr__", getattr_)
    try:
        assert pkg_mod.helper_fn() == "helper"
    finally:
        sys.modules.pop("lazy_pkg_b.helper", None)
        sys.modules.pop("lazy_pkg_b", None)


def test_unknown_name_falls_back_to_submodule_import(tmp_path, monkeypatch):
    """A name not in subpackages or reexports gets tried as a real submodule."""
    _make_pkg(tmp_path, "lazy_pkg_c")
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg_mod = __import__("lazy_pkg_c")
    getattr_, _dir = make_lazy_module(pkg_mod.__name__, pkg_mod.__dict__)
    setattr(pkg_mod, "__getattr__", getattr_)
    try:
        # `helper` is not declared in subpackages or reexports but exists as
        # a submodule, so the fallback import path resolves it.
        assert pkg_mod.helper.helper_fn() == "helper"
    finally:
        sys.modules.pop("lazy_pkg_c.helper", None)
        sys.modules.pop("lazy_pkg_c", None)


def test_dunder_name_does_not_trigger_fallback(tmp_path, monkeypatch):
    """Names starting with `_` skip the fallback import path; raise immediately."""
    _make_pkg(tmp_path, "lazy_pkg_d")
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg_mod = __import__("lazy_pkg_d")
    getattr_, _dir = make_lazy_module(pkg_mod.__name__, pkg_mod.__dict__)
    setattr(pkg_mod, "__getattr__", getattr_)
    try:
        with pytest.raises(AttributeError, match="has no attribute '_private'"):
            getattr(pkg_mod, "_private")
    finally:
        sys.modules.pop("lazy_pkg_d", None)


def test_nonexistent_name_raises_attributeerror(tmp_path, monkeypatch):
    """A name with no matching subpackage / reexport / file raises AttributeError."""
    _make_pkg(tmp_path, "lazy_pkg_e")
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg_mod = __import__("lazy_pkg_e")
    getattr_, _dir = make_lazy_module(pkg_mod.__name__, pkg_mod.__dict__)
    setattr(pkg_mod, "__getattr__", getattr_)
    try:
        with pytest.raises(AttributeError, match="has no attribute 'totally_made_up'"):
            getattr(pkg_mod, "totally_made_up")
    finally:
        sys.modules.pop("lazy_pkg_e", None)


def test_dir_lists_declared_names(tmp_path, monkeypatch):
    """__dir__ returns the union of package globals and declared lazy names."""
    _make_pkg(tmp_path, "lazy_pkg_f")
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg_mod = __import__("lazy_pkg_f")
    _g, dir_ = make_lazy_module(
        pkg_mod.__name__,
        pkg_mod.__dict__,
        subpackages=("sub",),
        reexports_by_module={".helper": ("helper_fn",)},
    )
    listing = dir_()
    assert "sub" in listing
    assert "helper_fn" in listing


def test_resolved_attribute_is_cached(tmp_path, monkeypatch):
    """Second access reads from globals; doesn't re-import."""
    _make_pkg(tmp_path, "lazy_pkg_g")
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg_mod = __import__("lazy_pkg_g")
    getattr_, _dir = make_lazy_module(pkg_mod.__name__, pkg_mod.__dict__, subpackages=("sub",))
    setattr(pkg_mod, "__getattr__", getattr_)
    try:
        first = pkg_mod.sub
        # After first access, "sub" lives in pkg_mod.__dict__ so __getattr__
        # is not called again.
        assert pkg_mod.__dict__["sub"] is first
    finally:
        sys.modules.pop("lazy_pkg_g.sub", None)
        sys.modules.pop("lazy_pkg_g", None)
