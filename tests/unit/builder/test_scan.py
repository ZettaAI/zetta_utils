# pylint: disable=missing-docstring,redefined-outer-name
import json
import os
import textwrap
from pathlib import Path

import pytest

from zetta_utils.builder import scan


def _write(root: Path, rel: str, body: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip())
    return path


@pytest.fixture
def pkg(tmp_path: Path) -> Path:
    """Create a fake package layout under tmp_path/pkg/."""
    root = tmp_path / "pkg"
    _write(root, "__init__.py", "")
    return root


def test_decorator_form_extracted(pkg, monkeypatch):
    _write(
        pkg,
        "mod_a.py",
        """
        from zetta_utils.builder import register

        @register("foo")
        def make_foo():
            return 1
        """,
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(pkg.parent / "cache"))
    result = scan.scan(roots=[pkg], use_cache=False)
    by_name = result.by_name()
    assert "foo" in by_name
    [entry] = by_name["foo"]
    assert entry.qualname == "make_foo"
    assert entry.module == "pkg.mod_a"
    assert entry.allow_partial is True
    assert entry.allow_parallel is True
    assert entry.versions == ">=0.0.0"


def test_qualified_register_attribute(pkg, monkeypatch):
    _write(
        pkg,
        "mod_b.py",
        """
        import zetta_utils.builder as builder

        @builder.register("bar")
        class Bar:
            pass
        """,
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(pkg.parent / "cache"))
    result = scan.scan(roots=[pkg], use_cache=False)
    [entry] = result.by_name()["bar"]
    assert entry.qualname == "Bar"


def test_kwargs_extracted(pkg, monkeypatch):
    _write(
        pkg,
        "mod_c.py",
        """
        from zetta_utils.builder import register

        @register("baz", versions=">=1.0", allow_partial=False, allow_parallel=False)
        def make_baz():
            ...
        """,
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(pkg.parent / "cache"))
    result = scan.scan(roots=[pkg], use_cache=False)
    [entry] = result.by_name()["baz"]
    assert entry.versions == ">=1.0"
    assert entry.allow_partial is False
    assert entry.allow_parallel is False


def test_direct_call_form(pkg, monkeypatch):
    _write(
        pkg,
        "mod_d.py",
        """
        from zetta_utils.builder import register

        class Q:
            @classmethod
            def from_path(cls, p):
                return cls()

        register("Q.from_path")(Q.from_path)
        """,
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(pkg.parent / "cache"))
    result = scan.scan(roots=[pkg], use_cache=False)
    by_name = result.by_name()
    assert "Q.from_path" in by_name
    [entry] = by_name["Q.from_path"]
    assert entry.qualname == "Q.from_path"
    assert entry.module == "pkg.mod_d"


def test_non_literal_name_warns(pkg, monkeypatch):
    _write(
        pkg,
        "mod_e.py",
        """
        from zetta_utils.builder import register

        for x in ["a", "b"]:
            register(f"dyn.{x}")(lambda: None)
        """,
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(pkg.parent / "cache"))
    result = scan.scan(roots=[pkg], use_cache=False)
    assert "dyn.a" not in result.by_name()
    assert any("not a string literal" in w.reason for w in result.warnings)


def test_non_literal_kwargs_warns_but_emits(pkg, monkeypatch):
    _write(
        pkg,
        "mod_f.py",
        """
        from zetta_utils.builder import register

        VER = ">=2.0"

        @register("ver_dynamic", versions=VER)
        def f():
            ...
        """,
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(pkg.parent / "cache"))
    result = scan.scan(roots=[pkg], use_cache=False)
    assert "ver_dynamic" in result.by_name()
    assert any("non-literal kwargs" in w.reason for w in result.warnings)


def test_files_without_register_skipped(pkg, monkeypatch):
    _write(pkg, "noreg.py", "x = 1\n")
    _write(
        pkg,
        "withreg.py",
        """
        from zetta_utils.builder import register

        @register("y")
        def y():
            ...
        """,
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(pkg.parent / "cache"))
    result = scan.scan(roots=[pkg], use_cache=False)
    assert result.candidate_files == 1
    assert "y" in result.by_name()


def test_cache_roundtrip(pkg, tmp_path, monkeypatch):
    _write(
        pkg,
        "mod_g.py",
        """
        from zetta_utils.builder import register

        @register("cached")
        def f():
            ...
        """,
    )
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_dir))

    cold = scan.scan(roots=[pkg], use_cache=True)
    cache_file = cache_dir / "zetta_utils" / "builder_index.json"
    assert cache_file.exists()
    cached_payload = json.loads(cache_file.read_text())
    assert any(e["name"] == "cached" for e in cached_payload["entries"])

    warm = scan.scan(roots=[pkg], use_cache=True)
    assert {e.name for e in warm.entries} == {e.name for e in cold.entries}


def test_cache_invalidated_on_mtime_change(pkg, tmp_path, monkeypatch):
    target = _write(
        pkg,
        "mod_h.py",
        """
        from zetta_utils.builder import register

        @register("v1")
        def f():
            ...
        """,
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    scan.scan(roots=[pkg], use_cache=True)

    target.write_text(
        textwrap.dedent(
            """
            from zetta_utils.builder import register

            @register("v2")
            def f():
                ...
            """
        ).lstrip()
    )
    # Bump mtime to defeat second-resolution stat cache.
    new_mtime = target.stat().st_mtime + 5
    os.utime(target, (new_mtime, new_mtime))

    refreshed = scan.scan(roots=[pkg], use_cache=True)
    names = set(refreshed.by_name())
    assert "v2" in names
    assert "v1" not in names


def test_get_index_caches_in_process(monkeypatch):
    """get_index() returns the same object on repeat calls without refresh."""
    monkeypatch.setattr(scan, "_INDEX", None)
    a = scan.get_index()
    b = scan.get_index()
    assert a is b
    c = scan.get_index(refresh=True)
    assert c is not a
