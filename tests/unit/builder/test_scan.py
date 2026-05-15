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


def test_syntax_error_emits_warning(pkg, monkeypatch):
    """A file that fails to parse contributes a parse-error ScanWarning."""
    _write(pkg, "broken.py", "register(\n")  # truncated, unparsable
    monkeypatch.setenv("XDG_CACHE_HOME", str(pkg.parent / "cache"))
    result = scan.scan(roots=[pkg], use_cache=False)
    assert any("parse error" in w.reason for w in result.warnings)


def test_nonexistent_root_skipped(tmp_path, monkeypatch):
    """A configured root that doesn't exist is silently skipped."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    result = scan.scan(roots=[tmp_path / "does_not_exist"], use_cache=False)
    assert result.entries == ()
    assert result.candidate_files == 0


def test_large_file_path_taken(pkg, monkeypatch):
    """Files >64KB hit the second branch in _candidate_files (read full bytes)."""
    # 70 KB of harmless content + one register call past the head-read window
    body = "# padding\n" * 10000 + "from x import register\nregister('big_one')(lambda: 1)\n"
    _write(pkg, "big.py", body)
    monkeypatch.setenv("XDG_CACHE_HOME", str(pkg.parent / "cache"))
    result = scan.scan(roots=[pkg], use_cache=False)
    assert "big_one" in result.by_name()


def test_unreadable_file_emits_warning(pkg, monkeypatch, mocker):
    """A file that becomes unreadable between candidate-detection and full read
    records a warning. Simulated via Path.read_text raising on the target."""
    _write(pkg, "noread.py", "register('x')(lambda: 1)\n")
    monkeypatch.setenv("XDG_CACHE_HOME", str(pkg.parent / "cache"))

    real_read_text = Path.read_text

    def maybe_fail(self, *a, **kw):
        if self.name == "noread.py":
            raise OSError("simulated read failure")
        return real_read_text(self, *a, **kw)

    mocker.patch.object(Path, "read_text", maybe_fail)
    result = scan.scan(roots=[pkg], use_cache=False)
    assert any("read error" in w.reason for w in result.warnings)


def test_cache_corrupt_json_treated_as_miss(pkg, tmp_path, monkeypatch):
    """A corrupt cache file is silently invalidated; scan recomputes."""
    _write(pkg, "mod.py", "register('cm')(lambda: 1)\n")
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_dir))
    cache_path = cache_dir / "zetta_utils" / "builder_index.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("not valid json {{{")
    result = scan.scan(roots=[pkg], use_cache=True)
    assert "cm" in result.by_name()


def test_cache_wrong_shape_treated_as_miss(pkg, tmp_path, monkeypatch):
    """Cache JSON with the wrong schema is invalidated via TypeError handling."""
    _write(pkg, "mod.py", "register('cs')(lambda: 1)\n")
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_dir))
    cache_path = cache_dir / "zetta_utils" / "builder_index.json"
    cache_path.parent.mkdir(parents=True)
    # Compute the key the same way scan() does — only files matching the
    # register-grep filter — so the key check passes and we reach the
    # TypeError-handled IndexEntry unpacking.
    candidates = scan._candidate_files([pkg])  # pylint: disable=protected-access
    key = scan._cache_key(candidates)  # pylint: disable=protected-access
    cache_path.write_text(
        json.dumps(
            {
                "key": key,
                "entries": [{"unexpected_field": "x"}],
                "warnings": [],
            }
        )
    )
    result = scan.scan(roots=[pkg], use_cache=True)
    assert "cs" in result.by_name()


def test_scan_multi_root_resolves_each_path_to_correct_root(tmp_path, monkeypatch):
    """When multiple roots are passed, package_root resolution iterates through
    them, hitting the ValueError-continue branch for roots that don't contain
    a given file."""
    # Place 'other' deep enough that its parent is NOT an ancestor of pkg —
    # so relative_to(other.parent) raises ValueError for files in pkg, and
    # the loop continues to the second root.
    deep = tmp_path / "deep" / "sub" / "inner"
    deep.mkdir(parents=True)
    (deep / "__init__.py").write_text("")
    pkg = tmp_path / "lazy_pkg_main"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text("from x import register\nregister('mr')(lambda: 1)\n")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    result = scan.scan(roots=[deep, pkg], use_cache=False)
    [entry] = result.by_name()["mr"]
    assert entry.module == "lazy_pkg_main.mod"


def test_cache_write_oserror_swallowed(pkg, tmp_path, monkeypatch, mocker):
    """A failed cache write logs at DEBUG and doesn't raise."""
    _write(pkg, "mod.py", "register('cw')(lambda: 1)\n")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    # Patch write_text on the target cache file path so the failure happens
    # inside _save_cache's try/except.
    real_write_text = Path.write_text

    def maybe_fail(self, *a, **kw):
        if self.name == "builder_index.json":
            raise OSError("simulated write failure")
        return real_write_text(self, *a, **kw)

    mocker.patch.object(Path, "write_text", maybe_fail)
    result = scan.scan(roots=[pkg], use_cache=True)
    # Scan still returns a valid result; the cache write just got logged.
    assert "cw" in result.by_name()


def test_register_call_with_kwargs_warns(pkg, monkeypatch):
    """A register(...) call with **kwargs produces an unresolved-kwargs warning."""
    _write(
        pkg,
        "kw.py",
        """
        from zetta_utils.builder import register

        EXTRA = {"versions": ">=1.0"}

        @register("with_kw", **EXTRA)
        def f():
            ...
        """,
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(pkg.parent / "cache"))
    result = scan.scan(roots=[pkg], use_cache=False)
    assert "with_kw" in result.by_name()
    assert any("**kwargs" in w.reason for w in result.warnings)
