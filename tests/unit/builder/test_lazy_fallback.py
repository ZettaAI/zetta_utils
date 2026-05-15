# pylint: disable=missing-docstring,redefined-outer-name,protected-access
"""Lookup-miss fallback in get_matching_entry.

If a name is already in REGISTRY, no fallback runs. If the name is unknown,
the registry consults the static INDEX and imports the candidate module; the
module's import-time @register side effect populates REGISTRY.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from zetta_utils.builder import registry, scan
from zetta_utils.builder.registry import REGISTRY, get_matching_entry


@pytest.fixture(autouse=True)
def reset_lazy_state(monkeypatch):
    """Reset the attempted-names cache so each test sees a fresh fallback path."""
    monkeypatch.setattr(registry, "_lazy_attempted", set())
    yield


@pytest.fixture
def lazy_spy(mocker):
    """Spy on _try_lazy_import so tests can count fallback invocations."""
    return mocker.spy(registry, "_try_lazy_import")


def _install_fake_index(monkeypatch, entries: list[scan.IndexEntry]):
    fake = scan.ScanResult(
        entries=tuple(entries),
        warnings=(),
        scanned_files=0,
        candidate_files=0,
        elapsed_s=0.0,
    )
    monkeypatch.setattr(scan, "_INDEX", fake)


def _make_lazy_pkg(tmp_path: Path, name: str, register_name: str) -> str:
    """Create an importable package with a single registration; return module path."""
    pkg_dir = tmp_path / name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "lazy_mod.py").write_text(
        textwrap.dedent(
            f"""
            from zetta_utils.builder import register

            @register({register_name!r})
            def lazy_target():
                return "ok-{register_name}"
            """
        ).lstrip()
    )
    return f"{name}.lazy_mod"


def test_already_registered_name_skips_fallback(lazy_spy):
    """A name that's already in REGISTRY must not trigger any import."""
    # "lambda" is registered by built_in_registrations.py which is in
    # ALWAYS_EAGER, so it's guaranteed in REGISTRY regardless of preload mode.
    entry = get_matching_entry("lambda")
    assert entry.fn is not None
    assert lazy_spy.call_count == 0


def test_lazy_import_populates_registry(tmp_path, monkeypatch, lazy_spy):
    name = "phase1_lazy_target"
    module_path = _make_lazy_pkg(tmp_path, "phase1_pkg_a", name)
    monkeypatch.syspath_prepend(str(tmp_path))

    _install_fake_index(
        monkeypatch,
        [
            scan.IndexEntry(
                name=name,
                module=module_path,
                qualname="lazy_target",
                versions=">=0.0.0",
                allow_partial=True,
                allow_parallel=True,
                source_path="<test>",
                lineno=1,
            )
        ],
    )

    assert name not in REGISTRY or not REGISTRY[name]

    try:
        entry = get_matching_entry(name)
        assert entry.fn() == f"ok-{name}"
        assert REGISTRY[name]
        assert lazy_spy.call_count == 1
    finally:
        REGISTRY.pop(name, None)
        sys.modules.pop(module_path, None)


def test_unknown_name_raises_after_fallback_miss(monkeypatch):
    _install_fake_index(monkeypatch, [])  # empty index
    with pytest.raises(RuntimeError, match="No matches found for name 'definitely_not_a_thing'"):
        get_matching_entry("definitely_not_a_thing")


def test_repeat_lookup_does_not_reattempt_import(tmp_path, monkeypatch, lazy_spy):
    name = "phase1_once_only"
    module_path = _make_lazy_pkg(tmp_path, "phase1_pkg_b", name)
    monkeypatch.syspath_prepend(str(tmp_path))
    _install_fake_index(
        monkeypatch,
        [
            scan.IndexEntry(
                name=name,
                module=module_path,
                qualname="lazy_target",
                versions=">=0.0.0",
                allow_partial=True,
                allow_parallel=True,
                source_path="<test>",
                lineno=1,
            )
        ],
    )
    try:
        get_matching_entry(name)
        assert lazy_spy.call_count == 1
        get_matching_entry(name)
        assert lazy_spy.call_count == 1  # short-circuited: REGISTRY already has it
    finally:
        REGISTRY.pop(name, None)
        sys.modules.pop(module_path, None)


def test_failed_lazy_import_surfaces_as_no_matches(monkeypatch):
    _install_fake_index(
        monkeypatch,
        [
            scan.IndexEntry(
                name="phase1_broken",
                module="this_module_does_not_exist_xyz",
                qualname="x",
                versions=">=0.0.0",
                allow_partial=True,
                allow_parallel=True,
                source_path="<test>",
                lineno=1,
            )
        ],
    )
    with pytest.raises(RuntimeError, match="No matches found for name 'phase1_broken'"):
        get_matching_entry("phase1_broken")


def test_index_entry_present_but_module_does_not_register(tmp_path, monkeypatch):
    """Indexed module imports cleanly but doesn't actually register the expected name.

    Should fall through to the regular 'no matches' error after the lazy attempt.
    """
    pkg_dir = tmp_path / "phase1_pkg_c"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "noop.py").write_text("# imports cleanly, registers nothing\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    _install_fake_index(
        monkeypatch,
        [
            scan.IndexEntry(
                name="phase1_noop",
                module="phase1_pkg_c.noop",
                qualname="x",
                versions=">=0.0.0",
                allow_partial=True,
                allow_parallel=True,
                source_path="<test>",
                lineno=1,
            )
        ],
    )
    try:
        with pytest.raises(RuntimeError, match="No matches found for name 'phase1_noop'"):
            get_matching_entry("phase1_noop")
    finally:
        sys.modules.pop("phase1_pkg_c.noop", None)
