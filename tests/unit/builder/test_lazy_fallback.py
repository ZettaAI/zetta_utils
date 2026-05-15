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
from packaging.specifiers import SpecifierSet

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


def test_failed_lazy_import_logs_warning(tmp_path, monkeypatch, caplog):
    """When a candidate module raises during import, the registry logs and continues."""
    pkg_dir = tmp_path / "phase1_pkg_d"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "raises.py").write_text("raise RuntimeError('boom on import')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    _install_fake_index(
        monkeypatch,
        [
            scan.IndexEntry(
                name="phase1_raises",
                module="phase1_pkg_d.raises",
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
        with caplog.at_level("WARNING", logger="zetta_utils.builder.registry"):
            with pytest.raises(RuntimeError, match="No matches found for name 'phase1_raises'"):
                get_matching_entry("phase1_raises")
        assert any("import of phase1_pkg_d.raises failed" in r.message for r in caplog.records)
    finally:
        sys.modules.pop("phase1_pkg_d.raises", None)


def test_dynamic_resolver_fires_when_index_misses():
    """np.* / torch.* names go through the resolver because INDEX has no entry."""
    entry = get_matching_entry("np.allclose")
    assert callable(entry.fn)
    # Cached after first lookup, so REGISTRY now has it.
    assert REGISTRY["np.allclose"]


def test_dynamic_resolver_exception_is_swallowed(monkeypatch, caplog):
    """If a resolver raises, the registry logs a warning and falls through."""

    def angry_resolver(_name):
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(registry, "_dynamic_resolvers", [("phase1_angry.", angry_resolver)])
    with caplog.at_level("WARNING", logger="zetta_utils.builder.registry"):
        with pytest.raises(RuntimeError, match="No matches found for name 'phase1_angry.foo'"):
            get_matching_entry("phase1_angry.foo")
    assert any("resolver for 'phase1_angry.'" in r.message for r in caplog.records)


def test_lookup_with_multiple_matches_raises(monkeypatch):
    """Two registered entries whose version_spec both contain DEFAULT_VERSION → error."""
    name = "phase1_multi"
    monkeypatch.setattr(registry, "REGISTRY", registry.defaultdict(list))
    registry.REGISTRY[name] = [
        registry.RegistryEntry(
            fn=lambda: "a",
            allow_partial=True,
            allow_parallel=True,
            version_spec=SpecifierSet(">=0.0.0"),
        ),
        registry.RegistryEntry(
            fn=lambda: "b",
            allow_partial=True,
            allow_parallel=True,
            version_spec=SpecifierSet(">=0.0.0"),
        ),
    ]
    with pytest.raises(RuntimeError, match="Multiple matches found for name 'phase1_multi'"):
        get_matching_entry(name)


def test_repeat_attempted_short_circuits(monkeypatch, mocker):
    """If a name is already in _lazy_attempted, the function returns False immediately."""
    monkeypatch.setattr(registry, "_lazy_attempted", {"already_tried"})
    spy = mocker.spy(registry, "_try_lazy_import")
    assert registry._try_lazy_import("already_tried") is False
    assert spy.call_count == 1  # called once, returned False without doing work


def test_under_lock_recheck_short_circuits(monkeypatch):
    """If REGISTRY[name] gets populated between attempted-set and the under-lock
    re-check, _try_lazy_import bails without importing anything (race guard)."""
    name = "phase1_prepop"
    monkeypatch.setattr(registry, "REGISTRY", registry.defaultdict(list))
    monkeypatch.setattr(registry, "_lazy_attempted", set())
    # Populate REGISTRY before the call; the in-set check is skipped (name not
    # attempted yet) but the post-add REGISTRY check returns False.
    registry.REGISTRY[name].append(
        registry.RegistryEntry(
            fn=lambda: "x",
            allow_partial=True,
            allow_parallel=True,
            version_spec=SpecifierSet(">=0.0.0"),
        )
    )
    assert registry._try_lazy_import(name) is False


def test_register_duplicate_version_raises(monkeypatch):
    """register() with a name+version already in REGISTRY raises immediately."""
    name = "phase1_dup"
    monkeypatch.setattr(registry, "REGISTRY", registry.defaultdict(list))
    registry.REGISTRY[name].append(
        registry.RegistryEntry(
            fn=lambda: 1,
            allow_partial=True,
            allow_parallel=True,
            version_spec=SpecifierSet(">=0.0.0"),
        )
    )
    with pytest.raises(RuntimeError, match="is already registered"):
        registry.register(name)(lambda: 2)


def test_unregister_removes_entry(monkeypatch):
    """unregister() pulls the matching entry back out of REGISTRY."""
    name = "phase1_unreg"
    monkeypatch.setattr(registry, "REGISTRY", registry.defaultdict(list))

    def victim():
        return "v"

    registry.register(name)(victim)
    assert registry.REGISTRY[name]
    registry.unregister(name=name, fn=victim)
    assert registry.REGISTRY[name] == []
