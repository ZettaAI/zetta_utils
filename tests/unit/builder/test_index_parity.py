# pylint: disable=missing-docstring
"""Static INDEX vs live REGISTRY parity check.

Relies on the session-scoped autouse fixture in tests/conftest.py that calls
setup_environment("all"), so REGISTRY is fully populated by the time these
tests run. Failures here mean either:
  - the scanner missed a registration form (treat as scanner bug), or
  - someone added a new dynamic-registration loop (add it to DYNAMIC_PREFIXES).
"""
from __future__ import annotations

from zetta_utils.builder.registry import REGISTRY
from zetta_utils.builder.scan import scan

# Names registered by import-time loops over external libraries.
# These are fundamentally not statically discoverable; the lookup-fallback
# path will need an explicit handler for each.
DYNAMIC_PREFIXES: tuple[str, ...] = (
    "np.",
    "imgaug.augmenters.",
    "torch.",  # convnet/architecture/primitives.py loops over torch.nn / torch.optim / etc
)


def _statically_discoverable(name: str) -> bool:
    return not any(name.startswith(p) for p in DYNAMIC_PREFIXES)


def _from_zetta_utils(name: str) -> bool:
    """True iff at least one registered fn for `name` originates in zetta_utils.

    Filters out test-fixture leaks (entries whose fn.__module__ starts with
    `tests.` or `__main__`) so this check is stable regardless of test order.
    Also skips lambdas — they're never statically discoverable, so seeing one
    in REGISTRY just means the test suite registered something dynamically.
    """
    for entry in REGISTRY[name]:
        mod = getattr(entry.fn, "__module__", "") or ""
        qualname = getattr(entry.fn, "__qualname__", "") or ""
        if "<lambda>" in qualname:
            continue
        if mod.startswith("zetta_utils"):
            return True
    return False


def test_scan_covers_all_static_registrations():
    """Every zetta_utils name in REGISTRY (excluding known-dynamic) appears in INDEX."""
    index_names = {e.name for e in scan(use_cache=False).entries}
    registry_static = {n for n in REGISTRY if _statically_discoverable(n) and _from_zetta_utils(n)}

    missing = registry_static - index_names
    assert not missing, (
        f"{len(missing)} registry name(s) not discovered by static scan; "
        f"sample: {sorted(missing)[:10]}"
    )


def test_scan_finds_at_least_as_much_as_known_baseline():
    """Catch silent regressions: scan should keep finding hundreds of entries."""
    result = scan(use_cache=False)
    # Floor chosen well below current count (~317) to absorb minor refactors
    # without false alarms; bump it up if it turns out to be too lax.
    assert len(result.entries) >= 250


def test_scan_warnings_are_only_known_dynamic_loops():
    """Any scanner warning must come from a known dynamic-registration site."""
    result = scan(use_cache=False)
    known_dynamic_files = {
        "imgaug.py",  # imgaug.augmenters loop
        "__init__.py",  # builder/__init__.py: numpy loop
        "primitives.py",  # tensor_ops/primitives loop (if present)
    }
    unexpected = [
        w for w in result.warnings if w.source_path.rsplit("/", 1)[-1] not in known_dynamic_files
    ]
    assert not unexpected, (
        "unexpected scanner warnings; new dynamic registration site? "
        f"{[(w.source_path, w.lineno, w.reason) for w in unexpected[:5]]}"
    )
