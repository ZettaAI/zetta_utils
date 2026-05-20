import importlib
import importlib.util
import pkgutil

import zetta_utils as zu


def test_load_all_modules():
    zu.load_all_modules()  # pylint: disable=protected-access


def _iter_all_modules(pkg_name: str):
    pkg = importlib.import_module(pkg_name)
    if not hasattr(pkg, "__path__"):
        return
    for mi in pkgutil.iter_modules(pkg.__path__, prefix=pkg.__name__ + "."):
        # Orphaned .pyc artifacts from removed sources surface as
        # discoverable modules; skip anything without a .py origin.
        spec = importlib.util.find_spec(mi.name)
        if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
            continue
        yield mi.name
        if mi.ispkg:
            yield from _iter_all_modules(mi.name)


def test_every_submodule_imports_cleanly(monkeypatch):
    """Walk every zetta_utils submodule and confirm it imports without error.

    load_all_modules() only pulls in what preload/all.py names explicitly;
    after the lazy-module refactor that no longer cascades into every
    submodule via __init__.py. Walking here surfaces import-time breakage
    that would otherwise only show up in production.

    A few modules read env vars at module scope (set in production before
    zutils starts). Stub them with dummy values so the walk doesn't trip.
    """
    monkeypatch.setenv("SLACK_BOT_TOKEN", "test-token")
    monkeypatch.setenv("SLACK_CHANNEL", "test-channel")

    failures: list[tuple[str, str]] = []
    for name in _iter_all_modules(zu.__name__):
        try:
            importlib.import_module(name)
        except Exception as e:  # pylint: disable=broad-except
            failures.append((name, f"{type(e).__name__}: {e}"))
    assert not failures, "\n".join(f"{n}: {msg}" for n, msg in failures)


def test_load_apis():
    # this is how user would do it, but `import *`
    # is only allowed at top level, thus use exec:
    exec("from zetta_utils.api.v0 import *")  # pylint: disable=exec-used
