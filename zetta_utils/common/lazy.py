"""Helper for lazy attribute resolution in package ``__init__.py`` (PEP 562).

A package that re-exports many submodules and named symbols pays a transitive
import cost on first ``import package`` even when the caller wants only one
symbol. ``make_lazy_module`` lets each ``__init__.py`` declare what it
exposes without actually loading any of it; each name is imported on first
attribute access and cached in the module's globals.

Typical usage in a package's ``__init__.py``::

    from zetta_utils.common.lazy import make_lazy_module

    _LAZY_SUBPACKAGES = ("alignment", "meshing")
    _LAZY_REEXPORTS = {
        ".meshing": ("MakeMeshFragsOperation", "build_generate_meshes_flow"),
        ".shards":  ("compute_shard_params_for_hashed", "MakeShardsFlowSchema"),
    }
    __getattr__, __dir__ = make_lazy_module(
        __name__, globals(), _LAZY_SUBPACKAGES, _LAZY_REEXPORTS,
    )

``__name__`` and ``globals()`` must be evaluated inside the calling module;
they cannot be moved into the helper because they bind to that module's
identity and namespace.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Iterable, Mapping


def make_lazy_module(
    package_name: str,
    package_globals: dict[str, Any],
    subpackages: Iterable[str] = (),
    reexports_by_module: Mapping[str, Iterable[str]] | None = None,
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build ``__getattr__`` and ``__dir__`` for lazy resolution.

    :param package_name: pass ``__name__`` from the calling ``__init__.py``.
    :param package_globals: pass ``globals()`` from the calling ``__init__.py``.
    :param subpackages: names exposed lazily as submodule attributes.
    :param reexports_by_module: ``{relative_module_path: [public_name, ...]}``.
    :return: ``(__getattr__, __dir__)`` to assign at module level.
    """
    sub_set = frozenset(subpackages)
    name_to_module: dict[str, str] = {}
    for module, names in (reexports_by_module or {}).items():
        for n in names:
            name_to_module[n] = module
    all_names = sorted(sub_set | name_to_module.keys())

    def __getattr__(name: str) -> Any:
        if name in sub_set:
            mod = importlib.import_module(f".{name}", package_name)
            package_globals[name] = mod
            return mod
        if name in name_to_module:
            mod = importlib.import_module(name_to_module[name], package_name)
            attr = getattr(mod, name)
            package_globals[name] = attr
            return attr
        # Fall back to importing `name` as a submodule of this package. The
        # pre-lazy `from .X import Y` had a side effect of binding `pkg.X`
        # as an attribute, so callers that access `pkg.X.Z` directly worked
        # for free. Mirror that here so we don't have to enumerate every
        # submodule explicitly.
        if not name.startswith("_"):
            try:
                mod = importlib.import_module(f".{name}", package_name)
            except ImportError:
                pass
            else:
                package_globals[name] = mod
                return mod
        raise AttributeError(f"module {package_name!r} has no attribute {name!r}")

    def __dir__() -> list[str]:
        return sorted(set(package_globals) | set(all_names))

    return __getattr__, __dir__
