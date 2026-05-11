"""Bulding objects from nested specs."""
from __future__ import annotations

import importlib
import logging
import threading
from collections import defaultdict
from typing import Callable, TypeVar

import attrs
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from . import constants

T = TypeVar("T", bound=Callable)

REGISTRY: dict[str, list[RegistryEntry]] = defaultdict(list)
MUTLIPROCESSING_INCOMPATIBLE_CLASSES: set[str] = set()

logger = logging.getLogger(__name__)

_lazy_lock = threading.Lock()
_lazy_attempted: set[str] = set()


@attrs.frozen
class RegistryEntry:
    fn: Callable
    allow_partial: bool
    allow_parallel: bool
    version_spec: SpecifierSet


def _try_lazy_import(name: str) -> bool:
    """Consult the static index and import any modules that should register `name`.

    Returns True iff at least one candidate module was imported as a result of
    this call. Idempotent across calls (a name is only attempted once).
    """
    # pylint: disable=import-outside-toplevel
    with _lazy_lock:
        if name in _lazy_attempted:
            return False
        _lazy_attempted.add(name)

        # Re-check under the lock: a concurrent caller may have populated it.
        if REGISTRY[name]:
            return False

        # Import the scanner here to avoid a circular import at module load.
        from .scan import get_index

        candidates = get_index().by_name().get(name, [])
        if not candidates:
            return False

        modules = {c.module for c in candidates}
        imported_any = False
        for module in modules:
            try:
                importlib.import_module(module)
                imported_any = True
                logger.debug("registry miss: name=%r → imported %s", name, module)
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(
                    "registry miss: name=%r → import of %s failed: %s",
                    name,
                    module,
                    e,
                )
        return imported_any


def get_matching_entry(
    name: str, version: str | Version = constants.DEFAULT_VERSION
) -> RegistryEntry:
    if not REGISTRY[name]:
        _try_lazy_import(name)

    version_ = Version(str(version))
    matches = []
    for e in REGISTRY[name]:
        if version_ in e.version_spec:
            matches.append(e)
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise RuntimeError(
            f"Multiple matches found for name '{name}', version '{version}': " f"{matches}"
        )
    else:
        raise RuntimeError(
            f"No matches found for name '{name}', version '{version}'. "
            f"Registry entries for '{name}': {REGISTRY[name]}"
        )


def register(
    name: str,
    allow_partial: bool = True,
    allow_parallel: bool = True,
    versions: str | SpecifierSet = constants.DEFAULT_VERSION_SPEC,
) -> Callable[[T], T]:
    """Decorator for registering classes to be buildable.

    :param name: Name which will be used for to indicate an object of the
        decorated type.
    :param allow_partial: Whether to allow `@mode: "partial"`.
    """

    version_spec = SpecifierSet(str(versions))

    # Check if the same name with the same version spec is already present
    for e in REGISTRY[name]:
        if e.version_spec == version_spec:
            raise RuntimeError(
                f"`builder` primitive with name '{name}' is already registered "
                f"for version '{version_spec}'"
            )

    def decorator(fn: T) -> T:
        nonlocal allow_parallel
        for k in MUTLIPROCESSING_INCOMPATIBLE_CLASSES:
            if fn.__module__ is not None and k.lower() in fn.__module__.lower():
                allow_parallel = False
                break

        REGISTRY[name].append(
            RegistryEntry(
                fn=fn,
                allow_partial=allow_partial,
                version_spec=version_spec,
                allow_parallel=allow_parallel,
            )
        )
        return fn

    return decorator


def unregister(
    name: str,
    fn: Callable,
    allow_partial: bool = True,
    allow_parallel: bool = True,
    versions: str | SpecifierSet = constants.DEFAULT_VERSION_SPEC,
):
    version_spec = SpecifierSet(str(versions))
    entry = RegistryEntry(
        fn=fn,
        allow_partial=allow_partial,
        allow_parallel=allow_parallel,
        version_spec=version_spec,
    )
    REGISTRY[name].remove(entry)


MUTLIPROCESSING_INCOMPATIBLE_CLASSES.add("mazepa")
MUTLIPROCESSING_INCOMPATIBLE_CLASSES.add("lightning")
