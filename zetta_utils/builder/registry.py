"""Bulding objects from nested specs."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, TypeVar

import attrs
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from . import constants

T = TypeVar("T", bound=Callable)

REGISTRY: dict[str, list[RegistryEntry]] = defaultdict(list)
MULTIPROCESSING_INCOMPATIBLE_CLASSES: set[str] = set()


@attrs.frozen
class RegistryEntry:
    fn: Callable
    allow_partial: bool
    allow_parallel: bool
    version_spec: SpecifierSet


def get_matching_entry(
    name: str, version: str | Version = constants.DEFAULT_VERSION
) -> RegistryEntry:
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
        for k in MULTIPROCESSING_INCOMPATIBLE_CLASSES:
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
