"""Bulding objects from nested specs."""
from __future__ import annotations

from typing import Any, Callable, Final, Optional, TypeVar, Union

import attrs

REGISTRY: dict[str, RegistryEntry] = {}


@attrs.frozen
class RegistryEntry:
    fn: Callable
    allow_partial: bool


T = TypeVar("T", bound=Callable)


def get_callable_from_name(name: str):  # pragma: no cover
    return REGISTRY[name].fn


def register(name: str, allow_partial: bool = True) -> Callable[[T], T]:
    """Decorator for registering classes to be buildable.

    :param name: Name which will be used for to indicate an object of the
        decorated type.
    :param allow_partial: Whether to allow `@mode: "partial"`.
    """

    if name in REGISTRY:
        raise RuntimeError(f"`builder` primitive with name '{name}' is already registered.")

    def decorator(fn: T) -> T:
        REGISTRY[name] = RegistryEntry(fn=fn, allow_partial=allow_partial)
        return fn

    return decorator
