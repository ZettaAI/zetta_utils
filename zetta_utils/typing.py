"""Basic type definitions used for type annotations."""
from __future__ import annotations

from typing import Any, List, Tuple, Union

import typeguard

Slices3D = Tuple[slice, slice, slice]
Vec3D = Union[Tuple[float, float, float], Tuple[int, int, int], List[float], List[int]]
IntVec3D = Union[Tuple[int, int, int], List[int]]


def check_type(obj: Any, cls: Any) -> bool:  # pragma: no cover
    """Type checking that works for type generics"""
    result = True
    try:
        typeguard.check_type("value", obj, cls)
    except TypeError:
        result = False
    return result
