"""Basic type definitions used for type annotations."""
from __future__ import annotations

from typing import Union, Any, Tuple, List
import typeguard

Slices3D = Tuple[slice, slice, slice]
# int acceptible cc: https://peps.python.org/pep-0484/#the-numeric-tower
Vec3D = Union[Tuple[float, float, float], Tuple[int, int, int], List[float], List[int]]


def check_type(obj: Any, cls: Any) -> bool:  # pragma: no cover
    """Type checking that works for type generics"""
    result = True
    try:
        typeguard.check_type("value", obj, cls)
    except TypeError:
        result = False
    return result
