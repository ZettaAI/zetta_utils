"""Basic type definitions used for type annotations."""
from __future__ import annotations

from typing import Union, Any, Tuple
import torch
import numpy.typing as npt
import typeguard


Array = Union[torch.Tensor, npt.NDArray]
Slice3D = Tuple[slice, slice, slice]
IntVec3D = Tuple[int, int, int]
FloatVec3D = Tuple[int, int, int]
Vec3D = IntVec3D


def check_type(obj: Any, cls: Any) -> bool:
    """Type checking that works for type generics"""
    result = True
    try:
        typeguard.check_type("value", obj, cls)
    except TypeError:
        result = False
    return result
