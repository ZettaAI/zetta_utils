"""Basic type definitions used for type annotations."""
from __future__ import annotations

from typing import Union, Any, Tuple, List
import torch
import numpy.common.typing as npt
import typeguard

# number_types = (int, float)
Number = Union[int, float]
Tensor = Union[torch.Tensor, npt.NDArray]
Slices3D = Tuple[slice, slice, slice]
Vec3D = Union[Tuple[Number, Number, Number], List[Number]]


def check_type(obj: Any, cls: Any) -> bool:  # pragma: no cover
    """Type checking that works for type generics"""
    result = True
    try:
        typeguard.check_type("value", obj, cls)
    except TypeError:
        result = False
    return result
