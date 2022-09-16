"""Basic type definitions used for type annotations."""
from __future__ import annotations

from typing import Union, Any, Tuple, List, TypeVar
import torch
import numpy.typing as npt
import typeguard

Number = Union[int, float]
TensorTypeVar = TypeVar(
    "TensorTypeVar", torch.Tensor, npt.NDArray, Union[torch.Tensor, npt.NDArray]
)

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
