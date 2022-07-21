"""Basic type definitions used for type annotations."""
from __future__ import annotations

from typing import Union, Any, Tuple, Literal
import torch
import numpy.typing as npt
import typeguard

# number_types = (int, float)
Number = Union[int, float]
Array = Union[torch.Tensor, npt.NDArray]
Slice3D = Tuple[slice, slice, slice]
FloatVec3D = Tuple[float, float, float]
IntVec3D = Tuple[int, int, int]
Vec3D = Tuple[Number, Number, Number]
Coord3D = Union[str, Vec3D]
DimIdx3D = Union[
    Literal[0],
    Literal[1],
    Literal[2],
]
DimName3D = Union[
    Literal["x"],
    Literal["y"],
    Literal["z"],
]
Dim3D = Union[DimName3D, DimIdx3D]
Padding3D = Tuple[
    Union[Number, Tuple[Number, Number]],
    Union[Number, Tuple[Number, Number]],
    Union[Number, Tuple[Number, Number]],
]


def check_type(obj: Any, cls: Any) -> bool:  # pragma: no cover
    """Type checking that works for type generics"""
    result = True
    try:
        typeguard.check_type("value", obj, cls)
    except TypeError:
        result = False
    return result
