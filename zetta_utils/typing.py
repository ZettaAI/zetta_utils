"""Basic type definitions used for type annotations."""
from __future__ import annotations

from inspect import currentframe
from types import FrameType, NoneType
from typing import Any, Generic, Tuple, Type, TypeVar, Union, get_args, overload

import numpy as np
import typeguard

Slices3D = Tuple[slice, slice, slice]

T = TypeVar("T", covariant=True, bound=float)

"""
Helper function to determine the original class for a generic subclass. This function
is necessary because `obj.__orig_class__` does not work during `__init__`.
Taken verbatim from https://github.com/Stewori/pytypes/blob/master/pytypes/type_util.py
"""


def get_orig_class(obj: Any) -> Type:
    try:
        return object.__getattribute__(obj, "__orig_class__")
    except AttributeError:
        cls = object.__getattribute__(obj, "__class__")
        # frame = currentframe().f_back.f_back unrolled for mypy
        frame = currentframe()
        assert type(frame) is FrameType
        frame = frame.f_back
        assert type(frame) is FrameType
        frame = frame.f_back
        try:
            while frame:
                try:
                    res = frame.f_locals["self"]
                    if res.__origin__ is cls:
                        return res
                except (KeyError, AttributeError):
                    frame = frame.f_back
        finally:
            del frame
        return NoneType


# dunder methods must be overwritten for type annotation reasons
class GenericVec3D(Generic[T]):  # pragma: no cover
    @property
    def dtype(self) -> Type:
        orig_class = get_orig_class(self)
        assert hasattr(orig_class, "__args__"), "GenericVec3D instantiated without type parameter"
        return orig_class.__args__[0]

    def __init__(self, *args: T):
        assert len(args) == 3, "Vec3D must have 3 elements"
        for arg in args:
            assert isinstance(arg, (int, float)), "Vec3D requires int/float arguments"
        self.vec = tuple(self.dtype(arg) for arg in args)

    # avoid returning Vec3D whenever sliced, defaulting to float or tuple
    def __getitem__(self, key) -> Union[T, Tuple[T, ...]]:
        return self.vec[key]

    #    def __iter__(self, other)

    def __eq__(self, other) -> bool:
        return self.vec == other.vec

    def __repr__(self) -> str:
        if self.dtype is int:
            return f"IntVec3D({self.vec[0]}, {self.vec[1]}, {self.vec[2]})"
        if self.dtype is float:
            return f"Vec3D({self.vec[0]}, {self.vec[1]}, {self.vec[2]})"
        return f"Vec3D({self.vec[0]}, {self.vec[1]}, {self.vec[2]})"

    def get_return_dtype(self, fname, other: Union[GenericVec3D, int, float]) -> Type:
        self_arg = self.vec[0]
        if isinstance(other, (int, float)):
            other_arg = other
        else:
            assert isinstance(
                other, GenericVec3D
            ), "operations only supported between two GenericVec3Ds or a GenericVec3D and an int/float"
            other_arg = other.vec[0]
        return type(getattr(self_arg, fname)(other_arg))

    @overload
    def __add__(
        self: GenericVec3D[int], other: Union[GenericVec3D[int], int]
    ) -> GenericVec3D[int]:
        ...

    @overload
    def __add__(self, other: Union[GenericVec3D, float]) -> GenericVec3D:
        ...

    def __add__(self, other):
        return_dtype = self.get_return_dtype("__add__", other)
        if isinstance(other, (int, float)):
            return GenericVec3D[return_dtype](*(arg + other for i in range(3)))
        return GenericVec3D[return_dtype](*(self.vec[i] + other.vec[i] for i in range(3)))

    @overload
    def __sub__(
        self: GenericVec3D[int], other: Union[GenericVec3D[int], int]
    ) -> GenericVec3D[int]:
        ...

    @overload
    def __sub__(self, other: Union[GenericVec3D, float]) -> GenericVec3D:
        ...

    def __sub__(self, other):
        return_dtype = self.get_return_dtype("__sub__", other)
        if isinstance(other, (int, float)):
            return GenericVec3D[return_dtype](*(self.vec[i] - other for i in range(3)))
        return GenericVec3D[return_dtype](*(self.vec[i] - other.vec[i] for i in range(3)))

    @overload
    def __mul__(
        self: GenericVec3D[int], other: Union[GenericVec3D[int], int]
    ) -> GenericVec3D[int]:
        ...

    @overload
    def __mul__(self, other: Union[GenericVec3D, float]) -> GenericVec3D:
        ...

    def __mul__(self, other):
        return_dtype = self.get_return_dtype("__mul__", other)
        if isinstance(other, (int, float)):
            return GenericVec3D[return_dtype](*(self.vec[i] * other for i in range(3)))
        return GenericVec3D[return_dtype](*(self.vec[i] * other.vec[i] for i in range(3)))

    def __floordiv__(self, other: Union[GenericVec3D, int, float]) -> GenericVec3D[int]:
        return_dtype = self.get_return_dtype("__floordiv__", other)
        return GenericVec3D[int](*(self.vec[i] // other.vec[i] for i in range(3)))

    def __truediv__(self, other: GenericVec3D) -> GenericVec3D[float]:
        return_dtype = self.get_return_dtype("__truediv__", other)
        if isinstance(other, (int, float)):
            return GenericVec3D[return_dtype](*(self.vec[i] * other for i in range(3)))
        return GenericVec3D[return_dtype](*(self.vec[i] * other.vec[i] for i in range(3)))

    def to(self, dtype):
        return GenericVec3D[dtype](*(dtype(arg) for arg in self.vec))


Vec3D = GenericVec3D[float]
IntVec3D = GenericVec3D[int]


def check_type(obj: Any, cls: Any) -> bool:  # pragma: no cover
    """Type checking that works for type generics"""
    result = True
    try:
        typeguard.check_type("value", obj, cls)
    except TypeError:
        result = False
    return result
