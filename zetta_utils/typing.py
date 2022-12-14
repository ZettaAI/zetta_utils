"""Basic type definitions used for type annotations."""
from __future__ import annotations

from inspect import currentframe
from types import FrameType
from typing import Any, Generic, Literal, Tuple, Type, TypeVar, Union, overload

import typeguard

Slices3D = Tuple[slice, slice, slice]
NDType = Literal[1, 2, 3, 4, 5]


N = TypeVar("N", bound=NDType)
T_co = TypeVar("T_co", covariant=True, bound=float)

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
        assert isinstance(frame, FrameType)
        frame = frame.f_back
        assert isinstance(frame, FrameType)
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
        return type(None)


# dunder methods must be overloaded and explicitly written for type annotation reasons


class _VecND(Generic[N, T_co]):  # pragma: no cover

    # Note on the following two methods: if orig_class exists, then it is guaranteed to have
    # both size and type parameters - not specifying any type parameter is okay with Python
    # which is why the explicit check is necessary, but specifying only one throws a TypeError.
    def _get_dtype(self) -> Type:
        try:
            orig_class = get_orig_class(self)
            ret = orig_class.__args__[1]
            assert ret is int or ret is float
            return ret
        except:
            raise TypeError(
                "_VecND must be instantiated with ndim (Literal[int]) and dtype (int or float)"
            ) from None

    def _get_ndim(self) -> int:
        # doesn't need to be checked
        return self._get_ndim_type().__args__[0]

    def _get_ndim_type(self) -> Type:
        try:
            orig_class = get_orig_class(self)
            ret = orig_class.__args__[0]
            # isinstance doesn't work with subscripted types; this is better than nothing,
            # but doesn't keep the user from using some other Literal since the type returned
            # is just typing._LiteralGenericAlias.
            assert type(ret) == type(Literal[0])  # pylint: disable=unidiomatic-typecheck
            return ret
        except:
            raise TypeError(
                "_VecND must be instantiated with ndim (Literal[int]) and dtype (int or float)"
            ) from None

    def __init__(self, *args: T_co):
        self.dtype: Type[N] = self._get_dtype()
        self.ndim_t: Type[N] = self._get_ndim_type()
        self.ndim: int = self._get_ndim()
        try:
            assert len(args) == self.ndim
        except:
            raise ValueError(f"Vec{self.ndim}D must have {self.ndim} elements") from None
        try:
            for arg in args:
                assert self.dtype(arg) == arg
            self.vec = tuple(self.dtype(arg) for arg in args)
        except:
            raise ValueError(f"cannot cast {arg} to {self.dtype} implicitly") from None

    # avoid returning Vec3D whenever sliced, defaulting to float or tuple
    @overload
    def __getitem__(self, key: int) -> T_co:
        ...

    @overload
    def __getitem__(self, key: slice) -> Tuple[T_co, ...]:
        ...

    def __getitem__(self, key):
        return self.vec[key]

    def __iter__(self):
        return self.vec.__iter__()

    def __len__(self) -> int:
        return self.ndim

    def __eq__(self, other) -> bool:
        return self.vec == other.vec

    def __repr__(self) -> str:
        if self.dtype is int:
            return f"IntVec{self.ndim}D({', '.join(str(e) for e in self)})"
        if self.dtype is float:
            return f"Vec{self.ndim}D({', '.join(str(e) for e in self)})"
        return f"_VecND({', '.join(str(e) for e in self)})"

    def get_return_dtype(self, fname, other: Union[_VecND, float, int]) -> Type:
        self_arg = self.vec[0]
        if isinstance(other, (float, int)):
            other_arg = other
        else:
            try:
                assert len(self) == len(other)
                other_arg = other.vec[0]
            except:
                raise TypeError(
                    "operation only supported between two vectors"
                    + " of the same length or between a vector and a float / int"
                ) from None
        return type(getattr(self_arg, fname)(other_arg))

    def __truediv__(
        self: _VecND[N, float], other: Union[_VecND[N, float], float, int]
    ) -> _VecND[N, float]:
        if isinstance(other, (float, int)):
            return _VecND[self.ndim_t, float](*(e / other for e in self))  # type: ignore
        return _VecND[self.ndim_t, float](*(e / f for (e, f) in zip(self, other)))  # type: ignore

    @overload
    def __add__(self: _VecND[N, int], other: Union[_VecND[N, int], int]) -> _VecND[N, int]:
        ...

    @overload
    def __add__(self: _VecND[N, int], other: Union[_VecND[N, float], float]) -> _VecND[N, float]:
        ...

    @overload
    def __add__(
        self: _VecND[N, float], other: Union[_VecND[N, float], _VecND[N, int], float, int]
    ) -> _VecND[N, float]:
        ...

    def __add__(self, other):
        dtype = self.get_return_dtype("__add__", other)
        if isinstance(other, (float, int)):
            return _VecND[self.ndim_t, dtype](*(e + other for e in self))  # type: ignore
        return _VecND[self.ndim_t, dtype](*(e + f for (e, f) in zip(self, other)))  # type: ignore

    @overload
    def __sub__(self: _VecND[N, int], other: Union[_VecND[N, int], int]) -> _VecND[N, int]:
        ...

    @overload
    def __sub__(self: _VecND[N, int], other: Union[_VecND[N, float], float]) -> _VecND[N, float]:
        ...

    @overload
    def __sub__(
        self: _VecND[N, float], other: Union[_VecND[N, float], _VecND[N, int], float, int]
    ) -> _VecND[N, float]:
        ...

    def __sub__(self, other):
        dtype = self.get_return_dtype("__sub__", other)
        if isinstance(other, (float, int)):
            return _VecND[self.ndim_t, dtype](*(e - other for e in self))  # type: ignore
        return _VecND[self.ndim_t, dtype](*(e - f for (e, f) in zip(self, other)))  # type: ignore

    @overload
    def __mul__(self: _VecND[N, int], other: Union[_VecND[N, int], int]) -> _VecND[N, int]:
        ...

    @overload
    def __mul__(self: _VecND[N, int], other: Union[_VecND[N, float], float]) -> _VecND[N, float]:
        ...

    @overload
    def __mul__(
        self: _VecND[N, float], other: Union[_VecND[N, float], _VecND[N, int], float, int]
    ) -> _VecND[N, float]:
        ...

    def __mul__(self, other):
        dtype = self.get_return_dtype("__mul__", other)
        if isinstance(other, (float, int)):
            return _VecND[self.ndim_t, dtype](*(e * other for e in self))  # type: ignore
        return _VecND[self.ndim_t, dtype](*(e * f for (e, f) in zip(self, other)))  # type: ignore

    @overload
    def __floordiv__(self: _VecND[N, int], other: Union[_VecND[N, int], int]) -> _VecND[N, int]:
        ...

    @overload
    def __floordiv__(
        self: _VecND[N, int], other: Union[_VecND[N, float], float]
    ) -> _VecND[N, float]:
        ...

    @overload
    def __floordiv__(
        self: _VecND[N, float], other: Union[_VecND[N, float], _VecND[N, int], float, int]
    ) -> _VecND[N, float]:
        ...

    def __floordiv__(self, other):
        dtype = self.get_return_dtype("__floordiv__", other)
        if isinstance(other, (float, int)):
            return _VecND[self.ndim_t, dtype](*(e // other for e in self))  # type: ignore
        return _VecND[self.ndim_t, dtype](*(e // f for (e, f) in zip(self, other)))  # type: ignore

    def to(self, dtype):  # pylint: disable=invalid-name
        try:
            assert dtype is int or dtype is float
            return _VecND[self.ndim_t, dtype](*(dtype(arg) for arg in self.vec))  # type: ignore
        except:
            raise TypeError("dtype must be float or int") from None


# Type Aliases
VecND = _VecND[N, float]
Vec1D = _VecND[Literal[1], float]
Vec2D = _VecND[Literal[2], float]
Vec3D = _VecND[Literal[3], float]
Vec4D = _VecND[Literal[4], float]
Vec5D = _VecND[Literal[5], float]
IntVecND = _VecND[N, int]
IntVec1D = _VecND[Literal[1], int]
IntVec2D = _VecND[Literal[2], int]
IntVec3D = _VecND[Literal[3], int]
IntVec4D = _VecND[Literal[4], int]
IntVec5D = _VecND[Literal[5], int]


def check_type(obj: Any, cls: Any) -> bool:  # pragma: no cover
    """Type checking that works for type generics"""
    result = True
    try:
        typeguard.check_type("value", obj, cls)
    except TypeError:
        result = False
    return result
