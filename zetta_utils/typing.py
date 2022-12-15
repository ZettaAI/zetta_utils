"""Basic type definitions used for type annotations."""
from __future__ import annotations

from inspect import currentframe
from types import FrameType
from typing import Any, Generic, Literal, Tuple, Type, TypeVar, Union, overload

import typeguard

Slices3D = Tuple[slice, slice, slice]
NDType = Literal[1, 2, 3, 4, 5]
DType = float

N = TypeVar("N", bound=NDType)
T_co = TypeVar("T_co", covariant=True, bound=DType)


def get_orig_class(obj: Any) -> Type:  # pragma: no cover
    """
    Helper function to determine the original class for a generic subclass. This function
    is necessary because `obj.__orig_class__` does not work during `__init__`.
    Taken verbatim from https://github.com/Stewori/pytypes/blob/master/pytypes/type_util.py
    """
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


class _VecND(Generic[N, T_co]):
    """
    The backend primitive for an N-dimensional vector. This class should not be used or
    its constructor called directly. Use `VecND`, `IntVecND` to type annotate arbitrarily
    long N-dimensional float and int vectors respectively, and `Vec1D`, IntVec1D`, 'Vec2D`, ...
    to type annotate and instantiate an N dimensional vector of floats and ints respectively.

    `IntVecND` is a subclass of `VecND`, `IntVec1D`, `IntVec2D`, ... are subclasses of `IntVecND`
    (similar for `VecND`), and `IntVec1D`, `IntVec2D`, ... are respectively subclasses of
    the corresponding `VecND`s.
    """

    """
    Note on the following two methods: if orig_class exists, then it is guaranteed to have
    both size and type parameters - not specifying any type parameter is okay with Python
    which is why the explicit check is necessary, but specifying only one throws a TypeError.
    """

    def _get_dtype(self) -> Type:
        try:
            orig_class = get_orig_class(self)
            ret = orig_class.__args__[1]
            assert ret is int or ret is float
            return ret
        except Exception as e:
            raise TypeError(
                "_VecND must be instantiated with ndim (Literal[int]) and dtype (int or float)"
            ) from e

    def _get_ndim(self) -> int:
        return self.ndim_t.__args__[0]  # type: ignore

    def _get_ndim_type_and_set_ndim(self) -> Type:
        try:
            orig_class = get_orig_class(self)
            ret = orig_class.__args__[0]
            # isinstance doesn't work with subscripted types; this is better than nothing,
            # but doesn't keep the user from using some other Literal since the type returned
            # is just typing._LiteralGenericAlias.
            assert type(ret) == type(Literal[0])  # pylint: disable=unidiomatic-typecheck
            return ret
        except Exception as e:
            raise TypeError(
                "_VecND must be instantiated with ndim (Literal[int]) and dtype (int or float)"
            ) from e

    def __init__(self, *args: T_co):
        self.dtype: Type[DType] = self._get_dtype()
        self.ndim_t: Type[NDType] = self._get_ndim_type_and_set_ndim()
        self.ndim: int = self._get_ndim()
        try:
            assert len(args) == self.ndim
        except Exception as e:
            raise ValueError(f"Vec{self.ndim}D must have {self.ndim} elements") from e
        try:
            for arg in args:
                assert isinstance(arg, self.dtype) or (
                    self.dtype is float and isinstance(arg, int)
                )
            self.vec = tuple(self.dtype(arg) for arg in args)
        except Exception as e:
            raise TypeError(
                f"{type(arg)} argument {arg} cannot be interpreted as an {self.dtype} object"
            ) from e

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

    """
    dunder methods must be overloaded and explicitly written for type annotation reasons.
    `_get_args` is used to mirror the builtin behaviour for the arguments given for good
    practice, even though the static types have to be hardcoded.

    For typing purposes, int is a float subclass (even though `issubclass(int, float)` is False)
    which means that `foo(IntVecND, int)` matches `def foo(_VecND[N, float], float)`.
    The typing system matches from the top down, so the order of the overloads matter.
    Furthermore, type aliases `VecND` and `IntVecND` cannot be used where the return ndim needs to
    be inferred since using the alias binds the typevar `N` to the scope of the argument
    and not all the arguments.

    The `# type: ignore`s also should be necessary inside the overloaded function implementations,
    but mypy does not catch this requirement (pyright does).     """

    def _get_args(
        self, other: Union[_VecND, float, int]
    ) -> Tuple[Union[float, int], Union[float, int]]:
        self_arg = self.vec[0]
        if isinstance(other, (float, int)):
            other_arg = other
        else:
            try:
                assert len(self) == len(other)
                other_arg = other.vec[0]
            except Exception as e:
                raise NotImplementedError(
                    "operations are only supported between two vectors"
                    + " of the same length or between a vector and a float / int"
                ) from e
        return self_arg, other_arg

    def __truediv__(
        self: _VecND[N, float], other: Union[_VecND[N, float], float]
    ) -> _VecND[N, float]:
        self_arg, other_arg = self._get_args(other)
        dtype = type(self_arg / other_arg)
        if isinstance(other, (float, int)):
            return _VecND[self.ndim_t, dtype](*(e / other for e in self))  # type: ignore
        return _VecND[self.ndim_t, dtype](*(e / f for (e, f) in zip(self, other)))  # type: ignore

    def __rtruediv__(self: _VecND[N, float], other: float) -> _VecND[N, float]:
        self_arg, other_arg = self._get_args(other)
        dtype = type(other_arg / self_arg)
        return _VecND[self.ndim_t, dtype](*(other / e for e in self))  # type: ignore

    @overload
    def __add__(self: _VecND[N, int], other: Union[_VecND[N, int], int]) -> _VecND[N, int]:
        ...

    @overload
    def __add__(self: _VecND[N, float], other: Union[_VecND[N, float], float]) -> _VecND[N, float]:
        ...

    def __add__(self, other):
        self_arg, other_arg = self._get_args(other)
        dtype = type(self_arg + other_arg)
        if isinstance(other, (float, int)):
            return _VecND[self.ndim_t, dtype](*(e + other for e in self))
        return _VecND[self.ndim_t, dtype](*(e + f for (e, f) in zip(self, other)))

    @overload
    def __radd__(self: _VecND[N, int], other: int) -> _VecND[N, int]:
        ...

    @overload
    def __radd__(self: _VecND[N, float], other: float) -> _VecND[N, float]:
        ...

    def __radd__(self, other):
        self_arg, other_arg = self._get_args(other)
        dtype = type(other_arg + self_arg)
        return _VecND[self.ndim_t, dtype](*(other + e for e in self))

    @overload
    def __sub__(self: _VecND[N, int], other: Union[_VecND[N, int], int]) -> _VecND[N, int]:
        ...

    @overload
    def __sub__(self: _VecND[N, float], other: Union[_VecND[N, float], float]) -> _VecND[N, float]:
        ...

    def __sub__(self, other):
        self_arg, other_arg = self._get_args(other)
        dtype = type(self_arg - other_arg)
        if isinstance(other, (float, int)):
            return _VecND[self.ndim_t, dtype](*(e - other for e in self))
        return _VecND[self.ndim_t, dtype](*(e - f for (e, f) in zip(self, other)))

    @overload
    def __rsub__(self: _VecND[N, int], other: int) -> _VecND[N, int]:
        ...

    @overload
    def __rsub__(self: _VecND[N, float], other: float) -> _VecND[N, float]:
        ...

    def __rsub__(self, other):
        self_arg, other_arg = self._get_args(other)
        dtype = type(other_arg - self_arg)
        return _VecND[self.ndim_t, dtype](*(other - e for e in self))

    @overload
    def __mul__(self: _VecND[N, int], other: Union[_VecND[N, int], int]) -> _VecND[N, int]:
        ...

    @overload
    def __mul__(self: _VecND[N, float], other: Union[_VecND[N, float], float]) -> _VecND[N, float]:
        ...

    def __mul__(self, other):
        self_arg, other_arg = self._get_args(other)
        dtype = type(self_arg * other_arg)
        if isinstance(other, (float, int)):
            return _VecND[self.ndim_t, dtype](*(e * other for e in self))
        return _VecND[self.ndim_t, dtype](*(e * f for (e, f) in zip(self, other)))

    @overload
    def __rmul__(self: _VecND[N, int], other: int) -> _VecND[N, int]:
        ...

    @overload
    def __rmul__(self: _VecND[N, float], other: float) -> _VecND[N, float]:
        ...

    def __rmul__(self, other):
        self_arg, other_arg = self._get_args(other)
        dtype = type(other_arg * self_arg)
        return _VecND[self.ndim_t, dtype](*(other * e for e in self))

    @overload
    def __floordiv__(self: _VecND[N, int], other: Union[_VecND[N, int], int]) -> _VecND[N, int]:
        ...

    @overload
    def __floordiv__(
        self: _VecND[N, float], other: Union[_VecND[N, float], float]
    ) -> _VecND[N, float]:
        ...

    def __floordiv__(self, other):
        self_arg, other_arg = self._get_args(other)
        dtype = type(self_arg // other_arg)
        if isinstance(other, (float, int)):
            return _VecND[self.ndim_t, dtype](*(e // other for e in self))
        return _VecND[self.ndim_t, dtype](*(e // f for (e, f) in zip(self, other)))

    @overload
    def __rfloordiv__(self: _VecND[N, int], other: int) -> _VecND[N, int]:
        ...

    @overload
    def __rfloordiv__(self: _VecND[N, float], other: float) -> _VecND[N, float]:
        ...

    def __rfloordiv__(self, other):
        self_arg, other_arg = self._get_args(other)
        dtype = type(other_arg // self_arg)
        return _VecND[self.ndim_t, dtype](*(other // e for e in self))

    def to(self, dtype):  # pylint: disable=invalid-name #pragma: no-cover
        raise NotImplementedError(".to(dtype) is not implemented; use .int() or .float() instead")

    def int(self: _VecND[N, float]) -> _VecND[N, int]:  # pragma: no-cover
        return _VecND[self.ndim_t, int](*(int(arg) for arg in self.vec))  # type: ignore

    def float(self: _VecND[N, float]) -> _VecND[N, float]:  # pragma: no-cover
        return _VecND[self.ndim_t, float](*(float(arg) for arg in self.vec))  # type: ignore


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
    """Type checking that works better for type generics"""
    result = True
    try:
        typeguard.check_type("value", obj, cls)
    except TypeError:
        result = False
    return result
