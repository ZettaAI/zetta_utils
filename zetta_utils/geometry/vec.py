"""Basic type definitions used for type annotations."""
from __future__ import annotations

from collections import abc
from typing import Any, Sequence, Tuple, TypeVar, Union, overload

import attrs
import numpy as np
from typeguard import typechecked
from typing_extensions import TypeGuard

from zetta_utils import builder

BuiltinInt = int
BuiltinFloat = float

T = TypeVar("T", bound=float)


@typechecked
@attrs.mutable(init=False)
class Vec3D(abc.Sequence[T]):
    """
    Primitive for an 3-dimensional vector.  Code for other dimensioinalities will be autogenerated.
    """

    x: T
    y: T
    z: T

    def __init__(self, x: T | np.generic, y: T | np.generic, z: T | np.generic):
        if isinstance(x, np.generic):
            self.x = x.item()
        else:
            self.x = x

        if isinstance(y, np.generic):
            self.y = y.item()
        else:
            self.y = y

        if isinstance(z, np.generic):
            self.z = z.item()
        else:
            self.z = z

    @property
    def vec(self) -> tuple[T, T, T]:
        return (self.x, self.y, self.z)

    @overload
    def __getitem__(self, key: BuiltinInt) -> T:
        ...

    @overload
    def __getitem__(self, key: slice) -> Tuple[T, ...]:
        ...

    def __getitem__(self, key):
        return self.vec[key]

    @overload
    def __setitem__(self, key: BuiltinInt, val: T) -> None:
        ...

    @overload
    def __setitem__(self, key: slice, val: Sequence[T]) -> None:
        ...

    def __setitem__(self, key, val):
        newvec = list(self.vec)
        newvec.__setitem__(key, val)
        self.__init__(*newvec)  # pylint: disable=no-value-for-parameter

    def __iter__(self):
        return self.vec.__iter__()

    def __len__(self) -> BuiltinInt:
        return 3

    def __eq__(self, other) -> bool:
        if not hasattr(other, "vec"):
            return False
        return self.vec == other.vec

    def __lt__(self, other) -> bool:
        return all(s < o for (s, o) in zip(self.vec, other.vec))

    def __le__(self, other) -> bool:
        return all(s <= o for (s, o) in zip(self.vec, other.vec))

    def __gt__(self, other) -> bool:
        return all(s > o for (s, o) in zip(self.vec, other.vec))

    def __ge__(self, other) -> bool:
        return all(s >= o for (s, o) in zip(self.vec, other.vec))

    def __repr__(self) -> str:
        return f"Vec3D({', '.join(str(e) for e in self)})"

    def __truediv__(self, other: Vec3D | BuiltinFloat) -> Vec3D[BuiltinFloat]:
        if isinstance(other, (BuiltinFloat, BuiltinInt)):
            return Vec3D[BuiltinFloat](*(e / other for e in self))
        else:
            return Vec3D[BuiltinFloat](*(e / f for (e, f) in zip(self, other)))

    def __rtruediv__(self, other: BuiltinFloat) -> Vec3D[BuiltinFloat]:
        return Vec3D[BuiltinFloat](*(other / e for e in self))

    def __neg__(self) -> Vec3D[T]:
        return Vec3D[T](*(-e for e in self))

    @overload
    def __add__(self, other: Union[Vec3D[BuiltinInt], BuiltinInt]) -> Vec3D[T]:
        ...

    @overload
    def __add__(self, other: Union[Vec3D[BuiltinFloat], BuiltinFloat]) -> Vec3D[BuiltinFloat]:
        ...

    def __add__(self, other: Vec3D | BuiltinInt | BuiltinFloat):
        if isinstance(other, BuiltinInt):
            return Vec3D[T](*(e + other for e in self))
        elif isinstance(other, BuiltinFloat):
            return Vec3D[BuiltinFloat](*(e + other for e in self))
        elif is_int_vec(other):
            return Vec3D[T](*(e + f for (e, f) in zip(self, other)))  # type: ignore
        else:
            return Vec3D[T](*(e + f for (e, f) in zip(self, other)))

    @overload
    def __radd__(self, other: BuiltinInt) -> Vec3D[T]:
        ...

    @overload
    def __radd__(self, other: BuiltinFloat) -> Vec3D[BuiltinFloat]:
        ...

    def __radd__(self, other):
        if isinstance(other, BuiltinInt):
            return Vec3D[T](*(other + e for e in self))
        else:
            return Vec3D[BuiltinFloat](*(other + e for e in self))

    @overload
    def __sub__(self, other: Union[Vec3D[BuiltinInt], BuiltinInt]) -> Vec3D[T]:
        ...

    @overload
    def __sub__(self, other: Union[Vec3D[BuiltinFloat], BuiltinFloat]) -> Vec3D[BuiltinFloat]:
        ...

    def __sub__(self, other):
        if isinstance(other, (BuiltinInt)):
            return Vec3D[T](*(e - other for e in self))
        elif isinstance(other, (BuiltinFloat)):
            return Vec3D[BuiltinFloat](*(e - other for e in self))
        elif is_int_vec(other):
            return Vec3D[T](*(e - f for (e, f) in zip(self, other)))
        else:
            return Vec3D[BuiltinFloat](*(e - f for (e, f) in zip(self, other)))

    @overload
    def __rsub__(self, other: BuiltinInt) -> Vec3D[T]:
        ...

    @overload
    def __rsub__(self, other: BuiltinFloat) -> Vec3D[BuiltinFloat]:
        ...

    def __rsub__(self, other):
        if isinstance(other, BuiltinInt):
            return Vec3D[T](*(other - e for e in self))
        else:
            return Vec3D[BuiltinFloat](*(other - e for e in self))

    @overload
    def __mul__(self, other: Union[Vec3D[BuiltinInt], BuiltinInt]) -> Vec3D[T]:
        ...

    @overload
    def __mul__(self, other: Union[Vec3D[BuiltinFloat], BuiltinFloat]) -> Vec3D[BuiltinFloat]:
        ...

    def __mul__(self, other):
        if isinstance(other, (BuiltinInt)):
            return Vec3D[T](*(e * other for e in self))
        elif isinstance(other, (BuiltinFloat)):
            return Vec3D[BuiltinFloat](*(e * other for e in self))
        elif is_int_vec(other):
            return Vec3D[T](*(e * f for (e, f) in zip(self, other)))
        else:
            return Vec3D[BuiltinFloat](*(e * f for (e, f) in zip(self, other)))

    @overload
    def __rmul__(self, other: BuiltinInt) -> Vec3D[T]:
        ...

    @overload
    def __rmul__(self, other: BuiltinFloat) -> Vec3D[BuiltinFloat]:
        ...

    def __rmul__(self, other):
        if isinstance(other, BuiltinInt):
            return Vec3D[T](*(other * e for e in self))
        else:
            return Vec3D[BuiltinFloat](*(other * e for e in self))

    @overload
    def __floordiv__(
        self: Vec3D[BuiltinInt], other: Union[Vec3D[BuiltinInt], BuiltinInt]
    ) -> Vec3D[BuiltinInt]:
        ...

    @overload
    def __floordiv__(self, other: Union[Vec3D[BuiltinFloat], BuiltinFloat]) -> Vec3D[BuiltinFloat]:
        ...

    @overload
    def __floordiv__(
        self: Vec3D[BuiltinFloat], other: Union[Vec3D[BuiltinInt], BuiltinInt]
    ) -> Vec3D[BuiltinFloat]:
        ...

    def __floordiv__(self, other):
        if isinstance(other, (BuiltinFloat, BuiltinInt)):
            return Vec3D[BuiltinInt](*(e // other for e in self))
        return Vec3D[BuiltinInt](*(e // f for (e, f) in zip(self, other)))

    @overload
    def __rfloordiv__(self, other: BuiltinInt) -> Vec3D[T]:
        ...

    @overload
    def __rfloordiv__(self, other: BuiltinFloat) -> Vec3D[BuiltinFloat]:
        ...

    def __rfloordiv__(self, other):
        return Vec3D[BuiltinInt](*(other // e for e in self))

    @overload
    def __mod__(self, other: Union[Vec3D[BuiltinInt], BuiltinInt]) -> Vec3D[T]:
        ...

    @overload
    def __mod__(self, other: Union[Vec3D[BuiltinFloat], BuiltinFloat]) -> Vec3D[BuiltinFloat]:
        ...

    def __mod__(self, other):
        if isinstance(other, (BuiltinInt)):
            return Vec3D[T](*(e % other for e in self))
        elif isinstance(other, (BuiltinFloat)):
            return Vec3D[BuiltinFloat](*(e % other for e in self))
        elif is_int_vec(other):
            return Vec3D[T](*(e % f for (e, f) in zip(self, other)))
        else:
            return Vec3D[BuiltinFloat](*(e % f for (e, f) in zip(self, other)))

    @overload
    def __rmod__(self, other: BuiltinInt) -> Vec3D[T]:
        ...

    @overload
    def __rmod__(self, other: BuiltinFloat) -> Vec3D[BuiltinFloat]:
        ...

    def __rmod__(self, other):
        if isinstance(other, BuiltinInt):
            return Vec3D[T](*(other % e for e in self))
        else:
            return Vec3D[BuiltinFloat](*(other % e for e in self))

    def int(self) -> Vec3D[BuiltinInt]:  # pragma: no-cover
        return Vec3D[BuiltinInt](*(BuiltinInt(arg) for arg in self.vec))

    def float(self) -> Vec3D[BuiltinFloat]:  # pragma: no-cover
        return Vec3D[BuiltinFloat](*(BuiltinFloat(arg) for arg in self.vec))


@typechecked
def is_int_vec(vec: Vec3D) -> TypeGuard[Vec3D[int]]:
    return all(isinstance(v, int) for v in vec.vec)


def convert_list3_to_vec3d(value: Any) -> Any:
    if (
        isinstance(value, Sequence)
        and len(value) == 3
        and all(isinstance(e, (int, float)) for e in value)
    ):
        return Vec3D(*value)
    else:
        return value


builder.AUTOCONVERTERS.append(convert_list3_to_vec3d)
IntVec3D = Vec3D[int]
