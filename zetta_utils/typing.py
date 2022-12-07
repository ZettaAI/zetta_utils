"""Basic type definitions used for type annotations."""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import typeguard

Slices3D = Tuple[slice, slice, slice]


class Vec3D(np.ndarray):  # pragma: no cover
    def __new__(cls, vec: Tuple[float, float, float]):
        # check necessary for runtime
        assert len(vec) == 3, "Vec3D must have 3 elements"
        return np.asarray(vec, dtype=np.float64).view(cls)

    # avoid returning Vec3D whenever sliced, defaulting to float or tuple
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.view(np.ndarray).__getitem__(key)
        return tuple(self.view(np.ndarray).__getitem__(key))

    def __eq__(self, other):
        return np.array_equal(self, other)


class IntVec3D(np.ndarray):  # pragma: no cover
    def __new__(cls, vec: Tuple[int, int, int]):
        # check necessary for runtime
        assert len(vec) == 3, "IntVec3D must have 3 elements"
        return np.asarray(vec, dtype=np.int64).view(cls)

    # avoid returning IntVec3D whenever sliced, defaulting to int or tuple
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.view(np.ndarray).__getitem__(key)
        return tuple(self.view(np.ndarray).__getitem__(key))

    def __eq__(self, other):
        return np.array_equal(self, other)


def check_type(obj: Any, cls: Any) -> bool:  # pragma: no cover
    """Type checking that works for type generics"""
    result = True
    try:
        typeguard.check_type("value", obj, cls)
    except TypeError:
        result = False
    return result
