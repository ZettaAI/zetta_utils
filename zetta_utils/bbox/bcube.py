# pylint: disable=missing-docstring, no-else-raise
from __future__ import annotations

from typing import Union, Sequence, Tuple, cast, Literal

import attrs
from typeguard import typechecked

from zetta_utils.typing import Number
from zetta_utils.bbox import BoundingBoxND, DEFAULT_UNIT

@typechecked
@attrs.frozen()
class BoundingCube(BoundingBoxND):
    expected_ndim: Literal[3] = 3

    def to_slices(
        self, resolution: Sequence[Number], allow_rounding: bool = False
    ) -> Tuple[slice, slice, slice]:
        ndim_result = super().to_slices(resolution, allow_rounding)
        result = cast(Tuple[slice, slice, slice], ndim_result)
        return result

    @classmethod
    def from_slices(
        cls,
        slices: Sequence[slice],
        resolution: Sequence[Number],
        unit: str = DEFAULT_UNIT,
    ) -> BoundingCube:
        ndim_result = super().from_slices(slices, resolution, unit)
        result = BoundingCube(
            bounds=ndim_result.bounds,
            unit=ndim_result.unit
        )
        return result


    @classmethod
    def from_coords(
        cls,
        start_coord: Sequence[Number],
        end_coord: Sequence[Number],
        resolution: Sequence[Number],
        unit: str = DEFAULT_UNIT,
    ) -> BoundingCube:
        ndim_result = super().from_coords(start_coord, end_coord, resolution, unit)
        result = BoundingCube(
            bounds=ndim_result.bounds,
            unit=ndim_result.unit
        )
        return result

    def pad(
        self,
        pad: Sequence[Union[int, float, tuple[Union[int, float], Union[int, float]]]],
        resolution: Sequence[Number],
        in_place: bool = False,
    ) -> BoundingCube:
        ndim_result = super().pad(pad, resolution, in_place)
        result = BoundingCube(
            bounds=ndim_result.bounds,
            unit=ndim_result.unit
        )
        return result


    def translate(
        self,
        offset: Sequence[Number],
        resolution: Sequence[Number],
        in_place: bool = False,
    ) -> BoundingCube:
        ndim_result = super().translate(offset, resolution, in_place)
        result = BoundingCube(
            bounds=ndim_result.bounds,
            unit=ndim_result.unit
        )
        return result
