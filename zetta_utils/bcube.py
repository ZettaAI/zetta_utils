# pylint: disable=missing-docstring, no-else-raise
from __future__ import annotations
import copy

from math import floor

from typing import Optional, Union
from typeguard import typechecked

from zetta_utils.typing import (
    Coord3D,
    Slice3D,
    Vec3D,
    Dim3D,
    DimIdx3D,
    Padding3D,
    Number,
)


def get_dim_idx(dim: Dim3D) -> DimIdx3D:  # pragma: no cover
    if dim == "x":
        result = 0  # type: DimIdx3D
    elif dim == "y":
        result = 1
    elif dim == "z":
        result = 2
    else:
        result = dim

    return result


@typechecked
class BoundingCube:
    """Represents a 3D cuboid in space."""

    def __init__(
        self,
        slices: Optional[Slice3D] = None,
        start_coord: Optional[Coord3D] = None,
        end_coord: Optional[Coord3D] = None,
        unit_name: str = "nm",
        resolution: Vec3D = (1, 1, 1),
    ):
        self.ranges = (
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        )

        self.unit_name = unit_name

        if slices is not None:
            if start_coord is not None or end_coord is not None:
                raise ValueError(
                    "Both `slices` and `start_coord/end_coord` provided to "
                    "bounding cube constructor."
                )
            self.set_to_slices(slices, resolution)

        elif start_coord is not None or end_coord is not None:
            if start_coord is None:
                raise ValueError(
                    "`end_coord` provided to bounding cube constructor, "
                    "but `start_coord` is `None`."
                )
            if end_coord is None:
                raise ValueError(
                    "`start_coord` provided to bounding cube constructor, "
                    "but `end_coord` is `None`."
                )
            self.set_to_coords(start_coord, end_coord, resolution)

    def set_to_slices(self, slices: Slice3D, resolution: Vec3D):
        for s in slices:
            assert s.step is None

        self.ranges = (
            (slices[0].start * resolution[0], slices[0].stop * resolution[0]),
            (slices[1].start * resolution[1], slices[1].stop * resolution[1]),
            (slices[2].start * resolution[2], slices[2].stop * resolution[2]),
        )

    def set_to_coords(
        self,
        start_coord: Coord3D,
        end_coord: Coord3D,
        resolution: Vec3D,
    ):
        if isinstance(start_coord, str):
            start_coord_ = tuple(float(i) for i in start_coord.split(","))
        else:
            start_coord_ = start_coord

        if isinstance(end_coord, str):
            end_coord_ = tuple(float(i) for i in end_coord.split(","))
        else:
            end_coord_ = end_coord

        if len(start_coord_) != 3:
            raise ValueError(f"Invalid `start_coord`: {start_coord_}")

        if len(end_coord_) != 3:
            raise ValueError(f"Invalid `end_coord`: {end_coord_}")

        self.ranges = (
            (start_coord_[0] * resolution[0], end_coord_[0] * resolution[0]),
            (start_coord_[1] * resolution[1], end_coord_[1] * resolution[1]),
            (start_coord_[2] * resolution[2], end_coord_[2] * resolution[2]),
        )

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, BoundingCube)
            and (self.ranges == other.ranges)
            and (self.unit_name == other.unit_name)
        )

    def get_slice(
        self,
        dim: Dim3D,
        resolution: Union[Vec3D, Number],
        allow_rounding: bool = False,
    ) -> slice:
        dim_idx = get_dim_idx(dim)

        if isinstance(resolution, (int, float)):
            dim_res = resolution
        else:
            dim_res = resolution[dim_idx]

        dim_range_start_raw = self.ranges[dim_idx][0] / dim_res
        dim_range_end_raw = self.ranges[dim_idx][1] / dim_res

        if not allow_rounding:
            if dim_range_start_raw != round(dim_range_start_raw):
                raise ValueError(
                    f"{self} results in slice_start == "
                    f"{dim_range_start_raw} along dimension {dim_idx} "
                    f"at resolution == {resolution} while "
                    "`allow_rounding` == False."
                )
            if dim_range_end_raw != round(dim_range_end_raw):
                raise ValueError(
                    f"{self} results in slice_end == "
                    f"{dim_range_end_raw} along dimension {dim_idx} "
                    f"at resolution == {resolution} while "
                    "`allow_rounding` == False."
                )
        slice_length = int(round(dim_range_end_raw - dim_range_start_raw))
        result = slice(
            floor(dim_range_start_raw),
            floor(dim_range_start_raw) + slice_length,
        )
        return result

    def get_slices(self, resolution: Vec3D, allow_rounding: bool = False) -> Slice3D:
        result = (
            self.get_slice(0, resolution[0], allow_rounding),
            self.get_slice(1, resolution[1], allow_rounding),
            self.get_slice(2, resolution[2], allow_rounding),
        )
        return result

    def pad(
        self,
        pad: Padding3D,
        in_place: bool = False,
        resolution: Vec3D = (1, 1, 1),
    ) -> BoundingCube:
        if in_place:
            raise NotImplementedError  # pragma: no cover
        else:
            double_sided_pad = []
            for i in pad:
                if isinstance(i, (int, float)):
                    double_sided_pad += [(i, i)]
                else:
                    double_sided_pad += [i]  # type: ignore

            result = BoundingCube(
                start_coord=(
                    self.ranges[0][0] - double_sided_pad[0][0] * resolution[0],
                    self.ranges[1][0] - double_sided_pad[1][0] * resolution[1],
                    self.ranges[2][0] - double_sided_pad[2][0] * resolution[2],
                ),
                end_coord=(
                    self.ranges[0][1] + double_sided_pad[0][1] * resolution[0],
                    self.ranges[1][1] + double_sided_pad[1][1] * resolution[1],
                    self.ranges[2][1] + double_sided_pad[2][1] * resolution[2],
                ),
                resolution=(1, 1, 1),
                unit_name=self.unit_name,
            )
        return result

    def translate(
        self,
        offset: Vec3D,
        resolution: Vec3D = (1, 1, 1),
        in_place: bool = False,
    ) -> BoundingCube:
        if in_place:
            raise NotImplementedError  # pragma: no cover
        else:
            result = BoundingCube(
                start_coord=(
                    self.ranges[0][0] + offset[0] * resolution[0],
                    self.ranges[1][0] + offset[1] * resolution[1],
                    self.ranges[2][0] + offset[2] * resolution[2],
                ),
                end_coord=(
                    self.ranges[0][1] + offset[0] * resolution[0],
                    self.ranges[1][1] + offset[1] * resolution[1],
                    self.ranges[2][1] + offset[2] * resolution[2],
                ),
                resolution=(1, 1, 1),
                unit_name=self.unit_name,
            )

        return result

    def clone(self) -> BoundingCube:  # pragma: no cover
        return copy.deepcopy(self)

    def copy(self) -> BoundingCube:  # pragma: no cover
        return self.clone()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            "BoundingCube("
            f"x: {self.ranges[0][0]}-{self.ranges[0][1]}{self.unit_name}, "
            f"y: {self.ranges[1][0]}-{self.ranges[1][1]}{self.unit_name}, "
            f"z: {self.ranges[2][0]}-{self.ranges[2][1]}{self.unit_name})"
        )
