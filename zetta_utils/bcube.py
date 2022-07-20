# pylint: disable=missing-docstring, no-else-raise
from __future__ import annotations
import copy

from math import floor

from typing import Tuple, Union, Optional
from typeguard import typechecked


VolumetricSlices = Tuple[slice, slice, slice]
VolumetricCoordInt = Tuple[int, int, int]
VolumetricCoordStr = str
VolumetricCoord = Union[VolumetricCoordStr, VolumetricCoordInt]
VolumetricResolution = Tuple[int, int, int]


@typechecked
class BoundingCube:
    """Represents a 3D cuboid in space."""

    def __init__(
        self,
        slices: Optional[VolumetricSlices] = None,
        start_coord: Optional[VolumetricCoord] = None,
        end_coord: Optional[VolumetricCoord] = None,
        unit_name: str = "nm",
        resolution: VolumetricResolution = (1, 1, 1),
    ):
        self.x_range = (0, 0)
        self.y_range = (0, 0)
        self.z_range = (0, 0)

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

    def set_to_slices(self, slices: VolumetricSlices, resolution: VolumetricResolution):
        assert len(slices) == 3
        for s in slices:
            assert s.step is None

        self.x_range = (
            slices[0].start * resolution[0],
            slices[0].stop * resolution[0],
        )
        self.y_range = (
            slices[1].start * resolution[1],
            slices[1].stop * resolution[1],
        )
        self.z_range = (
            slices[2].start * resolution[2],
            slices[2].stop * resolution[2],
        )

    def set_to_coords(
        self,
        start_coord: VolumetricCoord,
        end_coord: VolumetricCoord,
        resolution: VolumetricResolution,
    ):
        if isinstance(start_coord, str):
            start_coord_int = tuple(int(i) for i in start_coord.split(","))
        else:
            start_coord_int = start_coord

        if isinstance(end_coord, str):
            end_coord_int = tuple(int(i) for i in end_coord.split(","))
        else:
            end_coord_int = end_coord

        if len(start_coord_int) != 3:
            raise ValueError(f"Invalid `start_coord`: {start_coord_int}")

        if len(end_coord_int) != 3:
            raise ValueError(f"Invalid `end_coord`: {end_coord_int}")

        self.x_range = (
            start_coord_int[0] * resolution[0],
            end_coord_int[0] * resolution[0],
        )
        self.y_range = (
            start_coord_int[1] * resolution[1],
            end_coord_int[1] * resolution[1],
        )
        self.z_range = (
            start_coord_int[2] * resolution[2],
            end_coord_int[2] * resolution[2],
        )

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, BoundingCube)
            and (self.z_range == other.z_range)
            and (self.y_range == other.y_range)
            and (self.x_range == other.x_range)
            and (self.unit_name == other.unit_name)
        )

    def get_x_range(self, x_res: int = 1) -> Tuple[int, int]:
        # scale_factor = 2 ** mip
        # xl = int(round((self.m0_x[1] - self.m0_x[0]) / scale_factor))
        # xs = floor(self.m0_x[0] / scale_factor)
        # return [xs, xs + xl]

        return (
            floor(self.x_range[0] / x_res),
            int(round(self.x_range[1] / x_res)),
        )

    def get_y_range(self, y_res: int = 1) -> Tuple[int, int]:
        return (
            floor(self.y_range[0] / y_res),
            int(round(self.y_range[1] / y_res)),
        )

    def get_z_range(self, z_res: int = 1) -> Tuple[int, int]:
        return (
            floor(self.z_range[0] / z_res),
            int(round(self.z_range[1] / z_res)),
        )

    def pad(
        self,
        x_pad: int = 0,
        y_pad: int = 0,
        z_pad: int = 0,
        in_place: bool = False,
        resolution: VolumetricResolution = (1, 1, 1),
    ) -> BoundingCube:
        """Pads the bounding box. Pad values are measured by the given resolution.
        if not provided, resolution defaults to (1, 1, 1)"""
        if in_place:
            raise NotImplementedError  # pragma: no cover
        else:
            result = BoundingCube(
                start_coord=(
                    self.x_range[0] - x_pad * resolution[0],
                    self.y_range[0] - y_pad * resolution[1],
                    self.z_range[0] - z_pad * resolution[2],
                ),
                end_coord=(
                    self.x_range[1] + x_pad * resolution[0],
                    self.y_range[1] + y_pad * resolution[1],
                    self.z_range[1] + z_pad * resolution[2],
                ),
                resolution=(1, 1, 1),
                unit_name=self.unit_name,
            )
        return result

    def translate(
        self,
        offset: Tuple[int, int, int],
        resolution: VolumetricResolution = (1, 1, 1),
        in_place: bool = False,
    ) -> BoundingCube:
        if in_place:
            raise NotImplementedError  # pragma: no cover
        else:
            result = BoundingCube(
                start_coord=(
                    self.x_range[0] - offset[0] * resolution[0],
                    self.y_range[0] - offset[1] * resolution[1],
                    self.z_range[0] - offset[2] * resolution[2],
                ),
                end_coord=(
                    self.x_range[1] + offset[0] * resolution[0],
                    self.y_range[1] + offset[1] * resolution[1],
                    self.z_range[1] + offset[2] * resolution[2],
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
            f"z: {self.z_range[0]}-{self.z_range[1]}, "
            f"y: {self.y_range[0]}-{self.y_range[1]}{self.unit_name}, "
            f"x: {self.x_range[0]}-{self.x_range[1]}{self.unit_name})"
        )
