# pylint: disable=missing-docstring
from __future__ import annotations
import copy

from math import floor

from typing import List, Union, Optional
from typeguard import typechecked


@typechecked
class BoundingCube:
    """Represents a 3D cuboid in space."""

    def __init__(
        self,
        slices: Optional[list] = None,
        start_coord: Optional[Union[str, List[int]]] = None,
        end_coord: Optional[Union[str, List[int]]] = None,
        unit_name: str = "nm",
        resolution: Optional[List[int]] = None,
    ):
        if resolution is None:
            resolution = [1] * 3

        self.x_range = [0, 0]
        self.y_range = [0, 0]
        self.z_range = [0, 0]

        self.unit_name = unit_name

        if slices is not None:
            if start_coord is not None or end_coord is not None:
                raise ValueError(
                    "Both `slices` and `start_coord/end_coord` provided to "
                    "bounding cube constructor."
                )
            assert len(slices) == 3
            for s in slices:
                assert s.step is None

            self.x_range = [
                slices[0].start * resolution[0],
                slices[0].stop * resolution[0],
            ]
            self.y_range = [
                slices[1].start * resolution[1],
                slices[1].stop * resolution[1],
            ]
            self.z_range = [
                slices[2].start * resolution[2],
                slices[2].stop * resolution[2],
            ]
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

            if isinstance(start_coord, str):
                start_coord = [int(i) for i in start_coord.split(",")]
            if isinstance(end_coord, str):
                end_coord = [int(i) for i in end_coord.split(",")]

            assert len(start_coord) == 3
            assert len(end_coord) == 3

            self.x_range = [
                start_coord[0] * resolution[0],
                end_coord[0] * resolution[0],
            ]
            self.y_range = [
                start_coord[1] * resolution[1],
                end_coord[1] * resolution[1],
            ]
            self.z_range = [
                start_coord[2] * resolution[2],
                end_coord[2] * resolution[2],
            ]

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, BoundingCube)
            and (self.z_range == other.z_range)
            and (self.y_range == other.y_range)
            and (self.x_range == other.x_range)
            and (self.unit_name == other.unit_name)
        )

    def get_x_range(self, x_res: int = 1) -> List[int]:
        # scale_factor = 2 ** mip
        # xl = int(round((self.m0_x[1] - self.m0_x[0]) / scale_factor))
        # xs = floor(self.m0_x[0] / scale_factor)
        # return [xs, xs + xl]

        return [
            floor(self.x_range[0] / x_res),
            int(round(self.x_range[1] / x_res)),
        ]

    def get_y_range(self, y_res: int = 1) -> List[int]:
        return [
            floor(self.y_range[0] / y_res),
            int(round(self.y_range[1] / y_res)),
        ]

    def get_z_range(self, z_res: int = 1) -> List[int]:
        return [
            floor(self.z_range[0] / z_res),
            int(round(self.z_range[1] / z_res)),
        ]

    def pad(
        self,
        x_pad: int = 0,
        y_pad: int = 0,
        z_pad: int = 0,
        in_place: bool = False,
        resolution: Optional[List[int]] = None,
    ) -> Optional[BoundingCube]:
        """Pads the bounding box. Pad values are measured by the given resolution.
        if not provided, resolution defaults to [1, 1, 1]"""
        if resolution is None:
            resolution = [1, 1, 1]

        if in_place:
            raise NotImplementedError()

        result = BoundingCube(
            start_coord=[
                self.x_range[0] - x_pad * resolution[0],
                self.y_range[0] - y_pad * resolution[1],
                self.z_range[0] - z_pad * resolution[2],
            ],
            end_coord=[
                self.x_range[1] + x_pad * resolution[0],
                self.y_range[1] + y_pad * resolution[1],
                self.z_range[1] + z_pad * resolution[2],
            ],
            resolution=[1, 1, 1],
            unit_name=self.unit_name,
        )
        return result

    def clone(self) -> BoundingCube:  # pragma: no cover
        return copy.deepcopy(self)

    def copy(self) -> BoundingCube:  # pragma: no cover
        return self.clone()

    def __repr__(self) -> str:
        return (
            "BoundingCube("
            f"z: {self.z_range[0]}-{self.z_range[1]}, "
            f"y: {self.y_range[0]}-{self.y_range[1]}{self.unit_name}, "
            f"x: {self.x_range[0]}-{self.x_range[1]}{self.unit_name})"
        )
