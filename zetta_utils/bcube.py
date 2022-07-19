# pylint: disable=missing-docstring
import copy

# from math import floor, ceil

from typing import Optional, List, Union

# TODO: delete
DEFAULT_RESOLUTION = 1


def dim_name_to_idx(dim_name: str):
    dim_name = dim_name.lower()
    if dim_name == "x":
        dim_idx = 0
    elif dim_name == "y":
        dim_idx = 1
    elif dim_name == "z":
        dim_idx = 2
    else:
        raise ValueError(f"Unsupported dimension '{dim_name}'")

    return dim_idx


def ___get_dim_res(
    dim: Union[str, int], resolution: Optional[Union[int, List[int]]] = None
):
    res = DEFAULT_RESOLUTION

    if isinstance(resolution, int):
        # if only one int is given as resolution, assumed it's for the dimension
        # of iterest
        res = resolution
    else:
        assert isinstance(resolution, list)

        if isinstance(dim, str):
            dim_idx = dim_name_to_idx(dim)
        else:
            dim_idx = dim

        assert dim_idx >= 0
        assert dim_idx < 3
        res = resolution[dim_idx]

    return res


class BoundingCube:
    """Represents a 3D cuboid in space."""

    def __init__(
        self,
        slices: Optional[list] = None,
        start_coord: Optional[str] = None,
        end_coord: Optional[str] = None,
        unit_name: str = "nm",
        resolution: Optional[List[int]] = None,
    ):
        if resolution is None:
            resolution = [DEFAULT_RESOLUTION] * 3

        self.x_range = [0, 0]
        self.y_range = [0, 0]
        self.z_range = [0, 0]

        self.unit_name = unit_name

        if slices is not None:
            assert (
                start_coord is None and end_coord is None
            )  # TODO: raise appropriate exc
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

        if start_coord is not None:
            raise NotImplementedError()
            # TODO: raise appropriate exc

    def __eq__(self, other):
        return (
            (self.z_range == other.z_range)
            and (self.y_range == other.y_range)
            and (self.x_range == other.x_range)
            and (self.unit_name == other.unit_name)
        )

    def get_x_range(self, x_res: int = 1):
        # scale_factor = 2 ** mip
        # xl = int(round((self.m0_x[1] - self.m0_x[0]) / scale_factor))
        # xs = floor(self.m0_x[0] / scale_factor)
        # return [xs, xs + xl]

        return [
            self.x_range[0] / x_res,
            self.x_range[1] / x_res,
        ]

    def get_y_range(self, y_res: int = 1):
        return [
            self.y_range[0] / y_res,
            self.y_range[1] / y_res,
        ]

    def get_z_range(self, z_res: int = 1):
        return [
            self.z_range[0] / z_res,
            self.z_range[1] / z_res,
        ]

    def pad(
        self,
        x_pad: int = 0,
        y_pad: int = 0,
        z_pad: int = 0,
        in_place: bool = False,
        resolution: Optional[List[int]] = None,
    ):
        """Pads the bounding box. Pad values are measured by the given resolution.
        if not provided, resolution defaults to [1, 1, 1]"""
        if resolution is None:
            resolution = [1, 1, 1]

        if in_place:
            raise NotImplementedError()

        result = BoundingCube(
            slices=[
                [
                    self.x_range[0] - x_pad * resolution[0],
                    self.x_range[1] + x_pad * resolution[0],
                ],
                [
                    self.y_range[0] - y_pad * resolution[1],
                    self.y_range[1] + y_pad * resolution[1],
                ],
                [
                    self.z_range[0] - z_pad * resolution[2],
                    self.z_range[1] + z_pad * resolution[2],
                ],
            ],
            resolution=[1, 1, 1],
            unit_name=self.unit_name,
        )
        return result

    def clone(self):
        return copy.deepcopy(self)

    def copy(self):
        return self.clone()

    def __repr__(self):
        return (
            "BoundingCube("
            f"z: {self.z_range[0]}-{self.z_range[1]}, "
            f"y: {self.y_range[0]}-{self.y_range[1]}{self.unit_name}, "
            f"x: {self.x_range[0]}-{self.x_range[1]}{self.unit_name})"
        )
