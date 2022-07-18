# pylint: disable=missing-docstring
import copy

# from math import floor, ceil


class BoundingCube:
    """Represents a 3D cuboid. X and Y coordinates are represented with
    the given resolution measure, while Z is assumed to be indivisible."""

    def __init__(
        self,
        xy_res: int = 1,
        slices: list = None,
        start_coord: str = None,
        end_coord: str = None,
        unit_name: str = "nm",
    ):
        self.z_range = [0, 0]
        self.y_range = [0, 0]
        self.x_range = [0, 0]
        self.unit_name = unit_name

        if slices is not None:
            assert (
                start_coord is None and end_coord is None
            )  # TODO: raise appropriate exc
            assert len(slices) == 3
            for s in slices:
                assert s.step is None

            self.z_range = [slices[0].start, slices[0].stop]
            self.y_range = [slices[1].start * xy_res, slices[1].stop * xy_res]
            self.x_range = [slices[2].start * xy_res, slices[2].stop * xy_res]

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

    def get_x_range(self, xy_res):
        # scale_factor = 2 ** mip
        # xl = int(round((self.m0_x[1] - self.m0_x[0]) / scale_factor))
        # xs = floor(self.m0_x[0] / scale_factor)
        # return [xs, xs + xl]
        return [
            self.x_range[0] / xy_res,
            self.x_range[1] / xy_res,
        ]

    def get_y_range(self, xy_res):
        return [
            self.y_range[0] / xy_res,
            self.y_range[1] / xy_res,
        ]

    def get_z_range(self):
        return copy.deepcopy(self.z_range)

    def get_x_size(self, xy_res: int = 1):
        x_range = self.get_x_range(xy_res)
        return int(x_range[1] - x_range[0])

    def get_z_size(self):
        return int(self.z_range[1] - self.z_range[0])

    def pad(self, xy_pad=0, z_pad=0, xy_res=1, in_place=False):
        """Cube the bounding box by `xy_pad` at the given `xy_res` on the both sides
        along the XY dimensions and by `z_pad` on both sides along the Z dimension
        """
        if in_place:
            raise NotImplementedError()

        result = BoundingCube(
            slices=[
                [self.z_range[0] - z_pad, self.z_range[1] + z_pad],
                [self.y_range[0] - xy_pad * xy_res, self.y_range[1] + xy_pad * xy_res],
                [self.x_range[0] - xy_pad * xy_res, self.x_range[1] + xy_pad * xy_res],
            ],
            xy_res=1,
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
