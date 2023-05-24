# pylint: disable=missing-docstring, no-else-raise
from __future__ import annotations

from math import floor
from typing import Literal, Optional, Sequence, cast

import attrs
from typeguard import typechecked

from zetta_utils import builder

from . import Vec3D

Slices3D = tuple[slice, slice, slice]
EPS = 1e-4

DEFAULT_UNIT = "nm"
Tuple2D = tuple[float, float]


@attrs.frozen()
@typechecked
class BBox3D:
    """
    3-Dimentional cuboid in space.
    :param bounds: Bounds along X, Y, Z dimensions.
    :param unit: Unit name (for decorative purposes only).
    """

    bounds: tuple[Tuple2D, Tuple2D, Tuple2D]
    unit: str = DEFAULT_UNIT
    pprint_px_resolution: Sequence[float] = (1, 1, 1)

    @property
    def ndim(self) -> int:
        return 3

    @property
    def start(self) -> Vec3D:  # pragma: no cover
        """returns the start coordinates."""
        return Vec3D(*(b[0] for b in self.bounds))

    @property
    def end(self) -> Vec3D:  # pragma: no cover
        """returns the end coordinates."""
        return Vec3D(*(b[1] for b in self.bounds))

    @property
    def shape(self) -> Vec3D:  # pragma: no cover
        """returns the shape coordinates."""
        return self.end - self.start

    @classmethod
    def from_slices(
        cls,
        slices: Slices3D,
        resolution: Sequence[float] = (1, 1, 1),
        unit: str = DEFAULT_UNIT,
    ) -> BBox3D:
        """Create a `BBoxND` from slices at the given resolution.

        :param slices: Tuple of slices representing a bounding box.
        :param resolution: Resolution at which the slices are given.
            If not given, assumed to be unit resolution.
        :param unit: Unit name (decorative purposes only).

        """
        if len(resolution) != 3:
            raise ValueError("Only 3-dimensional inputs are supported.")

        for s in slices:
            if s.step is not None:
                raise ValueError(f"Cannot construct a boundingbox from strided slice: '{s}'.")

        bounds = cast(
            tuple[Tuple2D, Tuple2D, Tuple2D],
            tuple((float(s.start * r), float(s.stop * r)) for s, r in zip(slices, resolution)),
        )
        result = cls(bounds=bounds, unit=unit)
        return result

    @classmethod
    def from_coords(
        cls,
        start_coord: Sequence[float],
        end_coord: Sequence[float],
        resolution: Sequence[float] = (1, 1, 1),
        unit: str = DEFAULT_UNIT,
    ) -> BBox3D:
        """Create a `BBox3D` from start and end coordinates at the given resolution.

        :param start_coord: Start coordinate.
        :param end_coord: End coordinate.
        :param resolution: Resolution at which the coordinates are given.
            If not given, assumed to be unit resolution.
        :param unit: Unit name (decorative purposes only).

        """
        if len(start_coord) != 3 or len(end_coord) != 3 or len(resolution) != 3:
            raise ValueError("Only 3-dimensional inputs are supported.")

        bounds = cast(
            tuple[Tuple2D, Tuple2D, Tuple2D],
            tuple((s * r, e * r) for s, e, r in zip(start_coord, end_coord, resolution)),
        )

        result = cls(bounds=bounds, unit=unit)
        return result

    def get_slice(
        self,
        dim: int,
        resolution: int | float | Sequence[float],
        allow_slice_rounding: bool = False,
        round_to_int: bool = True,
    ) -> slice:
        """Represent the bounding box as a slice along the given dimension.

        :param dim: Dimension along which the slice will be taken.
        :param resolution: Resolution at which the slice will be taken.
        :param allow_slice_rounding: Whether to allow representing bounding box
            with non-integer slice start/end at the given resolution.
        :param round_to_int: Whether the slices should be returned as an int
            or whether they should be returned as raw. If set to `False`, takes
            precedence over `allow_slice_rounding`.
        :return: Slice representing the bounding box.

        """
        if isinstance(resolution, (int, float)):
            dim_res = resolution
        else:
            dim_res = resolution[dim]

        dim_range_start_raw = self.bounds[dim][0] / dim_res
        dim_range_end_raw = self.bounds[dim][1] / dim_res

        if not round_to_int:
            return slice(dim_range_start_raw, dim_range_end_raw)

        if not allow_slice_rounding:
            if dim_range_start_raw != round(dim_range_start_raw):
                raise ValueError(
                    f"{self} results in slice_start == "
                    f"{dim_range_start_raw} along dimension {dim} "
                    f"at resolution == {resolution} while "
                    "`allow_slice_rounding` == False."
                )
            if dim_range_end_raw != round(dim_range_end_raw):
                raise ValueError(
                    f"{self} results in slice_end == "
                    f"{dim_range_end_raw} along dimension {dim} "
                    f"at resolution == {resolution} while "
                    "`allow_slice_rounding` == False."
                )
        slice_length = int(round(dim_range_end_raw - dim_range_start_raw))
        result = slice(
            floor(dim_range_start_raw),
            floor(dim_range_start_raw) + slice_length,
        )
        return result

    def to_slices(
        self,
        resolution: Sequence[float],
        allow_slice_rounding: bool = False,
        round_to_int: bool = True,
    ) -> Slices3D:
        """Represent the bounding box as a tuple of slices.

        :param resolution: Resolution at which the slices will be taken.
        :param allow_slice_rounding: Whether to allow representing bounding box
            with non-integer slice start/end at the given resolution.
        :param round_to_int: Whether the slices should be returned as an int
            or whether they should be returned as raw. If set to `False`, takes
            precedence over `allow_slice_rounding`.
        :return: Slices representing the bounding box.

        """
        if len(resolution) != 3:
            raise ValueError("Only 3-dimensional inputs are supported.")

        result = cast(
            Slices3D,
            tuple(
                self.get_slice(i, resolution[i], allow_slice_rounding, round_to_int)
                for i in range(self.ndim)
            ),
        )
        return result

    def cropped(
        self,
        crop: Sequence[int | float | tuple[float, float]],
        resolution: Sequence[float],
    ) -> BBox3D:
        """Create a cropped version of this bounding box.

        :param crop: Specification of how much to crop along each dimension.
        :param resolution: Resolution at which ``crop`` specification was given.
        :return: Cropped bounding box.

        """
        if len(resolution) != 3 or len(crop) != 3:
            raise ValueError("Only 3-dimensional inputs are supported.")

        double_sided_crop = []
        for e in crop:
            if isinstance(e, (int, float)):
                double_sided_crop += [(e, e)]
            else:
                double_sided_crop += [e]

        slices = cast(
            Slices3D,
            tuple(
                slice(
                    s[0] + c[0] * r,
                    s[1] - c[1] * r,
                )
                for s, c, r in zip(self.bounds, double_sided_crop, resolution)
            ),
        )

        result = BBox3D.from_slices(
            slices=slices,
            unit=self.unit,
        )

        return result

    def padded(
        self,
        pad: Sequence[float | tuple[float, float]],
        resolution: Sequence[float],
    ) -> BBox3D:
        """Create a padded version of this bounding box.

        :param pad: Specification of how much to pad along each dimension.
        :param resolution: Resolution at which ``pad`` specification was given.
        :return: Padded bounding box.

        """
        if len(resolution) != 3 or len(pad) != 3:
            raise ValueError("Only 3-dimensional inputs are supported.")

        # if len(pad) != self.ndim:
        #    raise ValueError(
        #        f"Length of the padding specification ({len(pad)}) != "
        #        f"BBox3D ndim ({self.ndim})."
        #    )

        double_sided_pad = []
        for e in pad:
            if isinstance(e, (int, float)):
                double_sided_pad += [(e, e)]
            else:
                double_sided_pad += [e]

        slices = cast(
            Slices3D,
            tuple(
                slice(
                    s[0] - p[0] * r,
                    s[1] + p[1] * r,
                )
                for s, p, r in zip(self.bounds, double_sided_pad, resolution)
            ),
        )

        result = BBox3D.from_slices(
            slices=slices,
            unit=self.unit,
        )

        return result

    def translated(
        self,
        offset: Sequence[float],
        resolution: Sequence[float],
    ) -> BBox3D:
        """Create a translated version of this bounding box.

        :param offset: Specification of how much to translate along each dimension.
        :param resolution: Resolution at which ``offset`` specification was given.
        :return: Translated bounding box.

        """
        if len(resolution) != 3 or len(offset) != 3:
            raise ValueError("Only 3-dimensional inputs are supported.")

        slices = cast(
            Slices3D,
            tuple(
                slice(
                    s[0] + o * r,
                    s[1] + o * r,
                )
                for s, o, r in zip(self.bounds, offset, resolution)
            ),
        )

        result = BBox3D.from_slices(
            slices=slices,
            unit=self.unit,
        )

        return result

    def translated_start(
        self,
        offset: Sequence[float],
        resolution: Sequence[float],
    ) -> BBox3D:
        """Create a version of the bounding box where the start (and not the stop)
        has been moved by the given offset.

        :param offset: Specification of how much to translate along each dimension.
        :param resolution: Resolution at which ``offset`` specification was given.
        :return: Translated bounding box.

        """
        if len(offset) != 3 or len(resolution) != 3:
            raise ValueError("Only 3-dimensional inputs are supported.")

        slices = cast(
            Slices3D,
            tuple(slice(s[0] + o * r, s[1]) for s, o, r in zip(self.bounds, offset, resolution)),
        )

        result = BBox3D.from_slices(
            slices=slices,
            unit=self.unit,
        )

        return result

    def translated_end(
        self,
        offset: Sequence[float],
        resolution: Sequence[float],
    ) -> BBox3D:
        """Create a version of the bounding box where the end (and not the start)
        has been moved by the given offset.

        :param offset: Specification of how much to translate along each dimension.
        :param resolution: Resolution at which ``offset`` specification was given.
        :return: Translated bounding box.

        """
        if len(offset) != 3 or len(resolution) != 3:
            raise ValueError("Only 3-dimensional inputs are supported.")

        slices = cast(
            Slices3D,
            tuple(
                slice(
                    s[0],
                    s[1] + o * r,
                )
                for s, o, r in zip(self.bounds, offset, resolution)
            ),
        )

        result = BBox3D.from_slices(
            slices=slices,
            unit=self.unit,
        )

        return result

    def snapped(
        self,
        grid_offset: Sequence[float],
        grid_size: Sequence[float],
        mode: Literal["shrink", "expand"],
    ) -> BBox3D:
        """Returns a BoundingBox snapped to a grid with the given offset and size.

        :param grid_offset: The offset of the grid to snap to.
        :param grid_size: The size of the grid to snap to.
        :param mode: Whether to ``shrink`` to the given grid (discard partial boxes) or
            to ``expand`` to the given grid(fill partial boxes).
        """
        if len(grid_offset) != 3 or len(grid_size) != 3:  # pragma: no cover
            raise ValueError("Only 3-dimensional inputs are supported.")

        if mode == "shrink":
            start_final = tuple(
                ((b[0] - o - EPS) // s + 1) * s + o
                for b, o, s in zip(self.bounds, grid_offset, grid_size)
            )
            end_final = tuple(
                (b[1] - o) // s * s + o for b, o, s in zip(self.bounds, grid_offset, grid_size)
            )
        else:
            assert mode == "expand", "Typechecking error"
            start_final = tuple(
                (b[0] - o) // s * s + o for b, o, s in zip(self.bounds, grid_offset, grid_size)
            )
            end_final = tuple(
                ((b[1] - o - EPS) // s + 1) * s + o
                for b, o, s in zip(self.bounds, grid_offset, grid_size)
            )
        return BBox3D.from_coords(
            start_coord=cast(tuple[float, float, float], start_final),
            end_coord=cast(tuple[float, float, float], end_final),
            unit=self.unit,
        )

    def pformat(self, resolution: Optional[Sequence[float]] = None) -> str:  # pragma: no cover
        """Returns a pretty formatted string for this bounding box at the given
        resolution that is suitable for copying into neuroglancer. For a 3D bbox, the
        string is of the form ``(x_start, y_start, z_start) - (x_end, y_end, z_end)``."""

        if resolution is not None:
            if len(resolution) != 3:
                raise ValueError("Only 3-dimensional inputs are supported.")
            resolution_final = resolution
        else:
            resolution_final = self.pprint_px_resolution

        slices = self.to_slices(resolution_final, round_to_int=False)
        s = ", "

        return (
            f"({s.join([str(slice.start) for slice in slices])})" + " - "
            f"({s.join([str(slice.stop) for slice in slices])})"
        )

    def get_size(self) -> int | float:  # pragma: no cover
        """Returns the size of the volume in N-D space, in `self.unit^N`."""
        resolution = (1, 1, 1)
        slices = self.to_slices(resolution)
        size = 1
        for _, slc in enumerate(slices):
            size *= slc.stop - slc.start
        return size

    def aligned(self, other: BBox3D) -> tuple[bool, ...]:  # pragma: no cover
        assert self.unit == other.unit
        """Returns whether two BoundingBoxNDs are aligned, in
        x_start, x_stop, y_start, y_stop, z_start, z_stop order."""
        return tuple(s[i] == o[i] for s, o in zip(self.bounds, other.bounds) for i in range(2))

    def intersects(self: BBox3D, other: BBox3D) -> bool:  # pragma: no cover
        assert self.unit == other.unit
        """Returns whether two BoundingBoxNDs intersect."""
        return all(
            (self_b[1] > other_b[0] and other_b[1] > self_b[0])
            for self_b, other_b in zip(self.bounds, other.bounds)
        )

    def intersection(self: BBox3D, other: BBox3D) -> BBox3D:
        assert self.unit == other.unit
        if not self.intersects(other):
            return BBox3D(((0, 0), (0, 0), (0, 0)), unit=self.unit)
        """Returns the intersection of two BoundingBoxNDs."""
        bounds = cast(
            tuple[Tuple2D, Tuple2D, Tuple2D],
            tuple((max(s[0], o[0]), min(s[1], o[1])) for s, o in zip(self.bounds, other.bounds)),
        )
        return BBox3D(bounds=bounds, unit=self.unit)


builder.register("BBox3D.from_slices")(BBox3D.from_slices)
builder.register("BBox3D.from_coords")(BBox3D.from_coords)
