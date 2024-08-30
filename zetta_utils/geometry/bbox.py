# pylint: disable=missing-docstring, no-else-raise
from __future__ import annotations

from itertools import product
from math import floor
from typing import Literal, Optional, Sequence, Union, cast
from neuroglancer.viewer_state import AxisAlignedBoundingBoxAnnotation
    
import attrs
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.geometry.vec import VEC3D_PRECISION

from . import Vec3D

Slices3D = tuple[slice, slice, slice]
EPS = 1e-4

DEFAULT_UNIT = "nm"
Tuple2D = tuple[float, float]


@attrs.frozen()
@typechecked
class BBox3D:  # pylint: disable=too-many-public-methods # fundamental class
    """
    3-Dimensional axis-aligned cuboid in space.

    By convention, values in a BBox3D always represent nm (the "unit" property
    used only as a sanity check in methods like `intersects`).  In particular,
    do not use a BBox3D to represent a bounding box in voxel space; see
    VolumetricIndex for that.  When a BBox3D is created with a 'resolution'
    argument, the values of the other arguments are immediately multiplied by
    resolution, resulting in values in nm.  To convert from a BBox3D (in nm)
    back to voxels at some resolution, you can use get_slice or from_slices.

    :param bounds: (Min, max) values along X, Y, Z dimensions.
    :param unit: Unit name (for decorative or validation purposes only).
    :param pprint_px_resolution: Resolution used ONLY by the pformat method
    (for pretty-printed coordinates suitable for use with Neuroglancer).
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

    @staticmethod
    def from_ng_bbox(
        ng_bbox: AxisAlignedBoundingBoxAnnotation,
        base_resolution: Sequence[float]
    ) -> BBox3D:
        point_a_nm = Vec3D(*ng_bbox.pointA).int() * Vec3D(*base_resolution)
        point_b_nm = Vec3D(*ng_bbox.pointB).int() * Vec3D(*base_resolution)
        start_coord = [min(point_a_nm[i], point_b_nm[i]) for i in range(3)]
        end_coord = [max(point_a_nm[i], point_b_nm[i]) for i in range(3)]
        bbox = BBox3D.from_coords(
            start_coord=start_coord,
            end_coord=end_coord,
            resolution=[1, 1, 1] 
        )
        return bbox 
    
    @classmethod
    def from_slices(
        cls,
        slices: Slices3D,
        resolution: Sequence[float] = (1, 1, 1),
        unit: str = DEFAULT_UNIT,
    ) -> BBox3D:
        """Create a `BBox3D` from slices at the given resolution.

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

    @classmethod
    def from_points(
        cls,
        points: Sequence[Sequence[float]],
        resolution: Sequence[float] = (1, 1, 1),
        unit: str = DEFAULT_UNIT,
        epsilon: float = EPS,
    ) -> BBox3D:
        """Create a `BBox3D` tightly enclosing a set of points.  Note that
        since bounds are considered semi-inclusive, a small epsilon is
        added to the upper bounds; otherwise, the resulting box would not
        contain points lying on the upper bound of any dimension.

        :param points: sequence of 3D points to enclose.
        :param resolution: Resolution at which point coordinates are given.
            If not given, assumed to be unit resolution.
        :param unit: Unit name (decorative purposes only).
        :param epsilon: extra amount (in nm) added to upper bound to cause
        resulting box to actually contain all points.
        """
        if not points or len(points[0]) != 3 or len(resolution) != 3:
            raise ValueError("Only 3-dimensional points and resolution are supported.")

        # Calculate min and max bounds for each dimension using zip
        min_bounds = [min(p[i] * r for p in points) for i, r in enumerate(resolution)]
        max_bounds = [max(p[i] * r + epsilon for p in points) for i, r in enumerate(resolution)]

        bounds = cast(
            tuple[Tuple2D, Tuple2D, Tuple2D],
            tuple((min_b, max_b) for min_b, max_b in zip(min_bounds, max_bounds)),
        )

        return cls(bounds=bounds, unit=unit)

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

        dim_range_start_raw = round(self.bounds[dim][0] / dim_res, VEC3D_PRECISION)
        dim_range_end_raw = round(self.bounds[dim][1] / dim_res, VEC3D_PRECISION)

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
        """Create a padded (i.e. expanded) version of this bounding box.

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

    def split(
        self,
        num_splits: Sequence[int],
    ) -> list[BBox3D]:
        """Create a list of bounding boxes formed by splitting this bounding box
        evenly by the vector ``num_splits`` in each dimension.

        :param num_splits: How many bounding boxes to divide into along each
            dimension.
        :return: List of split bounding boxes.

        """
        if len(num_splits) != 3:
            raise ValueError("Number of splits must be 3-dimensional.")

        num_splits = Vec3D(*num_splits)
        stride = self.shape / num_splits
        splits: list[Vec3D] = [Vec3D(*k) for k in product(*(range(n) for n in num_splits))]
        return [
            BBox3D.from_coords(
                start_coord=self.start + split * stride,
                end_coord=self.start + (split + 1) * stride,
                unit=self.unit,
            )
            for split in splits
        ]

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
        """Create a version of the bounding box where the start (and not the end)
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

    def transposed(
        self,
        dim0: int,
        dim1: int,
        local: bool = False,
    ) -> BBox3D:
        """Transpose the bounding box.

        :param dim0: The first dimension to be transposed
        :param dim1: The second dimension to be transposed
        :param local: Whether to transpose with respect to the local coordinate
        system (i.e., relative to self.start)
        :return: Transposed bounding box.

        """
        assert -3 <= dim0 < 3
        assert -3 <= dim1 < 3

        # Make sure dims are in [0, 1, 2]
        dim0 = dim0 + 3 if dim0 < 0 else dim0
        dim1 = dim1 + 3 if dim1 < 0 else dim1

        # Mapping for transposing dims
        mapping = {i: i for i in range(self.ndim)}
        mapping[dim0] = dim1
        mapping[dim1] = dim0

        # Translate the bbox to the origin if local
        offset = -self.start if local else Vec3D(0, 0, 0)
        bbox = self.translated(offset, resolution=Vec3D(1, 1, 1))

        # Transposed slices
        slices = cast(
            Slices3D,
            tuple(bbox.get_slice(mapping[i], resolution=1) for i in range(self.ndim)),
        )

        # Create a transposed bbox
        transposed = BBox3D.from_slices(
            slices=slices,
            unit=self.unit,
        )

        # Translate back if local
        offset = self.start if local else Vec3D(0, 0, 0)
        result = transposed.translated(offset, resolution=Vec3D(1, 1, 1))
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
            to ``expand`` to the given grid (fill partial boxes).
        """
        if len(grid_offset) != 3 or len(grid_size) != 3:  # pragma: no cover
            raise ValueError("Only 3-dimensional inputs are supported.")

        if mode == "shrink":
            start_final = tuple(
                floor(round((b[0] - o) / s + 1, VEC3D_PRECISION) - EPS) * s + o
                for b, o, s in zip(self.bounds, grid_offset, grid_size)
            )
            end_final = tuple(
                floor(round((b[1] - o) / s, VEC3D_PRECISION)) * s + o
                for b, o, s in zip(self.bounds, grid_offset, grid_size)
            )
        else:
            assert mode == "expand", "Typechecking error"
            start_final = tuple(
                floor(round((b[0] - o) / s, VEC3D_PRECISION)) * s + o
                for b, o, s in zip(self.bounds, grid_offset, grid_size)
            )
            end_final = tuple(
                floor(round((b[1] - o) / s + 1, VEC3D_PRECISION) - EPS) * s + o
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
        string is of the form ``(x_start, y_start, z_start) - (x_end, y_end, z_end)``.

        :param resolution: optional resolution to use; if omitted, uses
        self.pprint_px_resolution.
        """

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

    def get_size(self) -> Union[int, float]:  # pragma: no cover
        """Returns the volume of the box, in base units (i.e. `nm^3`)."""
        resolution = (1, 1, 1)
        slices = self.to_slices(resolution, round_to_int=False)
        size = 1
        for _, slc in enumerate(slices):
            size *= slc.stop - slc.start
        return size

    def aligned(self, other: BBox3D) -> tuple[bool, ...]:
        assert self.unit == other.unit
        """Returns whether two BBox3Ds are aligned, in
        x_start, x_stop, y_start, y_stop, z_start, z_stop order."""
        return tuple(s[i] == o[i] for s, o in zip(self.bounds, other.bounds) for i in range(2))

    def contained_in(self: BBox3D, other: BBox3D) -> bool:
        assert self.unit == other.unit
        """Returns whether the other BBox3D contains this one."""
        return all(
            (self_b[0] >= other_b[0] and other_b[1] >= self_b[1])
            for self_b, other_b in zip(self.bounds, other.bounds)
        )

    def intersects(self: BBox3D, other: BBox3D) -> bool:
        """Returns whether the other BBox3D intersects this one.
        Note that the `unit` property must match."""
        assert self.unit == other.unit
        return all(
            (self_b[1] > other_b[0] and other_b[1] > self_b[0])
            for self_b, other_b in zip(self.bounds, other.bounds)
        )

    def intersection(self: BBox3D, other: BBox3D) -> BBox3D:
        """Returns the intersection of another BBox3D with this one.
        The `unit` property must match, but pprint_px_resolution is ignored.
        The resulting bounds will be all `0` if they do not intersect.
        """
        assert self.unit == other.unit
        if not self.intersects(other):
            return BBox3D(((0, 0), (0, 0), (0, 0)), unit=self.unit)
        bounds = cast(
            tuple[Tuple2D, Tuple2D, Tuple2D],
            tuple((max(s[0], o[0]), min(s[1], o[1])) for s, o in zip(self.bounds, other.bounds)),
        )
        return BBox3D(bounds=bounds, unit=self.unit)

    def supremum(self: BBox3D, other: BBox3D) -> BBox3D:
        """Returns the the smallest bounding box which contains both self
        and other (equivalent to the union if the two BBox3Ds are edge-aligned
        in two dimensions, and are contiguous or overlap).

        The `unit` property must match, but pprint_px_resolution is ignored.
        """
        assert self.unit == other.unit
        bounds = cast(
            tuple[Tuple2D, Tuple2D, Tuple2D],
            tuple((min(s[0], o[0]), max(s[1], o[1])) for s, o in zip(self.bounds, other.bounds)),
        )
        return BBox3D(bounds=bounds, unit=self.unit)

    def contains(self: BBox3D, point: Sequence[float], resolution: Sequence[float]) -> bool:
        """Returns whether the given point is within the bounds of this BBox3D.
        Note that bounds in each dimension are semi-inclusive, so this method will
        return True for a point directly on a minimum bound, but not for a point
        on a maximum bound.

        :param point: point of interest.
        :param resolution: Resolution at which ``point`` was given.
        """
        if len(point) != 3 or len(resolution) != 3:
            raise ValueError("Only 3-dimensional points and resolution are supported.")
        point_in_nm = [p * r for p, r in zip(point, resolution)]
        return all(self.bounds[i][0] <= point_in_nm[i] < self.bounds[i][1] for i in range(3))

    def line_intersects(
        self, endpoint1: Sequence[float], endpoint2: Sequence[float], resolution: Sequence[float]
    ) -> bool:
        """Returns whether the line segment from endpoint1 to endpoint2 intersects
        this BBox3D.

        :param endpoint1: one endpoint of the line.
        :param endpoint2: other endpoint of the line.
        :param resolution: Resolution and which the endpoints were given.
        """
        if len(endpoint1) != 3 or len(endpoint2) != 3 or len(resolution) != 3:
            raise ValueError("Only 3-dimensional points and resolution are supported.")

        # Early out: if either point is inside the box, return True
        if self.contains(endpoint1, resolution) or self.contains(endpoint2, resolution):
            return True

        # Convert endpoints to nanometer coordinates
        point1_in_nm = [p * r for p, r in zip(endpoint1, resolution)]
        point2_in_nm = [p * r for p, r in zip(endpoint2, resolution)]

        # Liang-Barsky line clipping algorithm adapted for a 3D box
        tmin, tmax = 0.0, 1.0
        for i in range(3):  # Iterate over X, Y, Z axes
            p1, p2 = point1_in_nm[i], point2_in_nm[i]
            box_min, box_max = self.bounds[i]

            direction = p2 - p1
            if direction == 0:
                if p1 < box_min or p1 >= box_max:
                    return False
            else:
                t1 = (box_min - p1) / direction
                t2 = (box_max - p1) / direction
                tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
                if tmin > tmax:
                    return False

        return True


builder.register("BBox3D.from_slices")(BBox3D.from_slices)
builder.register("BBox3D.from_coords")(BBox3D.from_coords)
builder.register("BBox3D.from_points")(BBox3D.from_points)
