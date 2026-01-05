from __future__ import annotations

from typing import Literal, Optional, Sequence

import attrs
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.geometry.bbox import Slices3D


def _convert_to_vec3d(value):
    """Convert sequence to Vec3D if needed."""
    if isinstance(value, Vec3D):
        return value
    return Vec3D(*value)


@typechecked
@attrs.mutable
# pylint: disable=too-many-public-methods # fundamental class
class VolumetricIndex:  # pragma: no cover # pure delegation, no logic
    """
    3D axis-aligned, resolution-aware bounding box.

    VolumetricIndex represents a cuboid region of space defined in terms
    of some 3D resolution.  So, while the bounding box is kept in nm,
    this class knows how to use and return coordinates in voxels (found
    by multiplying or dividing by the resolution in each dimension).

    It also has a couple of other properties commonly needed for
    subchunkable flow.

    :param resolution: size of one voxel, in nm, in each dimension.
    :param bbox: bounds of the volume, in nm.
    :param chunk_id: unique ID for the chunk, when this index is used to
            define a processing chunk in a subchunkable flow.
    :param allow_slice_rounding: Whether to allow representing bounding box
            with non-integer slice start/end at the given resolution.
    """

    resolution: Vec3D = attrs.field(converter=_convert_to_vec3d)
    bbox: BBox3D
    chunk_id: int = 0
    allow_slice_rounding: bool = False

    @property
    def start(self) -> Vec3D[int]:
        return Vec3D[int](*(e.start for e in self.to_slices()))

    @property
    def stop(self) -> Vec3D[int]:
        return Vec3D[int](*(e.stop for e in self.to_slices()))

    @property
    def shape(self) -> Vec3D[int]:
        return self.stop - self.start

    @classmethod
    def from_coords(
        cls,
        start_coord: Sequence[int],
        end_coord: Sequence[int],
        resolution: Vec3D | Sequence[float],
        chunk_id: int = 0,
        allow_slice_rounding: bool = False,
    ) -> VolumetricIndex:
        """
        Construct a VolumetricIndex from start and end coordinates.

        :param start_coord: start (minimum) coordinates, in voxels
        :param end_coord: end (maximum) coordinates, in voxels
        :param resolution: voxel size, in nm
        :param chunk_id: optional chunk ID to use
        :param allow_slice_rounding: Whether to allow representing slices
            with non-integer slice start/end at the given resolution.
        """
        return VolumetricIndex(
            bbox=BBox3D.from_coords(start_coord, end_coord, resolution),
            resolution=Vec3D(*resolution),
            chunk_id=chunk_id,
            allow_slice_rounding=allow_slice_rounding,
        )

    def __truediv__(self, vec: Vec3D) -> VolumetricIndex:
        return VolumetricIndex(
            self.resolution, self.bbox / vec, self.chunk_id, self.allow_slice_rounding
        )

    def __mul__(self, vec: Vec3D) -> VolumetricIndex:
        return VolumetricIndex(
            self.resolution, self.bbox * vec, self.chunk_id, self.allow_slice_rounding
        )

    def to_slices(self):
        """
        Represent this index as a tuple of slices.

        :return: Slices representing the bounding box.
        """
        return self.bbox.to_slices(self.resolution, self.allow_slice_rounding)

    def padded(self, pad: Sequence[int]) -> VolumetricIndex:
        """
        Return a new VolumetricIndex that is padded (expanded) relative
        to this one by the given amount in X, Y, and Z.

        :param pad: How much to pad along each dimension, in voxels
        """
        return VolumetricIndex(
            bbox=self.bbox.padded(pad=pad, resolution=self.resolution),
            resolution=self.resolution,
            chunk_id=self.chunk_id,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def cropped(self, crop: Sequence[int]) -> VolumetricIndex:
        """
        Return a new VolumetricIndex that is cropped (inset) relative
        to this one by the given amount in X, Y, and Z.

        :param crop: How much to inset each dimension, in voxels
        """
        return VolumetricIndex(
            bbox=self.bbox.cropped(crop=crop, resolution=self.resolution),
            resolution=self.resolution,
            chunk_id=self.chunk_id,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def split(self, num_splits: Sequence[int]) -> list[VolumetricIndex]:
        """
        Return a list of smaller VolumetricIndexes that represent subdivisions
        of this one.

        :param num_splits: How many bounding boxes to divide into along each
            dimension.
        """
        return [
            VolumetricIndex(
                bbox=split_bbox,
                resolution=self.resolution,
                chunk_id=self.chunk_id,
                allow_slice_rounding=self.allow_slice_rounding,
            )
            for split_bbox in self.bbox.split(num_splits)
        ]

    def translated(self, offset: Sequence[float]) -> VolumetricIndex:
        """
        Return a new VolumetricIndex that is translated (offset) by a given
        amount relative to this one.

        :param offset: amount to add to each bounds in X, Y, and Z (in voxels)
        """
        return VolumetricIndex(
            bbox=self.bbox.translated(offset=offset, resolution=self.resolution),
            resolution=self.resolution,
            chunk_id=self.chunk_id,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def translated_start(self, offset: Sequence[float]) -> VolumetricIndex:
        """
        Return a new VolumetricIndex with a start (but not end) offset by
        the given amount.

        :param offset: amount to add to start in X, Y, and Z (in voxels)
        """
        return VolumetricIndex(
            bbox=self.bbox.translated_start(offset=offset, resolution=self.resolution),
            resolution=self.resolution,
            chunk_id=self.chunk_id,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def translated_end(self, offset: Sequence[float]) -> VolumetricIndex:
        """
        Return a new VolumetricIndex with an end (but not start) offset by
        the given amount.

        :param offset: amount to add to end in X, Y, and Z (in voxels)
        """
        return VolumetricIndex(
            bbox=self.bbox.translated_end(offset=offset, resolution=self.resolution),
            resolution=self.resolution,
            chunk_id=self.chunk_id,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def transposed(self, dim0: int, dim1: int, local: bool = False) -> VolumetricIndex:
        """
        Return a new VolumetricIndex with two dimensions transposed
        relative to this one.

        :param dim0: The first dimension to be transposed
        :param dim1: The second dimension to be transposed
        :param local: Whether to transpose with respect to the local coordinate
        system (i.e., relative to self.start)
        :return: Transposed VolumetricIndex.
        """
        return VolumetricIndex(
            bbox=self.bbox.transposed(dim0, dim1, local),
            resolution=self.resolution,
            chunk_id=self.chunk_id,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def pformat(self, resolution: Optional[Sequence[float]] = None) -> str:
        """
        Returns a pretty formatted string for the bounding box at the given
        resolution (or if omitted, the resolution of this VolumetricIndex)
        that is suitable for copying into Neuroglancer. For a 3D bbox, the
        string is of the form ``(x_start, y_start, z_start) - (x_end, y_end, z_end)``.

        :param resolution: optional resolution to use; if omitted, uses
        self.resolution.
        """
        if resolution is not None:
            resolution_ = resolution
        else:
            resolution_ = self.resolution
        return self.bbox.pformat(resolution_)

    def get_size(self):
        """
        Returns the volume of the box, in base units (i.e. `nm^3`).

        Note that this method assumes the bounds are valid, i.e., the
        end coordinate is >= the start coordinate in every dimension.
        """
        return self.bbox.get_size()

    def aligned(self, other: VolumetricIndex) -> tuple[bool, ...]:
        """
        Returns whether two BBox3Ds are aligned (equal) at each edge
        of the bounds, as a tuple of bool values in the ordr:
        (x_start, x_stop, y_start, y_stop, z_start, z_stop)."""
        return self.bbox.aligned(other.bbox)

    def contained_in(self, other: VolumetricIndex) -> bool:
        """
        Return whether this volume is contained within the given one.
        """
        return self.bbox.contained_in(other.bbox)

    def intersects(self, other: VolumetricIndex) -> bool:
        """
        Return whether this volume intersects the given one.
        """
        return self.bbox.intersects(other.bbox)

    def intersection(self, other: VolumetricIndex) -> VolumetricIndex:
        """
        Return the intersection of this volume with the given other.
        Note that the resolution of the returned VolumetricIndex will
        be set to self.resolution (even if other.resolution differs).
        """
        return VolumetricIndex(
            bbox=self.bbox.intersection(other.bbox),
            resolution=self.resolution,
            chunk_id=self.chunk_id,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def get_intersection_and_subindex(
        self, large: VolumetricIndex
    ) -> tuple[VolumetricIndex, Slices3D]:
        """
        Given a 'larger' VolumetricIndex, returns the intersection VolumetricIndex of
        VolumetricIndex of the two as well as the slices for that intersection within the
        large VolumetricIndex.
        """
        intersection = self.intersection(large)
        subindex = intersection.translated(-large.start).to_slices()
        return intersection, subindex

    def snapped(
        self,
        grid_offset: Sequence[int],
        grid_size: Sequence[int],
        mode: Literal["shrink", "expand"],
    ) -> VolumetricIndex:
        """
        Returns a Volumetric snapped to a grid with the given offset and size.

        :param grid_offset: The offset of the grid to snap to (in voxels).
        :param grid_size: The size of the grid to snap to (in voxels).
        :param mode: Whether to ``shrink`` to the given grid (discard partial boxes) or
            to ``expand`` to the given grid (fill partial boxes).
        """
        return VolumetricIndex(
            bbox=self.bbox.snapped(
                grid_offset=Vec3D(*grid_offset) * self.resolution,
                grid_size=Vec3D(*grid_size) * self.resolution,
                mode=mode,
            ),
            resolution=self.resolution,
            chunk_id=self.chunk_id,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def supremum(self: VolumetricIndex, other: VolumetricIndex) -> VolumetricIndex:
        """
        Returns the the smallest volume which contains both self and other
        (equivalent to the union if the two volumes are edge-aligned in two
        dimensions, and are contiguous or overlap).
        """
        return VolumetricIndex(
            bbox=self.bbox.supremum(other.bbox),
            resolution=self.resolution,
            chunk_id=self.chunk_id,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def contains(self: VolumetricIndex, point: Sequence[float]) -> bool:
        """
        Returns whether the given point is within the bounds of this volume.
        Note that bounds in each dimension are semi-inclusive, so this method will
        return True for a point directly on a minimum bound, but not for a point
        on a maximum bound.

        :param point: point of interest, in voxels.
        """
        return self.bbox.contains(point, self.resolution)

    def line_intersects(
        self: VolumetricIndex, pointA: Sequence[float], pointB: Sequence[float]
    ) -> bool:
        """
        Returns whether the line segment from endpointA to endpointB intersects
        this volume.

        :param endpointA: one end point of the line, in voxels.
        :param endpointB: other end point of the line, in voxels.
        """
        return self.bbox.line_intersects(pointA, pointB, self.resolution)


builder.register("VolumetricIndex")(VolumetricIndex)
builder.register("VolumetricIndex.from_coords")(VolumetricIndex.from_coords)
