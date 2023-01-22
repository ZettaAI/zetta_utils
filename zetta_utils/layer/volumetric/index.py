# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Literal, Optional

import attrs

from zetta_utils import builder

# from zetta_utils.common.partial import ComparablePartial
from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D


@builder.register("VolumetricIndex", cast_to_vec3d=["resolution"])
@attrs.mutable
class VolumetricIndex:  # pragma: no cover # pure delegation, no logic
    resolution: Vec3D
    bbox: BBox3D
    allow_slice_rounding: bool = False

    @property
    def start(self) -> IntVec3D:
        return IntVec3D(*(e.start for e in self.to_slices()))

    @property
    def stop(self) -> IntVec3D:
        return IntVec3D(*(e.stop for e in self.to_slices()))

    @property
    def shape(self) -> IntVec3D:
        return self.stop - self.start

    def to_slices(self):
        return self.bbox.to_slices(self.resolution, self.allow_slice_rounding)

    def padded(self, pad: IntVec3D) -> VolumetricIndex:
        return VolumetricIndex(
            bbox=self.bbox.padded(pad=pad, resolution=self.resolution),
            resolution=self.resolution,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def croppped(self, crop: IntVec3D) -> VolumetricIndex:
        return VolumetricIndex(
            bbox=self.bbox.cropped(crop=crop, resolution=self.resolution),
            resolution=self.resolution,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def translated(self, offset: Vec3D) -> VolumetricIndex:
        return VolumetricIndex(
            bbox=self.bbox.translated(offset=offset, resolution=self.resolution),
            resolution=self.resolution,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def translated_start(self, offset: Vec3D) -> VolumetricIndex:
        return VolumetricIndex(
            bbox=self.bbox.translated_start(offset=offset, resolution=self.resolution),
            resolution=self.resolution,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def translated_stop(self, offset: Vec3D) -> VolumetricIndex:
        return VolumetricIndex(
            bbox=self.bbox.translated_end(offset=offset, resolution=self.resolution),
            resolution=self.resolution,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def pformat(self, resolution: Optional[Vec3D] = None) -> str:
        if resolution is None:
            resolution_ = self.resolution
        else:
            resolution_ = resolution
        return self.bbox.pformat(resolution_)

    def get_size(self):
        return self.bbox.get_size()

    def intersects(self, other: VolumetricIndex) -> bool:
        return self.bbox.intersects(other.bbox)

    def intersection(self, other: VolumetricIndex) -> VolumetricIndex:
        return VolumetricIndex(
            bbox=self.bbox.intersection(other.bbox),
            resolution=self.resolution,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def snapped(
        self, grid_offset: IntVec3D, grid_size: IntVec3D, mode: Literal["shrink", "expand"]
    ) -> VolumetricIndex:
        return VolumetricIndex(
            bbox=self.bbox.snapped(
                grid_offset=grid_offset * self.resolution,
                grid_size=grid_size * self.resolution,
                mode=mode,
            ),
            resolution=self.resolution,
            allow_slice_rounding=self.allow_slice_rounding,
        )
