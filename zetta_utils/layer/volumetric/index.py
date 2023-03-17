from __future__ import annotations

from typing import Literal, Optional, Sequence

import attrs
from typeguard import typechecked

from zetta_utils.geometry import BBox3D, Vec3D


@typechecked
@attrs.mutable
class VolumetricIndex:  # pragma: no cover # pure delegation, no logic
    resolution: Vec3D
    bbox: BBox3D
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

    def to_slices(self):
        return self.bbox.to_slices(self.resolution, self.allow_slice_rounding)

    def padded(self, pad: Sequence[int]) -> VolumetricIndex:
        return VolumetricIndex(
            bbox=self.bbox.padded(pad=pad, resolution=self.resolution),
            resolution=self.resolution,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def croppped(self, crop: Sequence[int]) -> VolumetricIndex:
        return VolumetricIndex(
            bbox=self.bbox.cropped(crop=crop, resolution=self.resolution),
            resolution=self.resolution,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def translated(self, offset: Sequence[float]) -> VolumetricIndex:
        return VolumetricIndex(
            bbox=self.bbox.translated(offset=offset, resolution=self.resolution),
            resolution=self.resolution,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def translated_start(self, offset: Sequence[float]) -> VolumetricIndex:
        return VolumetricIndex(
            bbox=self.bbox.translated_start(offset=offset, resolution=self.resolution),
            resolution=self.resolution,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def translated_end(self, offset: Sequence[float]) -> VolumetricIndex:
        return VolumetricIndex(
            bbox=self.bbox.translated_end(offset=offset, resolution=self.resolution),
            resolution=self.resolution,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def pformat(self, resolution: Optional[Sequence[float]] = None) -> str:
        if resolution is not None:
            resolution_ = resolution
        else:
            resolution_ = self.resolution
        return self.bbox.pformat(resolution_)

    def get_size(self):
        return self.bbox.get_size()

    def aligned(self, other: VolumetricIndex) -> tuple[bool, ...]:
        return self.bbox.aligned(other.bbox)

    def intersects(self, other: VolumetricIndex) -> bool:
        return self.bbox.intersects(other.bbox)

    def intersection(self, other: VolumetricIndex) -> VolumetricIndex:
        return VolumetricIndex(
            bbox=self.bbox.intersection(other.bbox),
            resolution=self.resolution,
            allow_slice_rounding=self.allow_slice_rounding,
        )

    def snapped(
        self,
        grid_offset: Sequence[int],
        grid_size: Sequence[int],
        mode: Literal["shrink", "expand"],
    ) -> VolumetricIndex:
        return VolumetricIndex(
            bbox=self.bbox.snapped(
                grid_offset=Vec3D(*grid_offset) * self.resolution,
                grid_size=Vec3D(*grid_size) * self.resolution,
                mode=mode,
            ),
            resolution=self.resolution,
            allow_slice_rounding=self.allow_slice_rounding,
        )
