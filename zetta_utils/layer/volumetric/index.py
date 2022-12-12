# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Optional

import attrs

from zetta_utils import builder

# from zetta_utils.partial import ComparablePartial
from zetta_utils.bcube import BoundingCube
from zetta_utils.typing import Vec3D


@builder.register("VolumetricIndex")
@attrs.mutable
class VolumetricIndex:  # pragma: no cover # pure delegation, no logic
    resolution: Vec3D
    bcube: BoundingCube
    allow_slice_rounding: bool = False

    def to_slices(self):
        return self.bcube.to_slices(self.resolution, self.allow_slice_rounding)

    def pad(self, pad: Vec3D):
        return VolumetricIndex(
            bcube=self.bcube.pad(pad=pad, resolution=self.resolution),
            resolution=self.resolution,
        )

    def crop(self, crop: Vec3D):
        return VolumetricIndex(
            bcube=self.bcube.crop(crop=crop, resolution=self.resolution),
            resolution=self.resolution,
        )

    def translate(self, offset: Vec3D):
        return VolumetricIndex(
            bcube=self.bcube.translate(offset=offset, resolution=self.resolution),
            resolution=self.resolution,
        )

    def pformat(self, resolution: Optional[Vec3D] = None):  # pragma: no cover
        return self.bcube.pformat(resolution)

    def get_size(self):  # pragma: no cover
        return self.bcube.get_size()
