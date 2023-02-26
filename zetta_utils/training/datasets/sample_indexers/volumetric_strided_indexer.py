from __future__ import annotations

from typing import Literal, Sequence

import attrs
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.geometry import BBox3D, BBoxStrider, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex

from .base import SampleIndexer


@builder.register("VolumetricStridedIndexer")
@typechecked
@attrs.frozen
class VolumetricStridedIndexer(SampleIndexer):
    """SampleIndexer which takes chunks from a volumetric region at uniform intervals.

    :param bbox: Bounding cube representing the whole volume to be indexed.
    :param resolution: Resolution at which ``chunk_size`` is given and which
        to specify in the resulting `VolumetricIndex`es.
    :param chunk_size: Size of a training chunk.
    :param stride: Distance between neighboring chunks along each dimension.
    :param mode: Behavior when bbox cannot be divided evenly.
    """

    bbox: BBox3D
    chunk_size: Sequence[int]
    stride: Sequence[int]
    resolution: Sequence[float]
    bbox_strider: BBoxStrider = attrs.field(init=False)
    mode: Literal["expand", "shrink"] = "expand"

    def __attrs_post_init__(self):
        # Use `__setattr__` to keep the object frozen.
        bbox_strider = BBoxStrider(
            bbox=self.bbox,
            resolution=Vec3D(*self.resolution),
            chunk_size=Vec3D[int](*self.chunk_size),
            stride=Vec3D[int](*self.stride),
            mode=self.mode,
        )
        object.__setattr__(self, "bbox_strider", bbox_strider)

    def __len__(self):
        return self.bbox_strider.num_chunks

    def __call__(self, idx: int) -> VolumetricIndex:
        """Translate a chunk index to a volumetric region in space.

        :param idx: chunk index.
        :return: VolumetricIndex.

        """
        sample_bbox = self.bbox_strider.get_nth_chunk_bbox(idx)

        result = VolumetricIndex(
            bbox=sample_bbox,
            resolution=Vec3D(*self.resolution),
        )
        return result
