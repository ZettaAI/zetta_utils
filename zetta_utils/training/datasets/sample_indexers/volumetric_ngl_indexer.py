from __future__ import annotations

from typing import Sequence

import attrs
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.parsing import ngl_state

from .base import SampleIndexer


@builder.register("VolumetricNGLIndexer")
@typechecked
@attrs.frozen
class VolumetricNGLIndexer(SampleIndexer):
    """SampleIndexer which takes chunks from regions specified by a neuroglancer layer (NGL)
    annotation. The annotations is assumed to contain only points.
    For points, the chunk of the given size centered around the point will be taken.
    Index coordinates may be rounded.


    BCUBE ANNOTATIONS NOT IMPLEMENETED YET.
    For bboxes, the region specified by the bbox will be strided over with the given
    chunk size/stride. The region will rounded down to divide evenly.

    :param path: Path to the NGL annotation.
    :param resolution: Resolution at which ``chunk_size`` is given.
    :param chunk_size: Size of a chunk.

    """

    path: str
    chunk_size: Sequence[int]
    resolution: Sequence[float]
    annotations: list[Vec3D] = attrs.field(init=False)

    # stride: Sequence[int]
    #:param stride: Distance between neighboring chunks along each dimension for bbox annotations.
    # bbox_strider: BBoxStrider = attrs.field(init=False)

    def __attrs_post_init__(self):
        # Use `__setattr__` to keep the object frozen.
        annotations = ngl_state.read_remote_annotations(self.path)
        object.__setattr__(self, "annotations", annotations)

    def __len__(self):
        return len(self.annotations)

    def __call__(self, idx: int) -> VolumetricIndex:
        """Translate a chunk index to a volumetric region in space.

        :param idx: chunk index.
        :return: Volumetric index for the sample
        """
        point = Vec3D[float](*self.annotations[idx])
        resolution = Vec3D[float](*self.resolution)
        chunk_size = Vec3D[int](*self.chunk_size)

        start_coord_raw = point - chunk_size * resolution / 2
        # Offset to start with the given coord. Important for Z
        start_coord_raw += resolution
        # Round to get exact coords
        start_coord = (start_coord_raw // resolution) * resolution
        end_coord = start_coord + chunk_size * resolution

        sample_bbox = BBox3D.from_coords(start_coord, end_coord, resolution=Vec3D(1, 1, 1))

        result = VolumetricIndex(bbox=sample_bbox, resolution=resolution)

        return result
