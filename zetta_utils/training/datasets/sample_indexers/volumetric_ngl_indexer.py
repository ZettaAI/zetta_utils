from typing import List, Optional, Tuple

import attrs
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.bbox import BBox3D
from zetta_utils.parsing import ngl_state
from zetta_utils.typing import IntVec3D, Vec3D

from .base import SampleIndexer


@builder.register(
    "VolumetricNGLIndexer",
    cast_to_vec3d=["resolution", "index_resolution", "desired_resolution"],
    cast_to_intvec3d=["chunk_size", "stride"],
)
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

    :param index_resolution: Resolution at at which to form an index for each chunk.
        When ``index_resolution is None``, the `resolution` value will be used.
    :param desired_resolution: Desired resolution which will be indicated as a part
        of index for each chunk. When ``desired_resolution is None``, no desired
        resolution will be specified in the index.

    """

    path: str
    chunk_size: IntVec3D
    resolution: Vec3D
    index_resolution: Optional[Vec3D] = None
    desired_resolution: Optional[Vec3D] = None
    annotations: List[Vec3D] = attrs.field(init=False)

    # stride: IntVec3D
    #:param stride: Distance between neighboring chunks along each dimension for bbox annotations.
    # bbox_strider: BBoxStrider = attrs.field(init=False)

    def __attrs_post_init__(self):
        # Use `__setattr__` to keep the object frozen.
        annotations = ngl_state.read_remote_annotations(self.path)
        object.__setattr__(self, "annotations", annotations)

    def __len__(self):
        return len(self.annotations)

    def __call__(self, idx: int) -> Tuple[Optional[Vec3D], slice, slice, slice]:
        """Translate a chunk index to a volumetric region in space.

        :param idx: chunk index.
        :return: Volumetric index for the training chunk chunk, including
            ``self.desired_resolution`` and the slice representation of the region
            at ``self.index_resolution``.

        """
        point = self.annotations[idx]
        start_coord_raw = point - self.chunk_size * self.resolution / 2
        # Offset to start with the given coord. Important for Z
        start_coord_raw += self.resolution
        # Round to get exact coords
        start_coord = (start_coord_raw // self.resolution) * self.resolution
        end_coord = start_coord + self.chunk_size * self.resolution

        sample_bbox = BBox3D.from_coords(start_coord, end_coord, resolution=Vec3D(1, 1, 1))

        if self.index_resolution is not None:
            slices = sample_bbox.to_slices(resolution=self.index_resolution)
        else:
            slices = sample_bbox.to_slices(resolution=self.resolution)

        result = (self.desired_resolution, slices[0], slices[1], slices[2])
        return result
