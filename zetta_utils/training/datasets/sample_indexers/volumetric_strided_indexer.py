from typing import Optional, Tuple

import attrs
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.bcube import BcubeStrider, BoundingCube
from zetta_utils.typing import IntVec3D, Vec3D

from .base import SampleIndexer


@builder.register(
    "VolumetricStridedIndexer",
    cast_to_vec3d=["resolution", "index_resolution", "desired_resolution"],
    cast_to_intvec3d=["chunk_size", "stride"],
)
@typechecked
@attrs.frozen
class VolumetricStridedIndexer(SampleIndexer):
    """SampleIndexer which takes chunks from a volumetric region at uniform intervals.

    :param bcube: Bounding cube representing the whole volume to be indexed.
    :param resolution: Resolution at which ``chunk_size`` is given.
    :param chunk_size: Size of a training chunk.
    :param stride: Distance between neighboring chunks along each dimension.

    :param index_resolution: Resolution at at which to form an index for each chunk.
        When ``index_resolution is None``, the `resolution` value will be used.
    :param desired_resolution: Desired resolution which will be indicated as a part
        of index for each chunk. When ``desired_resolution is None``, no desired
        resolution will be specified in the index.

    """

    bcube: BoundingCube
    chunk_size: IntVec3D
    stride: IntVec3D
    resolution: Vec3D
    index_resolution: Optional[Vec3D] = None
    desired_resolution: Optional[Vec3D] = None
    bcube_strider: BcubeStrider = attrs.field(init=False)

    def __attrs_post_init__(self):
        # Use `__setattr__` to keep the object frozen.
        bcube_strider = BcubeStrider(
            bcube=self.bcube,
            resolution=self.resolution,
            chunk_size=self.chunk_size,
            stride=self.stride,
        )
        object.__setattr__(self, "bcube_strider", bcube_strider)

    def __len__(self):
        return self.bcube_strider.num_chunks

    def __call__(self, idx: int) -> Tuple[Optional[Vec3D], slice, slice, slice]:
        """Translate a chunk index to a volumetric region in space.

        :param idx: chunk index.
        :return: Volumetric index for the training chunk chunk, including
            ``self.desired_resolution`` and the slice representation of the region
            at ``self.index_resolution``.

        """
        sample_bcube = self.bcube_strider.get_nth_chunk_bcube(idx)
        if self.index_resolution is not None:
            slices = sample_bcube.to_slices(resolution=self.index_resolution)
        else:
            slices = sample_bcube.to_slices(resolution=self.resolution)

        result = (self.desired_resolution, slices[0], slices[1], slices[2])
        return result
