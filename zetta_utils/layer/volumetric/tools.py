from typing import Iterable, Literal, Optional

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder, tensor_ops
from zetta_utils.bcube import BcubeStrider
from zetta_utils.typing import IntVec3D, Vec3D

from .. import DataWithIndexProcessor, IndexChunker
from . import VolumetricIndex


@typechecked
def translate_volumetric_index(
    idx: VolumetricIndex, offset: Vec3D, resolution: Vec3D
):  # pragma: no cover # under 3 statements, no conditionals
    bcube = idx.bcube.translate(offset, resolution)
    result = VolumetricIndex(
        bcube=bcube,
        resolution=idx.resolution,
    )
    return result


@builder.register("VolumetricIndexTranslator", cast_to_vec3d=["offset", "resolution"])
@typechecked
@attrs.mutable
class VolumetricIndexTranslator:  # pragma: no cover # under 3 statements, no conditionals
    offset: Vec3D
    resolution: Vec3D

    def __call__(self, idx: VolumetricIndex) -> VolumetricIndex:
        result = translate_volumetric_index(
            idx=idx,
            offset=self.offset,
            resolution=self.resolution,
        )
        return result


@builder.register("VolumetricIndexResolutionAdjuster", cast_to_vec3d=["resolution"])
@typechecked
@attrs.mutable
class VolumetricIndexResolutionAdjuster:  # pragma: no cover # under 3 statements, no conditionals
    resolution: Vec3D

    def __call__(self, idx: VolumetricIndex) -> VolumetricIndex:
        result = VolumetricIndex(
            bcube=idx.bcube,
            resolution=self.resolution,
        )
        return result


@builder.register("VolumetricIndexResolutionAdjuster")
@typechecked
@attrs.mutable
class VolDataInterpolator(DataWithIndexProcessor):
    interpolation_mode: tensor_ops.InterpolationMode
    mode: Literal["read", "write"]
    allow_slice_rounding: bool = False

    def __call__(
        self,
        data: torch.Tensor,
        idx: VolumetricIndex,
        idx_proced: VolumetricIndex,
    ) -> torch.Tensor:
        if self.mode == "read":
            scale_factor = tuple(idx_proced.resolution[i] / idx.resolution[i] for i in range(3))
        else:
            assert self.mode == "write"
            scale_factor = tuple(idx.resolution[i] / idx_proced.resolution[i] for i in range(3))

        result = tensor_ops.interpolate(
            data=data,
            scale_factor=scale_factor,
            mode=self.interpolation_mode,
            allow_slice_rounding=self.allow_slice_rounding,
            unsqueeze_input_to=5,  # b + c + xyz
        )

        return result


@builder.register(
    "VolumetricIndexChunker",
    cast_to_vec3d=["resolution"],
    cast_to_intvec3d=["chunk_size", "stride"],
)
@typechecked
@attrs.mutable
class VolumetricIndexChunker(IndexChunker[VolumetricIndex]):
    chunk_size: IntVec3D
    stride: Optional[IntVec3D] = None
    resolution: Optional[Vec3D] = None

    def __call__(
        self, idx: VolumetricIndex
    ) -> Iterable[VolumetricIndex]:  # pragma: no cover # delegation, no cond
        if self.resolution is None:
            chunk_resolution = idx.resolution
        else:
            chunk_resolution = self.resolution

        if self.stride is None:
            stride = self.chunk_size
        else:
            stride = self.stride

        bcube_strider = BcubeStrider(
            bcube=idx.bcube,
            resolution=chunk_resolution,
            chunk_size=self.chunk_size,
            stride=stride,
        )
        bcube_chunks = bcube_strider.get_all_chunk_bcubes()
        result = [
            VolumetricIndex(
                resolution=idx.resolution,
                bcube=bcube_chunk,
            )
            for bcube_chunk in bcube_chunks
        ]
        return result
