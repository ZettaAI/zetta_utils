from typing import Iterable, List, Literal, Optional, Tuple

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder, log, tensor_ops
from zetta_utils.bcube import BcubeStrider
from zetta_utils.typing import IntVec3D, Vec3D

from .. import DataWithIndexProcessor, IndexChunker
from . import VolumetricIndex

logger = log.get_logger("zetta_utils")


@typechecked
def translate_volumetric_index(
    idx: VolumetricIndex, offset: Vec3D, resolution: Vec3D
):  # pragma: no cover # under 3 statements, no conditionals
    bcube = idx.bcube.translated(offset, resolution)
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


@builder.register("VolumetricDataInterpolator")
@typechecked
@attrs.mutable
class VolumetricDataInterpolator(DataWithIndexProcessor):
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
    max_superchunk_size: Optional[IntVec3D] = None
    stride: Optional[IntVec3D] = None
    resolution: Optional[Vec3D] = None
    offset: IntVec3D = IntVec3D(0, 0, 0)

    def __call__(
        self,
        idx: VolumetricIndex,
        stride_start_offset: Optional[IntVec3D] = None,
        mode: Literal["shrink", "expand", "exact"] = "shrink",
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
            bcube=idx.bcube.translated_start(offset=self.offset, resolution=chunk_resolution),
            resolution=chunk_resolution,
            chunk_size=self.chunk_size,
            max_superchunk_size=self.max_superchunk_size,
            stride=stride,
            stride_start_offset=stride_start_offset,
            mode=mode,
        )
        if self.max_superchunk_size is not None:
            logger.info(f"Superchunk size: {bcube_strider.chunk_size}")
        bcube_chunks = bcube_strider.get_all_chunk_bcubes()
        result = [
            VolumetricIndex(
                resolution=idx.resolution,
                bcube=bcube_chunk,
            )
            for bcube_chunk in bcube_chunks
        ]
        return result

    def split_into_nonoverlapping_chunkers(
        self, pad: IntVec3D = IntVec3D(0, 0, 0)
    ) -> List[Tuple[IndexChunker[VolumetricIndex], Tuple[int, int, int]]]:  # pragma: no cover
        try:
            assert (self.stride is None) or (self.stride == self.chunk_size)
            assert self.offset == IntVec3D(0, 0, 0)
        except Exception as e:
            raise NotImplementedError(
                "can only split chunkers that have stride equal to chunk size and no offset"
            ) from e
        try:
            for s, p in zip(self.chunk_size, pad):  # pylint: disable=invalid-name
                assert s >= 2 * p
        except Exception as e:
            raise ValueError("can only pad by less than half of the chunk size") from e
        offset_x = IntVec3D(self.chunk_size[0], 0, 0)
        offset_y = IntVec3D(0, self.chunk_size[1], 0)
        offset_z = IntVec3D(0, 0, self.chunk_size[2])
        chunk_size = self.chunk_size + pad * 2
        stride = self.chunk_size * 2

        return [
            (
                VolumetricIndexChunker(
                    chunk_size=chunk_size,
                    stride=stride,
                    resolution=self.resolution,
                    offset=x * offset_x + y * offset_y + z * offset_z,
                ),
                (x, y, z),
            )
            for x in range(2)
            for y in range(2)
            for z in range(2)
        ]
