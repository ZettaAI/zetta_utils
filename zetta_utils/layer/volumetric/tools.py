from typing import Iterable, List, Literal, Optional, Tuple

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder, tensor_ops
from zetta_utils.bcube import BcubeStrider
from zetta_utils.typing import IntVec3D, Vec3D

from .. import DataProcessor, DataWithIndexProcessor, IndexChunker
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
    stride: Optional[IntVec3D] = None
    resolution: Optional[Vec3D] = None
    offset: IntVec3D = IntVec3D(0, 0, 0)

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
            bcube=idx.bcube.translate_start(offset=self.offset, resolution=chunk_resolution),
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

    def split_into_nonoverlapping_chunkers(
        self, pad: IntVec3D = IntVec3D(0, 0, 0)
    ) -> List[Tuple[IndexChunker[VolumetricIndex], Tuple[int, int, int]]]:  # pragma: no cover
        try:
            assert (self.stride is None) or (self.stride == self.chunk_size)
            assert self.offset == IntVec3D(0, 0, 0)
        except Exception as e:
            raise NotImplementedError(
                "can only split chunkers that have stride equal to chunk size" + " and no offset"
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
        stride = self.chunk_size * 3

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
            for x in range(3)
            for y in range(3)
            for z in range(3)
        ]


@builder.register("VolumetricDataBlendingWeighter", cast_to_intvec3d=["blend_pad"])
@typechecked
@attrs.mutable
class VolumetricDataBlendingWeighter(DataProcessor):  # pragma: no cover
    blend_pad: IntVec3D
    blend_mode: Literal["linear", "quadratic"] = "linear"

    def __call__(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        try:
            for s, p in zip(  # pylint: disable=invalid-name
                IntVec3D(*data.shape[-3:]), self.blend_pad
            ):
                assert s >= 2 * p
        except Exception as e:
            raise ValueError(
                f"received {tuple(data.shape[-3:])} data, expected at least {2*self.blend_pad}"
            ) from e
        mask = torch.ones_like(data, dtype=torch.float)
        x_pad = self.blend_pad[0]
        y_pad = self.blend_pad[1]
        z_pad = self.blend_pad[2]
        if self.blend_mode == "linear":
            for x in range(2 * x_pad):
                weight = x / (2 * x_pad)
                mask[:, x, :, :] *= weight
                mask[:, -x, :, :] *= weight
            for y in range(2 * y_pad):
                weight = y / (2 * y_pad)
                mask[:, :, y, :] *= weight
                mask[:, :, -y, :] *= weight
            for z in range(2 * z_pad):
                weight = z / (2 * z_pad)
                mask[:, :, :, z] *= weight
                mask[:, :, :, -z] *= weight
        elif self.blend_mode == "quadratic":
            for x in range(x_pad):
                weight = ((x / x_pad) ** 2) / 2
                mask[:, x, :, :] *= weight
                mask[:, -x, :, :] *= weight
                mask[:, 2 * x_pad - x, :, :] *= 1 - weight
                mask[:, -(2 * x_pad - x), :, :] *= 1 - weight
            for y in range(y_pad):
                weight = ((y / y_pad) ** 2) / 2
                mask[:, :, y, :] *= weight
                mask[:, :, -y, :] *= weight
                mask[:, :, 2 * y_pad - y, :] *= 1 - weight
                mask[:, :, -(2 * y_pad - y), :] *= 1 - weight
            for z in range(z_pad):
                weight = ((z / z_pad) ** 2) / 2
                mask[:, :, :, z] *= weight
                mask[:, :, :, -z] *= weight
                mask[:, :, :, 2 * z_pad - z] *= 1 - weight
                mask[:, :, :, -(2 * z_pad - z)] *= 1 - weight

        result = tensor_ops.common.multiply(data, mask)

        return result
