from __future__ import annotations

from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder, log, tensor_ops
from zetta_utils.geometry import BBoxStrider, Vec3D

from .. import IndexChunker, JointIndexDataProcessor
from . import VolumetricIndex

logger = log.get_logger("zetta_utils")


@typechecked
def translate_volumetric_index(
    idx: VolumetricIndex, offset: Sequence[float], resolution: Sequence[float]
):  # pragma: no cover # under 3 statements, no conditionals
    bbox = idx.bbox.translated(offset, resolution)
    result = VolumetricIndex(
        bbox=bbox,
        resolution=Vec3D(*idx.resolution),
    )
    return result


@builder.register("VolumetricIndexTranslator")
@typechecked
@attrs.mutable
class VolumetricIndexTranslator:  # pragma: no cover # under 3 statements, no conditionals
    offset: Sequence[float]
    resolution: Sequence[float]

    def __call__(self, idx: VolumetricIndex) -> VolumetricIndex:
        result = translate_volumetric_index(
            idx=idx,
            offset=self.offset,
            resolution=self.resolution,
        )
        return result


@builder.register("DataResolutionInterpolator")
@typechecked
@attrs.mutable
class DataResolutionInterpolator(JointIndexDataProcessor):
    data_resolution: Sequence[float]
    interpolation_mode: tensor_ops.InterpolationMode
    allow_slice_rounding: bool = False
    prepared_scale_factor: Vec3D | None = attrs.field(init=False, default=None)

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        if mode == "read":
            self.prepared_scale_factor = Vec3D(*self.data_resolution) / idx.resolution
        else:
            assert mode == "write"
            self.prepared_scale_factor = idx.resolution / Vec3D(*self.data_resolution)

        result = VolumetricIndex(
            bbox=idx.bbox,
            resolution=Vec3D(*self.data_resolution),
            allow_slice_rounding=idx.allow_slice_rounding,
        )
        return result

    def process_data(self, data: torch.Tensor, mode: Literal["read", "write"]) -> torch.Tensor:
        assert self.prepared_scale_factor is not None

        result = tensor_ops.interpolate(
            data=data,
            scale_factor=self.prepared_scale_factor,
            mode=self.interpolation_mode,
            allow_slice_rounding=self.allow_slice_rounding,
            unsqueeze_input_to=5,  # b + c + xyz
        )
        self.prepared_scale_factor = None
        return result


@builder.register("InvertProcessor")
@typechecked
@attrs.mutable
class InvertProcessor(JointIndexDataProcessor):  # pragma: no cover
    invert: bool

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        return idx

    def process_data(self, data: torch.Tensor, mode: Literal["read", "write"]) -> torch.Tensor:
        if self.invert:
            if not data.dtype == torch.uint8:
                raise NotImplementedError("InvertProcessor is only supported for UInt8 layers.")
            result = torch.bitwise_not(data) + 2
        else:
            result = data
        return result


@attrs.mutable
# TODO: Refacter the offset part into a separate subclass
class VolumetricIndexChunker(IndexChunker[VolumetricIndex]):
    chunk_size: Vec3D[int]
    max_superchunk_size: Optional[Vec3D[int]] = None
    stride: Optional[Vec3D[int]] = None
    resolution: Optional[Vec3D] = None
    offset: Vec3D[int] = Vec3D[int](0, 0, 0)

    def __call__(
        self,
        idx: VolumetricIndex,
        stride_start_offset_in_unit: Optional[Vec3D[int]] = None,
        mode: Literal["shrink", "expand", "exact"] = "expand",
    ) -> Iterable[VolumetricIndex]:  # pragma: no cover # delegation, no cond
        if self.resolution is None:
            chunk_resolution = idx.resolution
        else:
            chunk_resolution = self.resolution

        if self.stride is None:
            stride = self.chunk_size
        else:
            stride = self.stride

        if stride_start_offset_in_unit is None:
            stride_start_offset_to_use = (idx.bbox.start + self.offset).int()
        else:
            stride_start_offset_to_use = (stride_start_offset_in_unit + self.offset).int()

        bbox_strider = BBoxStrider(
            bbox=idx.bbox.translated_start(offset=self.offset, resolution=chunk_resolution),
            resolution=chunk_resolution,
            chunk_size=self.chunk_size,
            max_superchunk_size=self.max_superchunk_size,
            stride=stride,
            stride_start_offset_in_unit=stride_start_offset_to_use,
            mode=mode,
        )
        if self.max_superchunk_size is not None:
            logger.info(f"Superchunk size: {bbox_strider.chunk_size}")
        bbox_chunks = bbox_strider.get_all_chunk_bboxes()
        result = [
            VolumetricIndex(
                resolution=idx.resolution,
                bbox=bbox_chunk,
            )
            for bbox_chunk in bbox_chunks
        ]
        return result

    def split_into_nonoverlapping_chunkers(
        self, pad: Vec3D[int] = Vec3D[int](0, 0, 0)
    ) -> List[Tuple[VolumetricIndexChunker, Tuple[int, int, int]]]:  # pragma: no cover
        try:
            assert (self.stride is None) or (self.stride == self.chunk_size)
            assert self.offset == Vec3D[int](0, 0, 0)
        except Exception as e:
            raise NotImplementedError(
                "can only split chunkers that have stride equal to chunk size and no offset"
            ) from e
        try:
            for s, p in zip(self.chunk_size, pad):  # pylint: disable=invalid-name
                assert s >= 2 * p
        except Exception as e:
            raise ValueError("can only pad by less than half of the chunk size") from e
        offset_x = Vec3D[int](self.chunk_size[0], 0, 0)
        offset_y = Vec3D[int](0, self.chunk_size[1], 0)
        offset_z = Vec3D[int](0, 0, self.chunk_size[2])
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
