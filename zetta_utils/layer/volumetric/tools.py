from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Tuple

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder, log, tensor_ops
from zetta_utils.geometry import BBoxStrider, IntVec3D, Vec3D
from zetta_utils.geometry.bbox import Slices3D

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


@builder.register("VolumetricIndexOverrider")
@typechecked
@attrs.mutable
class VolumetricIndexOverrider:
    override_offset: Optional[Sequence[int | None]] = None
    override_size: Optional[Sequence[int | None]] = None
    override_resolution: Optional[Sequence[float | None]] = None

    def __call__(self, idx: VolumetricIndex) -> VolumetricIndex:
        if self.override_offset is None:
            self.override_offset = idx.start
        if self.override_size is None:
            self.override_size = idx.shape
        if self.override_resolution is None:
            self.override_resolution = idx.resolution

        start = IntVec3D(
            *[x if x is not None else y for x, y in zip(self.override_offset, idx.start)]
        )
        size = IntVec3D(
            *[x if x is not None else y for x, y in zip(self.override_size, idx.shape)]
        )
        resolution = Vec3D[float](
            *[x if x is not None else y for x, y in zip(self.override_resolution, idx.resolution)]
        )
        stop = start + size

        return VolumetricIndex.from_coords(
            start_coord=start.vec,
            end_coord=stop.vec,
            resolution=resolution,
        )


@builder.register("VolumetricIndexPadder")
@typechecked
@attrs.mutable
class VolumetricIndexPadder:  # pragma: no cover
    pad: Sequence[int]

    def __call__(self, idx: VolumetricIndex) -> VolumetricIndex:
        result = idx.padded(pad=self.pad)
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


@builder.register("ROIMaskProcessor")
@typechecked
@attrs.mutable
class ROIMaskProcessor(JointIndexDataProcessor):
    """
    This processor dynamically produces an ROI (Region Of Interest) mask for a
    training patch in volumetric data processing. An ROI mask fills 1 for voxels
    within the ROI and 0 for voxels outside the ROI. The ROI is specified using
    ``start_coord``, ``end_coord``, and ``resolution``. ROI masks are produced
    only for those layers specified in `targets`.

    The naming convention for the generated ROI mask layers follows a preset pattern:
    the name of the target layer is appended with the suffix ``_mask``. For instance,
    if the target layer is ``layer1``, the corresponding mask will be named ``layer1_mask``.
    If a layer with the intended mask name already exists in the data, the processor
    skips the auto-generation of the ROI mask for that specific layer to avoid
    redundancy and potential data overwrite.

    :param start_coord: The starting coordinates of the ROI in the data, represented
        as a sequence of integers.
    :param end_coord: The ending coordinates of the ROI, aligning with ``start_coord``
        to define the ROI region.
    :param resolution: The resolution of the ROI, given as a sequence of floats.
        This defines the size of each voxel within the ROI.
    :param targets: A list of strings specifying the target layers in the data for which
        the ROI masks will be generated.
    """

    start_coord: Sequence[int]
    end_coord: Sequence[int]
    resolution: Sequence[float]
    targets: list[str]

    roi: VolumetricIndex = attrs.field(init=False)
    prepared_subidx: Slices3D | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        roi = VolumetricIndex.from_coords(
            start_coord=self.start_coord,
            end_coord=self.end_coord,
            resolution=Vec3D(*self.resolution),
        )
        object.__setattr__(self, "roi", roi)

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        intersection = idx.intersection(self.roi)
        self.prepared_subidx = intersection.translated(-idx.start).to_slices()
        return idx

    def process_data(
        self, data: dict[str, torch.Tensor], mode: Literal["read", "write"]
    ) -> dict[str, torch.Tensor]:
        assert self.prepared_subidx is not None

        for target in self.targets:
            assert target in data
            if target + "_mask" in data:
                continue
            roi_mask = torch.zeros_like(data[target])
            extra_dims = roi_mask.ndim - len(self.prepared_subidx)
            slices = [slice(0, None) for _ in range(extra_dims)]
            slices += list(self.prepared_subidx)
            roi_mask[tuple(slices)] = 1
            data[target + "_mask"] = roi_mask

        self.prepared_subidx = None
        return data


@attrs.mutable
# TODO: Refacter the offset part into a separate subclass
class VolumetricIndexChunker(IndexChunker[VolumetricIndex]):
    """
    Chunker for dividing a `VolumetricIndex` into smaller chunks.

    :param chunk_size: Size of each chunk in three dimensions.
    :param max_superchunk_size: Maximum size of superchunks, if applicable.
    :param stride: Stride for chunking, if different from chunk_size.
    :param resolution: Resolution of the chunks; if `None`, resolution will
        be taken from the `VolumetricIndex`.
    :param offset: Offset for chunking; default is `(0, 0, 0)`.

    Methods:
        `__call__(idx, stride_start_offset_in_unit=None, mode="expand")`:
            Divides a `VolumetricIndex` into chunks based on specified parameters.

        `get_shape(idx, stride_start_offset_in_unit=None, mode="expand")`:
            Returns the shape of the division (i.e., how many chunks the volume
            would be divided into in x, y, and z) without actually creating them.

        `split_into_nonoverlapping_chunkers(pad=Vec3D[int](0, 0, 0))`:
            Returns 8 chunkers related to this one in the following way: each one
            has the same `chunk_size` and `resolution` as this one, but with
            `stride` and `offset` set so that the chunks they generate will not
            overlap.  This is used to for checkerboarding.  If you call each
            of these 8 chunkers on an index, you'll get the same chunks you
            would have gotten from calling this one, but divided into 8 non-
            overlapping sets.

    Note:
        This class divides a `VolumetricIndex` into chunks, according to `chunk_size`,
        `stride`, and `offset`.  It can also produce 8 related chunkers that would return
        the same chunks, but in 8 non-overlapping sets (for checkerboarding).
    """

    chunk_size: Vec3D[int]
    max_superchunk_size: Optional[Vec3D[int]] = None
    stride: Optional[Vec3D[int]] = None
    resolution: Optional[Vec3D] = None
    offset: Vec3D[int] = Vec3D[int](0, 0, 0)

    def __call__(
        self,
        idx: VolumetricIndex,
        stride_start_offset_in_unit: Optional[Vec3D] = None,
        mode: Literal["shrink", "expand", "exact"] = "expand",
        chunk_id_increment: int = 0,
    ) -> List[VolumetricIndex]:
        bbox_strider = self._get_bbox_strider(idx, stride_start_offset_in_unit, mode)
        if self.max_superchunk_size is not None:
            logger.info(f"Superchunk size: {bbox_strider.chunk_size}")  # pragma: no cover
        bbox_chunks = bbox_strider.get_all_chunk_bboxes()
        result = [
            VolumetricIndex(
                resolution=idx.resolution,
                bbox=bbox_chunk,
                chunk_id=idx.chunk_id + i * chunk_id_increment,
            )
            for i, bbox_chunk in enumerate(bbox_chunks)
        ]
        return result

    def get_shape(
        self,
        idx: VolumetricIndex,
        stride_start_offset_in_unit: Optional[Vec3D] = None,
        mode: Literal["shrink", "expand", "exact"] = "expand",
    ) -> Vec3D[int]:  # pragma: no cover
        return self._get_bbox_strider(idx, stride_start_offset_in_unit, mode).shape

    def _get_bbox_strider(
        self,
        idx: VolumetricIndex,
        stride_start_offset_in_unit: Optional[Vec3D] = None,
        mode: Literal["shrink", "expand", "exact"] = "expand",
    ) -> BBoxStrider:
        if self.resolution is None:
            chunk_resolution = idx.resolution
        else:
            chunk_resolution = self.resolution

        if stride_start_offset_in_unit is None:
            stride_start_offset_to_use = idx.bbox.start + self.offset * chunk_resolution
        else:
            stride_start_offset_to_use = (
                stride_start_offset_in_unit + self.offset * chunk_resolution
            )

        if self.stride is None:
            stride = self.chunk_size
        else:
            stride = self.stride

        bbox_strider = BBoxStrider(
            bbox=idx.bbox.translated_start(offset=self.offset, resolution=chunk_resolution),
            resolution=chunk_resolution,
            chunk_size=self.chunk_size,
            max_superchunk_size=self.max_superchunk_size,
            stride=stride,
            stride_start_offset_in_unit=stride_start_offset_to_use,
            mode=mode,
        )
        return bbox_strider

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
