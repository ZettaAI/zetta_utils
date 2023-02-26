# pylint: disable=missing-docstring, no-else-raise
from __future__ import annotations

from math import ceil, floor
from typing import List, Literal, Optional, Tuple

import attrs
from typeguard import typechecked

from zetta_utils import builder, log

from . import Vec3D
from .bbox import BBox3D

logger = log.get_logger("zetta_utils")


@builder.register("BBoxStrider")
@attrs.frozen
@typechecked
class BBoxStrider:
    """Strides over the bounding cube to produce a list of bounding cubes.
    Allows random indexing of the chunks without keeping the full chunk set in memory.

    :param bbox: Input bounding cube.
    :param resolution: Resolution at which ``chunk_size`` and ``stride`` are given.
    :param chunk_size: Size of an individual chunk.
    :param max_superchunk_size: Upper bound for the superchunks to consolidate the
        individual chunks to. Defaults to ``chunk_size``.
    :param stride: Distance between neighboring chunks along each dimension.
    :param stride_start_offset: Where the stride should start in unit (not voxel),
        modulo ``chunk_size``.
    :param mode: The modes that can be chosen for the behaviour.
        `shrink` will round the bbox down to
        be aligned with the ``stride_start_offset`` (or just the start of the bbox
        if ``stride_start_offset`` is not set or ``stride`` != ``chunk_size``).
        `expand` is similar to `shrink` except it expands the bbox.
        `exact` will give full cubes aligned with the ``stride_start_offset`` and the
        ``chunk_size``, as well as the partial cubes at the edges.
    """

    bbox: BBox3D
    resolution: Vec3D
    chunk_size: Vec3D[int]
    stride: Vec3D[int]
    max_superchunk_size: Optional[Vec3D[int]] = None
    stride_start_offset_in_unit: Optional[Vec3D[int]] = None
    chunk_size_in_unit: Vec3D = attrs.field(init=False)
    stride_in_unit: Vec3D = attrs.field(init=False)
    bbox_snapped: BBox3D = attrs.field(init=False)
    step_limits: Tuple[int, int, int] = attrs.field(init=False)
    step_start_partial: Tuple[bool, bool, bool] = attrs.field(init=False)
    step_end_partial: Tuple[bool, bool, bool] = attrs.field(init=False)
    mode: Optional[Literal["shrink", "expand", "exact"]] = "expand"

    def __attrs_post_init__(self) -> None:
        stride_in_unit = self.stride * self.resolution
        chunk_size_in_unit = self.chunk_size * self.resolution
        object.__setattr__(self, "chunk_size_in_unit", chunk_size_in_unit)
        object.__setattr__(self, "stride_in_unit", stride_in_unit)

        if self.mode in ("shrink", "expand"):
            self._attrs_post_init_nonexact()
        if self.mode == "exact":
            self._attrs_post_init_exact()

        # recursively call __attrs_post_init__ if superchunking is set
        if self.max_superchunk_size is None:
            return
        else:
            object.__setattr__(self, "stride_start_offset_in_unit", self.bbox_snapped.start)
            if self.mode in ("expand", "shrink"):
                object.__setattr__(self, "bbox", self.bbox_snapped)
            if not self.max_superchunk_size >= self.chunk_size:
                raise ValueError("`max_superchunk_size` must be at least as large as `chunk_size`")
            if self.chunk_size != self.stride:
                raise NotImplementedError(
                    "superchunking is only supported when the `chunk_size` and `stride` are equal"
                )
            superchunk_size = self.chunk_size * (self.max_superchunk_size // self.chunk_size)
            object.__setattr__(self, "chunk_size", superchunk_size)
            object.__setattr__(self, "stride", superchunk_size)
            object.__setattr__(self, "mode", "exact")
            object.__setattr__(self, "max_superchunk_size", None)
            self.__attrs_post_init__()

    def _attrs_post_init_exact(self) -> None:
        if self.chunk_size != self.stride:
            raise NotImplementedError(
                "`exact` mode is only supported when the `chunk_size` and `stride` are equal"
            )
        if self.stride_start_offset_in_unit is None:
            stride_start_offset_in_unit = self.bbox.start
        else:
            stride_start_offset_in_unit = self.stride_start_offset_in_unit
        bbox_snapped = self.bbox.snapped(
            grid_offset=stride_start_offset_in_unit,
            grid_size=self.stride_in_unit,
            mode="shrink",
        )
        bbox_snapped_size_in_unit = bbox_snapped.shape
        step_limits_snapped = Vec3D[float](
            *(
                (b - s) / st + 1
                for b, s, st in zip(
                    bbox_snapped_size_in_unit,
                    self.chunk_size_in_unit,
                    self.stride_in_unit,
                )
            )
        )
        step_limits = Vec3D[int](*(floor(e) for e in step_limits_snapped))
        bbox_start_diff = bbox_snapped.start - self.bbox.start
        bbox_end_diff = self.bbox.end - bbox_snapped.end
        step_start_partial = tuple(e > 0 for e in bbox_start_diff)
        step_end_partial = tuple(e > 0 for e in bbox_end_diff)
        step_limits += Vec3D[int](*(int(e) for e in step_start_partial))
        step_limits += Vec3D[int](*(int(e) for e in step_end_partial))
        logger.info(
            f"Exact bbox requested; out of {self.bbox.bounds},"
            f" full cubes are in {bbox_snapped.bounds}, given offset"
            f" {stride_start_offset_in_unit}{self.bbox.unit} with chunk size"
            f" {self.chunk_size_in_unit}{self.bbox.unit}."
        )
        # Use `__setattr__` to keep the object frozen.
        object.__setattr__(self, "step_limits", step_limits)
        object.__setattr__(self, "bbox_snapped", bbox_snapped)
        object.__setattr__(self, "step_start_partial", step_start_partial)
        object.__setattr__(self, "step_end_partial", step_end_partial)

    def _attrs_post_init_nonexact(self) -> None:
        step_start_partial = [False, False, False]
        step_end_partial = [False, False, False]
        if self.stride_start_offset_in_unit is not None:
            # align stride_start_offset to just larger than the start of the bbox
            stride_start_offset_in_unit = Vec3D(*self.stride_start_offset_in_unit)
            stride_start_offset_in_unit += (
                (self.bbox.start - stride_start_offset_in_unit)
                // self.stride_in_unit
                * self.stride_in_unit
            )
        else:
            stride_start_offset_in_unit = self.bbox.start

        bbox_snapped = (
            self.bbox.translated_end(-self.chunk_size_in_unit, resolution=Vec3D(1, 1, 1))
            .snapped(
                grid_offset=stride_start_offset_in_unit,
                grid_size=self.stride_in_unit,
                mode=self.mode,  # type: ignore #mypy doesn't realise that mode has been checked
            )
            .translated_end(self.chunk_size_in_unit, resolution=Vec3D(1, 1, 1))
        )

        bbox_snapped_size_in_unit = bbox_snapped.end - bbox_snapped.start
        step_limits_snapped = Vec3D[float](
            *(
                (b - s) / st + 1
                for b, s, st in zip(
                    bbox_snapped_size_in_unit,
                    self.chunk_size_in_unit,
                    self.stride_in_unit,
                )
            )
        )
        bbox_size_in_unit = self.bbox.end - self.bbox.start
        step_limits_raw: Vec3D[float] = Vec3D[float](
            *(
                (b - s) / st + 1
                for b, s, st in zip(
                    bbox_size_in_unit,
                    self.chunk_size_in_unit,
                    self.stride_in_unit,
                )
            )
        )
        if self.mode == "shrink":
            step_limits = Vec3D[int](*(floor(e) for e in step_limits_snapped))
            if step_limits_raw != step_limits:
                rounded_bbox_bounds = tuple(
                    (
                        bbox_snapped.bounds[i][0],
                        (
                            bbox_snapped.bounds[i][0]
                            + self.chunk_size_in_unit[i]
                            + (step_limits[i] - 1) * self.stride_in_unit[i]
                        ),
                    )
                    for i in range(3)
                )
                logger.debug(
                    f"Rounding down bbox bounds from {self.bbox.bounds} to"
                    f" {rounded_bbox_bounds} to divide evenly by stride"
                    f" {self.stride_in_unit}{self.bbox.unit} with chunk size"
                    f" {self.chunk_size_in_unit}{self.bbox.unit}."
                )
        if self.mode == "expand":
            step_limits = Vec3D[int](*(ceil(e) for e in step_limits_snapped))
            if step_limits_raw != step_limits:
                rounded_bbox_bounds = tuple(
                    (
                        bbox_snapped.bounds[i][0],
                        (
                            bbox_snapped.bounds[i][0]
                            + self.chunk_size_in_unit[i]
                            + step_limits[i] * self.stride_in_unit[i]
                        ),
                    )
                    for i in range(3)
                )
                logger.debug(
                    f"Rounding up bbox bounds from {self.bbox.bounds} to"
                    f" {rounded_bbox_bounds} to divide evenly by stride"
                    f" {self.stride_in_unit}{self.bbox.unit} with chunk size"
                    f" {self.chunk_size_in_unit}{self.bbox.unit}."
                )
        # Use `__setattr__` to keep the object frozen.
        object.__setattr__(self, "step_limits", step_limits)
        object.__setattr__(self, "bbox_snapped", bbox_snapped)
        object.__setattr__(self, "step_start_partial", step_start_partial)
        object.__setattr__(self, "step_end_partial", step_end_partial)

    @property
    def num_chunks(self) -> int:
        """Total number of chunks."""
        result = self.step_limits[0] * self.step_limits[1] * self.step_limits[2]
        return result

    def get_all_chunk_bboxes(self) -> List[BBox3D]:
        """Get all of the chunks."""
        result = [self.get_nth_chunk_bbox(i) for i in range(self.num_chunks)]  # TODO: generator?
        return result

    def get_nth_chunk_bbox(self, n: int) -> BBox3D:
        """Get nth chunk bbox, in order.

        :param n: Integer chunk index.
        :return: Volumetric index for the chunk, including
            ``self.desired_resolution`` and the slice representation of the region
            at ``self.index_resolution``.

        """
        steps_along_dim = Vec3D[int](
            n % self.step_limits[0],
            (n // self.step_limits[0]) % self.step_limits[1],
            (n // (self.step_limits[0] * self.step_limits[1])) % self.step_limits[2],
        )
        if self.mode in ("shrink", "expand"):
            chunk_origin_in_unit = [
                self.bbox_snapped.bounds[i][0] + self.stride_in_unit[i] * steps_along_dim[i]
                for i in range(3)
            ]
            chunk_end_in_unit = [
                origin + size
                for origin, size in zip(chunk_origin_in_unit, self.chunk_size_in_unit)
            ]
        else:
            chunk_origin_in_unit = [
                self.bbox_snapped.bounds[i][0]
                + self.stride_in_unit[i] * (steps_along_dim[i] - int(self.step_start_partial[i]))
                for i in range(3)
            ]
            chunk_end_in_unit = [
                origin + size
                for origin, size in zip(chunk_origin_in_unit, self.chunk_size_in_unit)
            ]
            for i in range(3):
                if steps_along_dim[i] == 0 and self.step_start_partial[i]:
                    chunk_origin_in_unit[i] = self.bbox.start[i]
                    chunk_end_in_unit[i] = self.bbox_snapped.start[i]
                if steps_along_dim[i] == self.step_limits[i] - 1 and self.step_end_partial[i]:
                    chunk_origin_in_unit[i] = self.bbox_snapped.end[i]
                    chunk_end_in_unit[i] = self.bbox.end[i]
        slices = (
            slice(chunk_origin_in_unit[0], chunk_end_in_unit[0]),
            slice(chunk_origin_in_unit[1], chunk_end_in_unit[1]),
            slice(chunk_origin_in_unit[2], chunk_end_in_unit[2]),
        )

        result = BBox3D.from_slices(slices)
        return result
