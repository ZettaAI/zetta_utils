# pylint: disable=missing-docstring, no-else-raise
from __future__ import annotations

from math import ceil, floor
from typing import List, Literal, Optional, Tuple

import attrs
from typeguard import typechecked

from zetta_utils import builder, log
from zetta_utils.typing import IntVec3D, Vec3D

from .bcube import BoundingCube

logger = log.get_logger("zetta_utils")


@builder.register(
    "BcubeStrider",
    cast_to_vec3d=["resolution", "stride_start"],
    cast_to_intvec3d=["chunk_size", "stride"],
)
@typechecked
@attrs.frozen
class BcubeStrider:
    """Strides over the bounding cube to produce a list of bounding cubes.
    Allows random indexing of the chunks without keeping the full chunk set in memory.

    :param bcube: Input bounding cube.
    :param resolution: Resolution at which ``chunk_size`` and ``stride`` are given.
    :param chunk_size: Size of an individual chunk.
    :param max_superchunk_size: Upper bound for the superchunks to consolidate the
        individual chunks to. Defaults to ``chunk_size``.
    :param stride: Distance between neighboring chunks along each dimension.
    :param stride_start_offset: Where the stride should start in unit (not voxel),
        modulo ``chunk_size``.
    :param mode: The modes that can be chosen for the behaviour.
        `shrink` is the default behaviour, which will round the bcube down to
        be aligned with the ``stride_start_offset`` (or just the start of the bcube
        if ``stride_start_offset`` is not set or ``stride`` != ``chunk_size``).
        `expand` is similar to `shrink` except it expands the bcube.
        `exact` will give full cubes aligned with the ``stride_start_offset`` and the
        ``chunk_size``, as well as the partial cubes at the edges.
    """

    bcube: BoundingCube
    resolution: Vec3D
    chunk_size: IntVec3D
    stride: IntVec3D
    max_superchunk_size: Optional[IntVec3D] = None
    stride_start_offset: Optional[IntVec3D] = None
    chunk_size_in_unit: Vec3D = attrs.field(init=False)
    stride_in_unit: Vec3D = attrs.field(init=False)
    bcube_snapped: BoundingCube = attrs.field(init=False)
    step_limits: Tuple[int, int, int] = attrs.field(init=False)
    step_start_partial: Tuple[bool, bool, bool] = attrs.field(init=False)
    step_end_partial: Tuple[bool, bool, bool] = attrs.field(init=False)
    mode: Optional[Literal["shrink", "expand", "exact"]] = "shrink"

    def __attrs_post_init__(self):
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
            if self.mode not in ("expand", "shrink"):
                raise NotImplementedError(
                    "superchunking is only supported when the"
                    " `chunk_mode` is set to `expand` or `shrink`"
                )
            if not self.max_superchunk_size >= self.chunk_size:
                raise ValueError("`max_superchunk_size` must be at least as large as `chunk_size`")
            if self.chunk_size != self.stride:
                raise NotImplementedError(
                    "superchunking is only supported when the `chunk_size` and `stride` are equal"
                )
            superchunk_size = self.chunk_size * (self.max_superchunk_size // self.chunk_size)
            object.__setattr__(self, "chunk_size", superchunk_size)
            object.__setattr__(self, "stride", superchunk_size)
            object.__setattr__(self, "bcube", self.bcube_snapped)
            object.__setattr__(self, "stride_start_offset", None)
            object.__setattr__(self, "mode", "exact")
            object.__setattr__(self, "max_superchunk_size", None)
            self.__attrs_post_init__()

    def _attrs_post_init_exact(self) -> None:
        if self.chunk_size != self.stride:
            raise NotImplementedError(
                "`exact` mode is only supported when the `chunk_size` and `stride` are equal"
            )
        if self.stride_start_offset is None:
            stride_start_offset_in_unit = self.bcube.start
        else:
            stride_start_offset_in_unit = self.stride_start_offset * self.resolution
        bcube_snapped = self.bcube.snapped(
            grid_offset=stride_start_offset_in_unit,
            grid_size=self.chunk_size_in_unit,
            mode="shrink",
        )
        bcube_snapped_size_in_unit = bcube_snapped.shape
        step_limits_snapped = Vec3D(
            *(
                (b - s) / st + 1
                for b, s, st in zip(
                    bcube_snapped_size_in_unit,
                    self.chunk_size_in_unit,
                    self.stride_in_unit,
                )
            )
        )
        step_limits = IntVec3D(*(floor(e) for e in step_limits_snapped))
        bcube_start_diff = bcube_snapped.start - self.bcube.start
        bcube_end_diff = self.bcube.end - bcube_snapped.end
        step_start_partial = tuple(e > 0 for e in bcube_start_diff)
        step_end_partial = tuple(e > 0 for e in bcube_end_diff)
        step_limits += IntVec3D(*(int(e) for e in step_start_partial))
        step_limits += IntVec3D(*(int(e) for e in step_end_partial))
        logger.info(
            f"Exact bcube requested; out of {self.bcube.bounds},"
            f" full cubes are in {bcube_snapped.bounds}, given offset"
            f" {stride_start_offset_in_unit}{self.bcube.unit} with chunk size"
            f" {self.chunk_size_in_unit}{self.bcube.unit}."
        )
        # Use `__setattr__` to keep the object frozen.
        object.__setattr__(self, "step_limits", step_limits)
        object.__setattr__(self, "bcube_snapped", bcube_snapped)
        object.__setattr__(self, "step_start_partial", step_start_partial)
        object.__setattr__(self, "step_end_partial", step_end_partial)

    def _attrs_post_init_nonexact(self) -> None:
        step_start_partial = [False, False, False]
        step_end_partial = [False, False, False]
        if self.chunk_size != self.stride:
            if self.stride_start_offset is not None:
                raise NotImplementedError(
                    "`stride_start_offset` is only supported when the"
                    " `chunk_size` and `stride` are equal"
                )
            bcube_snapped = self.bcube
        else:
            if self.stride_start_offset is not None:
                stride_start_offset_in_unit = self.stride_start_offset * self.resolution
            else:
                stride_start_offset_in_unit = self.bcube.start * self.resolution
            bcube_snapped = self.bcube.snapped(
                grid_offset=stride_start_offset_in_unit,
                grid_size=self.chunk_size_in_unit,
                mode=self.mode,  # type: ignore #mypy doesn't realise that mode has been checked
            )

        bcube_snapped_size_in_unit = bcube_snapped.end - bcube_snapped.start
        step_limits_snapped = Vec3D(
            *(
                (b - s) / st + 1
                for b, s, st in zip(
                    bcube_snapped_size_in_unit,
                    self.chunk_size_in_unit,
                    self.stride_in_unit,
                )
            )
        )
        bcube_size_in_unit = self.bcube.end - self.bcube.start
        step_limits_raw = Vec3D(
            *(
                (b - s) / st + 1
                for b, s, st in zip(
                    bcube_size_in_unit,
                    self.chunk_size_in_unit,
                    self.stride_in_unit,
                )
            )
        )
        if self.mode == "shrink":
            step_limits = IntVec3D(*(floor(e) for e in step_limits_snapped))
            if step_limits_raw != step_limits:
                rounded_bcube_bounds = tuple(
                    (
                        bcube_snapped.bounds[i][0],
                        (
                            bcube_snapped.bounds[i][0]
                            + self.chunk_size_in_unit[i]
                            + (step_limits[i] - 1) * self.stride_in_unit[i]
                        ),
                    )
                    for i in range(3)
                )
                logger.info(
                    f"Rounding down bcube bounds from {self.bcube.bounds} to"
                    f" {rounded_bcube_bounds} to divide evenly by stride"
                    f" {self.stride_in_unit}{self.bcube.unit} with chunk size"
                    f" {self.chunk_size_in_unit}{self.bcube.unit}."
                )
        if self.mode == "expand":
            step_limits = IntVec3D(*(ceil(e) for e in step_limits_snapped))
            if step_limits_raw != step_limits:
                rounded_bcube_bounds = tuple(
                    (
                        bcube_snapped.bounds[i][0],
                        (
                            bcube_snapped.bounds[i][0]
                            + self.chunk_size_in_unit[i]
                            + step_limits[i] * self.stride_in_unit[i]
                        ),
                    )
                    for i in range(3)
                )
                logger.info(
                    f"Rounding up bcube bounds from {self.bcube.bounds} to"
                    f" {rounded_bcube_bounds} to divide evenly by stride"
                    f" {self.stride_in_unit}{self.bcube.unit} with chunk size"
                    f" {self.chunk_size_in_unit}{self.bcube.unit}."
                )
        # Use `__setattr__` to keep the object frozen.
        object.__setattr__(self, "step_limits", step_limits)
        object.__setattr__(self, "bcube_snapped", bcube_snapped)
        object.__setattr__(self, "step_start_partial", step_start_partial)
        object.__setattr__(self, "step_end_partial", step_end_partial)

    @property
    def num_chunks(self) -> int:
        """Total number of chunks."""
        result = self.step_limits[0] * self.step_limits[1] * self.step_limits[2]
        return result

    def get_all_chunk_bcubes(self) -> List[BoundingCube]:
        """Get all of the chunks."""
        result = [self.get_nth_chunk_bcube(i) for i in range(self.num_chunks)]  # TODO: generator?
        return result

    def get_nth_chunk_bcube(self, n: int) -> BoundingCube:
        """Get nth chunk bcube, in order.

        :param n: Integer chunk index.
        :return: Volumetric index for the chunk, including
            ``self.desired_resolution`` and the slice representation of the region
            at ``self.index_resolution``.

        """
        steps_along_dim = IntVec3D(
            n % self.step_limits[0],
            (n // self.step_limits[0]) % self.step_limits[1],
            (n // (self.step_limits[0] * self.step_limits[1])) % self.step_limits[2],
        )
        if self.mode in ("shrink", "expand"):
            chunk_origin_in_unit = [
                self.bcube_snapped.bounds[i][0] + self.stride_in_unit[i] * steps_along_dim[i]
                for i in range(3)
            ]
            chunk_end_in_unit = [
                origin + size
                for origin, size in zip(chunk_origin_in_unit, self.chunk_size_in_unit)
            ]
        else:
            chunk_origin_in_unit = [
                self.bcube_snapped.bounds[i][0]
                + self.stride_in_unit[i] * (steps_along_dim[i] - int(self.step_start_partial[i]))
                for i in range(3)
            ]
            chunk_end_in_unit = [
                origin + size
                for origin, size in zip(chunk_origin_in_unit, self.chunk_size_in_unit)
            ]
            for i in range(3):
                if steps_along_dim[i] == 0 and self.step_start_partial[i]:
                    chunk_origin_in_unit[i] = self.bcube.start[i]
                    chunk_end_in_unit[i] = self.bcube_snapped.start[i]
                if steps_along_dim[i] == self.step_limits[i] - 1 and self.step_end_partial[i]:
                    chunk_origin_in_unit[i] = self.bcube_snapped.end[i]
                    chunk_end_in_unit[i] = self.bcube.end[i]
        slices = (
            slice(chunk_origin_in_unit[0], chunk_end_in_unit[0]),
            slice(chunk_origin_in_unit[1], chunk_end_in_unit[1]),
            slice(chunk_origin_in_unit[2], chunk_end_in_unit[2]),
        )

        result = BoundingCube.from_slices(slices)
        return result
