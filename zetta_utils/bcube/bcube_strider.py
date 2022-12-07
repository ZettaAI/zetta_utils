# pylint: disable=missing-docstring, no-else-raise
from __future__ import annotations

from math import floor
from typing import List, Tuple

import attrs
from typeguard import typechecked

from zetta_utils import builder, log
from zetta_utils.typing import IntVec3D, Vec3D

from .bcube import BoundingCube

logger = log.get_logger("zetta_utils")


@builder.register(
    "BcubeStrider", cast_to_vec3d=["resolution"], cast_to_intvec3d=["chunk_size", "stride"]
)
@typechecked
@attrs.frozen
class BcubeStrider:
    """Strides over the bounding cube to produce a list of bounding cubes.
    Allows random indexing of the chunks without keeping the full chunk set in memory.

    :param bcube: Input bounding cube.
    :param resolution: Resoluiton at which ``chunk_size`` and ``stride`` are given.
    :param chunk_size: Size of an individual chunk.
    :param stride: Distance between neighboring chunks along each dimension.
    """

    bcube: BoundingCube
    resolution: Vec3D
    chunk_size: IntVec3D
    stride: IntVec3D
    chunk_size_in_unit: Vec3D = attrs.field(init=False)
    stride_in_unit: Vec3D = attrs.field(init=False)
    step_limits: Tuple[int, int, int] = attrs.field(init=False)

    def __attrs_post_init__(self):
        bcube_size_in_unit = tuple(
            self.bcube.bounds[i][1] - self.bcube.bounds[i][0] for i in range(3)
        )
        chunk_size_in_unit = tuple(s * r for s, r in zip(self.chunk_size, self.resolution))
        stride_in_unit = tuple(s * r for s, r in zip(self.stride, self.resolution))
        step_limits_raw = tuple(
            (b - s) / st + 1
            for b, s, st in zip(
                bcube_size_in_unit,
                chunk_size_in_unit,
                stride_in_unit,
            )
        )
        step_limits = tuple(floor(e) for e in step_limits_raw)

        if step_limits != step_limits_raw:
            rounded_bcube_bounds = tuple(
                (
                    self.bcube.bounds[i][0],
                    (
                        self.bcube.bounds[i][0]
                        + chunk_size_in_unit[i]
                        + (step_limits[i] - 1) * stride_in_unit[i]
                    ),
                )
                for i in range(3)
            )
            logger.warning(
                f"Rounding down bcube bounds from {self.bcube.bounds} to {rounded_bcube_bounds} "
                f"to divide evenly by stride {stride_in_unit}{self.bcube.unit} "
                f"with chunk size {chunk_size_in_unit}{self.bcube.unit}."
            )

        # Use `__setattr__` to keep the object frozen.
        object.__setattr__(self, "step_limits", step_limits)
        object.__setattr__(self, "chunk_size_in_unit", chunk_size_in_unit)
        object.__setattr__(self, "stride_in_unit", stride_in_unit)

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
        steps_along_dim = [
            n % self.step_limits[0],
            (n // self.step_limits[0]) % self.step_limits[1],
            (n // (self.step_limits[0] * self.step_limits[1])) % self.step_limits[2],
        ]
        chunk_origin_in_unit = [
            self.bcube.bounds[i][0] + self.stride_in_unit[i] * steps_along_dim[i] for i in range(3)
        ]
        chunk_end_in_unit = [
            origin + size for origin, size in zip(chunk_origin_in_unit, self.chunk_size_in_unit)
        ]
        slices = (
            slice(chunk_origin_in_unit[0], chunk_end_in_unit[0]),
            slice(chunk_origin_in_unit[1], chunk_end_in_unit[1]),
            slice(chunk_origin_in_unit[2], chunk_end_in_unit[2]),
        )

        result = BoundingCube.from_slices(slices)
        return result
