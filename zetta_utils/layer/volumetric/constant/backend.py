# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Literal, Union

import attrs
import numpy as np
from numpy import typing as npt

from zetta_utils.geometry import Vec3D

from .. import VolumetricBackend, VolumetricIndex


@attrs.mutable
class ConstantVolumetricBackend(VolumetricBackend):  # pylint: disable=too-few-public-methods
    """
    Read-only Backend that always returns a float tensor filled with a
    given constant value.

    :param value: Value to fill the tensor with.
    :param num_channels: Number of channels for the read tensor.
    """

    value: float = 0
    num_channels: int = 0

    @property
    def name(self) -> str:  # pragma: no cover
        return "ConstantVolumetricBackend<{self.value}>"

    @name.setter
    def name(self, name: str) -> None:  # pragma: no cover
        raise NotImplementedError("cannot set `name` for `ConstantVolumetricBackend` directly;")

    @property
    def dtype(self) -> np.dtype:  # pragma: no cover
        return np.dtype("float32")

    @property
    def is_local(self) -> bool:  # pragma: no cover
        return True

    @property
    def enforce_chunk_aligned_writes(self) -> bool:  # pragma: no cover
        return False

    @enforce_chunk_aligned_writes.setter
    def enforce_chunk_aligned_writes(self, value: bool) -> None:  # pragma: no cover
        raise NotImplementedError(
            "cannot set `enforce_chunk_aligned_writes` for ConstantVolumetricBackend directly;"
        )

    @property
    def allow_cache(self) -> bool:  # pragma: no cover
        return True

    @allow_cache.setter
    def allow_cache(self, value: Union[bool, str]) -> None:  # pragma: no cover
        raise NotImplementedError(
            "cannot set `allow_cache` for ConstantVolumetricBackend directly;"
        )

    @property
    def use_compression(self) -> bool:  # pragma: no cover
        return False

    @use_compression.setter
    def use_compression(self, value: bool) -> None:  # pragma: no cover
        raise NotImplementedError(
            "cannot set `use_compression` for ConstantVolumetricBackend directly;"
        )

    def clear_cache(self) -> None:  # pragma: no cover
        pass

    def read(self, idx: VolumetricIndex) -> npt.NDArray:
        # Data out: cxyz
        slices = idx.to_slices()
        result = (
            np.ones(
                (
                    self.num_channels,
                    slices[0].stop - slices[0].start,
                    slices[1].stop - slices[1].start,
                    slices[2].stop - slices[2].start,
                )
            )
            * self.value
        )
        return result

    def write(self, idx: VolumetricIndex, data: npt.NDArray):  # pragma: no cover
        raise RuntimeError("cannot perform `write` operation on a ConstantVolumetricBackend")

    def with_changes(self, **kwargs) -> ConstantVolumetricBackend:  # pragma: no cover
        return self

    def get_voxel_offset(self, resolution: Vec3D) -> Vec3D[int]:  # pragma: no cover
        return Vec3D[int](0, 0, 0)

    def get_chunk_size(self, resolution: Vec3D) -> Vec3D[int]:  # pragma: no cover
        return Vec3D[int](1, 1, 1)

    def get_dataset_size(self, resolution: Vec3D) -> Vec3D[int]:  # pragma: no cover
        return Vec3D[int](0, 0, 0)

    def get_bounds(self, resolution: Vec3D) -> VolumetricIndex:  # pragma: no cover
        return VolumetricIndex.from_coords((0, 0, 0), (0, 0, 0), Vec3D[int](1, 1, 1))

    def get_chunk_aligned_index(
        self, idx: VolumetricIndex, mode: Literal["expand", "shrink", "round"]
    ) -> VolumetricIndex:
        return idx  # pragma: no cover

    def assert_idx_is_chunk_aligned(self, idx: VolumetricIndex) -> None:  # pragma: no cover
        pass

    def pformat(self) -> str:  # pragma: no cover
        return self.name
