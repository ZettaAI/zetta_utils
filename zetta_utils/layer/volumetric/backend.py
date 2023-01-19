# pylint: disable=missing-docstring # pragma: no cover
from abc import abstractmethod
from typing import Literal

import torch

from zetta_utils.typing import IntVec3D, Vec3D

from .. import Backend
from . import VolumetricIndex


class VolumetricBackend(
    Backend[VolumetricIndex, torch.Tensor]
):  # pylint: disable=too-few-public-methods
    @property
    @abstractmethod
    def enforce_chunk_aligned_writes(self) -> bool:
        ...

    @enforce_chunk_aligned_writes.setter
    @abstractmethod
    def enforce_chunk_aligned_writes(self, value: bool) -> None:
        ...

    @abstractmethod
    def get_voxel_offset(self, resolution: Vec3D) -> IntVec3D:
        ...

    @abstractmethod
    def set_voxel_offset(self, voxel_offset: IntVec3D, resolution: Vec3D) -> None:
        """Sets the voxel offset at the given resolution for the backend.
        The offsets for other resolutions are unaffected."""

    @abstractmethod
    def get_chunk_size(self, resolution: Vec3D) -> IntVec3D:
        ...

    @abstractmethod
    def set_chunk_size(self, chunk_size: IntVec3D, resolution: Vec3D) -> None:
        """Sets the chunk size at the given resolution for the backend.
        The sizes for other resolutions are unaffected."""

    @abstractmethod
    def get_chunk_aligned_index(
        self, index: VolumetricIndex, mode: Literal["expand", "shrink", "round"]
    ) -> VolumetricIndex:
        ...

    @abstractmethod
    def assert_idx_is_chunk_aligned(self, idx: VolumetricIndex) -> None:
        ...
