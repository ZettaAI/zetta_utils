# pylint: disable=missing-docstring # pragma: no cover
from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import torch

from zetta_utils.geometry import IntVec3D, Vec3D

from .. import Backend
from . import VolumetricIndex


class VolumetricBackend(
    Backend[VolumetricIndex, torch.Tensor]
):  # pylint: disable=too-few-public-methods
    @property
    @abstractmethod
    def is_local(self) -> bool:
        ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        ...

    @property
    @abstractmethod
    def num_channels(self) -> int:
        ...

    @property
    @abstractmethod
    def allow_cache(self) -> bool:
        ...

    @property
    @abstractmethod
    def enforce_chunk_aligned_writes(self) -> bool:
        ...

    @property
    @abstractmethod
    def use_compression(self) -> bool:
        ...

    @abstractmethod
    def clear_cache(self) -> None:
        ...

    @abstractmethod
    def get_voxel_offset(self, resolution: Vec3D) -> IntVec3D:
        ...

    @abstractmethod
    def get_chunk_size(self, resolution: Vec3D) -> IntVec3D:
        ...

    @abstractmethod
    def get_chunk_aligned_index(
        self, index: VolumetricIndex, mode: Literal["expand", "shrink", "round"]
    ) -> VolumetricIndex:
        ...

    """
    TODO: Turn this into a ParamSpec.
    The .with_changes for VolumetricBackend
        MUST handle the following parameters:
    "allow_cache" = value: Union[bool, str]
    "use_compression" = value: str
    "enforce_chunk_aligned_writes" = value: bool
    "voxel_offset_res" = (voxel_offset, resolution): Tuple[IntVec3D, Vec3D]
    "chunk_size_res" = (chunk_size, resolution): Tuple[IntVec3D, Vec3D]
    """

    @abstractmethod
    def assert_idx_is_chunk_aligned(self, idx: VolumetricIndex) -> None:
        ...
