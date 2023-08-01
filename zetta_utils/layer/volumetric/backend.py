# pylint: disable=missing-docstring # pragma: no cover
from __future__ import annotations

from abc import abstractmethod
from typing import Literal, TypeVar

import attrs
import torch

from zetta_utils.geometry import Vec3D

from .. import Backend
from . import VolumetricIndex

DataT = TypeVar("DataT")


@attrs.mutable
class VolumetricBackend(Backend[VolumetricIndex, DataT]):  # pylint: disable=too-few-public-methods
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
    def get_voxel_offset(self, resolution: Vec3D) -> Vec3D[int]:
        ...

    @abstractmethod
    def get_chunk_size(self, resolution: Vec3D) -> Vec3D[int]:
        ...

    @abstractmethod
    def get_dataset_size(self, resolution: Vec3D) -> Vec3D[int]:
        ...

    @abstractmethod
    def get_chunk_aligned_index(
        self, idx: VolumetricIndex, mode: Literal["expand", "shrink", "round"]
    ) -> VolumetricIndex:
        ...

    """
    TODO: Turn this into a ParamSpec.
    The .with_changes for VolumetricBackend
        MUST handle the following parameters:
    "allow_cache" = value: Union[bool, str]
    "use_compression" = value: str
    "enforce_chunk_aligned_writes" = value: bool
    "voxel_offset_res" = (voxel_offset, resolution): Tuple[Vec3D[int], Vec3D]
    "chunk_size_res" = (chunk_size, resolution): Tuple[Vec3D[int], Vec3D]
    "dataest_size_res" = (dataset_size, resolution): Tuple[Vec3D[int], Vec3D]
    """

    def with_changes(self, **kwargs) -> VolumetricBackend[DataT]:
        return attrs.evolve(self, **kwargs)  # pragma: no cover

    @abstractmethod
    def assert_idx_is_chunk_aligned(self, idx: VolumetricIndex) -> None:
        ...

    @abstractmethod
    def pformat(self) -> str:
        ...
