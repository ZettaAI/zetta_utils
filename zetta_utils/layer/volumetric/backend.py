# pylint: disable=missing-docstring # pragma: no cover
from __future__ import annotations

from abc import abstractmethod
from typing import Literal, TypeVar

import attrs
import numpy as np

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
    def dtype(self) -> np.dtype:
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
    def pformat(self) -> str:
        ...

    def get_chunk_aligned_index(  # pragma: no cover
        self, idx: VolumetricIndex, mode: Literal["expand", "shrink"]
    ) -> VolumetricIndex:
        if mode not in ["expand", "shrink"]:
            raise NotImplementedError(
                f"mode must be set to 'expand' or 'shrink'; received '{mode}'"
            )
        return idx.snapped(
            self.get_voxel_offset(idx.resolution), self.get_chunk_size(idx.resolution), mode
        )

    def assert_idx_is_chunk_aligned(self, idx: VolumetricIndex) -> None:
        idx_expanded = self.get_chunk_aligned_index(idx, mode="expand")
        idx_shrunk = self.get_chunk_aligned_index(idx, mode="shrink")

        if idx != idx_expanded:
            raise ValueError(
                "The BBox3D of the specified VolumetricIndex is not chunk-aligned with"
                + f" the VolumetricLayer at `{self.name}`;\n"
                + f"in {tuple(idx.resolution)} {idx.bbox.unit} voxels:"
                + f" offset: {self.get_voxel_offset(idx.resolution)},"
                + f" chunk_size: {self.get_chunk_size(idx.resolution)}\n"
                + f"Received BBox3D: {idx.pformat()}\n"
                + "Nearest chunk-aligned BBox3Ds:\n"
                + f" - expanded : {idx_expanded.pformat()}\n"
                + f" - shrunk   : {idx_shrunk.pformat()}"
            )
