# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Literal

import attrs
import torch

from zetta_utils.geometry import Vec3D

from .. import VolumetricBackend, VolumetricIndex, VolumetricLayer


@attrs.frozen
class VolumetricSetBackend(
    VolumetricBackend[dict[str, torch.Tensor]]
):  # pylint: disable=too-few-public-methods
    layers: dict[str, VolumetricLayer]

    @property
    def name(self) -> str:  # pragma: no cover
        children_names = {k: v.backend.name for k, v in self.layers.items()}
        return f"VolumetricSet[{children_names}]"

    @name.setter
    def name(self, name: str) -> None:  # pragma: no cover
        raise NotImplementedError(
            "cannot set `name` for VolumetricSetBackend directly;"
            " use `backend.with_changes(name='name')` instead."
        )

    @property
    def dtype(self) -> torch.dtype:  # pragma: no cover
        dtypes = {k: v.backend.dtype for k, v in self.layers.items()}
        if not len(set(dtypes.values())) == 1:
            raise ValueError(
                "Cannot determine consistent data type for the "
                f"volumetric layer set backend. Got: {dtypes}"
            )

        return list(dtypes.values())[0]

    @property
    def num_channels(self) -> int:  # pragma: no cover
        num_channels = {k: v.backend.num_channels for k, v in self.layers.items()}
        if not len(set(num_channels.values())) == 1:
            raise ValueError(
                "Cannot determine consistent number of channels for the "
                f"volumetric layer set backend. Got: {num_channels}"
            )
        return list(num_channels.values())[0]

    @property
    def is_local(self) -> bool:  # pragma: no cover
        is_locals = {k: v.backend.is_local for k, v in self.layers.items()}
        if not len(set(is_locals.values())) == 1:
            raise ValueError(
                "Cannot determine consistent `is_local` for the "
                f"volumetric layer set backend. Got: {is_locals}"
            )
        return list(is_locals.values())[0]

    @property
    def enforce_chunk_aligned_writes(self) -> bool:  # pragma: no cover
        enforces = {k: v.backend.enforce_chunk_aligned_writes for k, v in self.layers.items()}
        if not len(set(enforces.values())) == 1:
            raise ValueError(
                "Cannot determine consistent `enforce_chunk_aligned_writes` for the "
                f"volumetric layer set backend. Got: {enforces}"
            )
        return list(enforces.values())[0]

    @enforce_chunk_aligned_writes.setter  # pragma: no cover
    def enforce_chunk_aligned_writes(self, value: bool) -> None:  # pragma: no cover
        raise NotImplementedError(
            "cannot set `enforce_chunk_aligned_writes` for VolumetricSetBackend directly;"
            " use `backend.with_changes(non_aligned_writes=value:bool)` instead."
        )

    @property
    def allow_cache(self) -> bool:  # pragma: no cover
        allow_caches = {k: v.backend.allow_cache for k, v in self.layers.items()}
        if not len(set(allow_caches.values())) == 1:
            raise ValueError(
                "Cannot determine consistent `allow_caches` for the "
                f"volumetric layer set backend. Got: {allow_caches}"
            )
        return list(allow_caches.values())[0]

    @allow_cache.setter
    def allow_cache(self, value: bool | str) -> None:  # pragma: no cover
        raise NotImplementedError(
            "cannot set `allow_cache` for VolumetricSetBackend directly;"
            " use `backend.with_changes(allow_cache=value:bool | str)` instead."
        )

    @property
    def use_compression(self) -> bool:  # pragma: no cover
        use_compressions = {k: v.backend.use_compression for k, v in self.layers.items()}
        if not len(set(use_compressions.values())) == 1:
            raise ValueError(
                "Cannot determine consistent `use_compressions` for the "
                f"volumetric layer set backend. Got: {use_compressions}"
            )
        return list(use_compressions.values())[0]

    @use_compression.setter
    def use_compression(self, value: bool) -> None:  # pragma: no cover
        raise NotImplementedError(
            "cannot set `use_compression` for VolumetricSetBackend directly;"
            " use `backend.with_changes(use_compression=value:bool)` instead."
        )

    def clear_cache(self) -> None:  # pragma: no cover
        for e in self.layers.values():
            e.backend.clear_cache()

    def get_voxel_offset(self, resolution: Vec3D) -> Vec3D[int]:  # pragma: no cover
        voxel_offsets = {
            k: v.backend.get_voxel_offset(resolution=resolution) for k, v in self.layers.items()
        }
        if not len(set(voxel_offsets.values())) == 1:
            raise ValueError(
                "Cannot determine consistent `voxel_offset` for the "
                f"volumetric layer set backend. Got: {voxel_offsets}"
            )
        return list(voxel_offsets.values())[0]

    def get_chunk_size(self, resolution: Vec3D) -> Vec3D[int]:  # pragma: no cover
        chunk_sizes = {
            k: v.backend.get_chunk_size(resolution=resolution) for k, v in self.layers.items()
        }
        if not len(set(chunk_sizes.values())) == 1:
            raise ValueError(
                "Cannot determine consistent `chunk_size` for the "
                f"volumetric layer set backend. Got: {chunk_sizes}"
            )
        return list(chunk_sizes.values())[0]

    def get_chunk_aligned_index(
        self, idx: VolumetricIndex, mode: Literal["expand", "shrink", "round"]
    ) -> VolumetricIndex:  # pragma: no cover
        chunk_aligned_indexs = {
            k: v.backend.get_chunk_aligned_index(idx=idx, mode=mode)
            for k, v in self.layers.items()
        }
        if not len(set(chunk_aligned_indexs.values())) == 1:
            raise ValueError(
                "Cannot determine consistent `get_chunk_aligned_index` for the "
                f"volumetric layer set backend. Got: {chunk_aligned_indexs}"
            )
        return list(chunk_aligned_indexs.values())[0]

    def assert_idx_is_chunk_aligned(self, idx: VolumetricIndex) -> None:  # pragma: no cover
        for e in self.layers.values():
            e.backend.assert_idx_is_chunk_aligned(idx=idx)

    def read(self, idx: VolumetricIndex) -> dict[str, torch.Tensor]:
        return {k: v.read_with_procs(idx) for k, v in self.layers.items()}

    def write(self, idx: VolumetricIndex, data: dict[str, torch.Tensor]):
        for k, v in data.items():
            self.layers[k].write_with_procs(idx, v)

    def with_changes(self, **kwargs) -> VolumetricSetBackend:  # pragma: no cover
        return attrs.evolve(
            self,
            layers={
                k: attrs.evolve(v, backend=v.backend.with_changes(**kwargs))
                for k, v in self.layers.items()
            },
        )
