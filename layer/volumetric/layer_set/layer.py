from __future__ import annotations

from typing import Mapping, Union

import attrs
import torch
from numpy import typing as npt

from zetta_utils.geometry import Vec3D

from ... import DataProcessor, IndexProcessor, JointIndexDataProcessor, Layer
from .. import UserVolumetricIndex, VolumetricIndex
from ..conversion import convert_idx, convert_write
from . import VolumetricSetBackend

VolumetricSetDataProcT = Union[
    DataProcessor[dict[str, npt.NDArray]],
    JointIndexDataProcessor[dict[str, npt.NDArray], VolumetricIndex],
]
VolumetricSetDataWriteProcT = Union[
    DataProcessor[Mapping[str, npt.NDArray | torch.Tensor]],
    JointIndexDataProcessor[Mapping[str, npt.NDArray | torch.Tensor], VolumetricIndex],
]


@attrs.frozen
class VolumetricLayerSet(
    Layer[VolumetricIndex, dict[str, npt.NDArray], Mapping[str, npt.NDArray | torch.Tensor]]
):
    backend: VolumetricSetBackend

    readonly: bool = False
    index_resolution: Vec3D | None = None
    default_desired_resolution: Vec3D | None = None
    allow_slice_rounding: bool = False

    index_procs: tuple[IndexProcessor[VolumetricIndex], ...] = ()
    read_procs: tuple[VolumetricSetDataProcT, ...] = ()
    write_procs: tuple[VolumetricSetDataWriteProcT, ...] = ()

    def __getitem__(self, idx: UserVolumetricIndex) -> dict[str, npt.NDArray]:
        idx_backend = convert_idx(
            idx,
            self.index_resolution,
            self.default_desired_resolution,
            self.allow_slice_rounding,
        )
        return self.read_with_procs(idx=idx_backend)

    def __setitem__(
        self,
        idx: UserVolumetricIndex,
        data: Mapping[str, Union[npt.NDArray, torch.Tensor, int, float, bool]],
    ):
        idx_backend: VolumetricIndex | None = None
        idx_last: VolumetricIndex | None = None
        data_backend = {}
        for k, v in data.items():
            this_idx_backend, this_data_backend = convert_write(
                idx,
                v,
                self.index_resolution,
                self.default_desired_resolution,
                self.allow_slice_rounding,
            )
            data_backend[k] = this_data_backend
            assert idx_last is None or this_idx_backend == idx_last
            idx_last = this_idx_backend
            idx_backend = this_idx_backend
        assert idx_backend is not None
        self.write_with_procs(idx=idx_backend, data=data_backend)

    def pformat(self) -> str:  # pragma: no cover
        return self.backend.pformat()

    def with_changes(self, **kwargs):
        return attrs.evolve(self, **kwargs)  # pragma: no cover
