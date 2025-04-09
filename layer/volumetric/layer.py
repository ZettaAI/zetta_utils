from __future__ import annotations

from typing import Union

import attrs
import torch
from numpy import typing as npt

from zetta_utils.geometry import Vec3D

from .. import DataProcessor, IndexProcessor, JointIndexDataProcessor, Layer
from . import UserVolumetricIndex, VolumetricBackend, VolumetricIndex
from .conversion import convert_idx, convert_write

VolumetricDataProcT = Union[
    DataProcessor[npt.NDArray], JointIndexDataProcessor[npt.NDArray, VolumetricIndex]
]
VolumetricDataWriteProcT = Union[
    DataProcessor[npt.NDArray | torch.Tensor],
    JointIndexDataProcessor[npt.NDArray | torch.Tensor, VolumetricIndex],
]


@attrs.frozen
class VolumetricLayer(Layer[VolumetricIndex, npt.NDArray, npt.NDArray | torch.Tensor]):
    backend: VolumetricBackend[npt.NDArray, npt.NDArray | torch.Tensor]
    index_resolution: Vec3D | None = None
    default_desired_resolution: Vec3D | None = None
    allow_slice_rounding: bool = False
    readonly: bool = False

    index_procs: tuple[IndexProcessor[VolumetricIndex], ...] = ()
    read_procs: tuple[VolumetricDataProcT, ...] = ()
    write_procs: tuple[VolumetricDataWriteProcT, ...] = ()

    def __getitem__(self, idx: UserVolumetricIndex) -> npt.NDArray:
        idx_backend = convert_idx(
            idx,
            self.index_resolution,
            self.default_desired_resolution,
            self.allow_slice_rounding,
        )
        return self.read_with_procs(idx=idx_backend)

    def __setitem__(
        self, idx: UserVolumetricIndex, data: npt.NDArray | torch.Tensor | float | int | bool
    ):
        idx_backend, data_backend = convert_write(
            idx,
            data,
            self.index_resolution,
            self.default_desired_resolution,
            self.allow_slice_rounding,
        )
        self.write_with_procs(idx=idx_backend, data=data_backend)

    def pformat(self) -> str:  # pragma: no cover
        return self.backend.pformat()

    def with_changes(
        self,
        **kwargs,
    ):
        return attrs.evolve(self, **kwargs)  # pragma: no cover
