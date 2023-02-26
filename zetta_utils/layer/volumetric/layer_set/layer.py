from __future__ import annotations

from typing import Mapping, Union

import attrs
import torch
from typeguard import typechecked

from ... import DataProcessor, IndexProcessor, JointIndexDataProcessor, Layer
from .. import UserVolumetricIndex, VolumetricFrontend, VolumetricIndex
from . import VolumetricSetBackend

VolumetricSetDataProcT = Union[
    DataProcessor[dict[str, torch.Tensor]],
    JointIndexDataProcessor[dict[str, torch.Tensor], VolumetricIndex],
]


@typechecked
@attrs.frozen
class VolumetricLayerSet(Layer[VolumetricIndex, dict[str, torch.Tensor]]):
    backend: VolumetricSetBackend
    frontend: VolumetricFrontend

    readonly: bool = False

    index_procs: tuple[IndexProcessor[VolumetricIndex], ...] = ()
    read_procs: tuple[VolumetricSetDataProcT, ...] = ()
    write_procs: tuple[VolumetricSetDataProcT, ...] = ()

    def __getitem__(self, idx: UserVolumetricIndex) -> dict[str, torch.Tensor]:
        idx_backend = self.frontend.convert_idx(idx)
        return self.read_with_procs(idx=idx_backend)

    def __setitem__(
        self, idx: UserVolumetricIndex, data: Mapping[str, torch.Tensor | int | float | bool]
    ):
        idx_backend: VolumetricIndex | None = None
        idx_last: VolumetricIndex | None = None
        data_backend = {}
        for k, v in data.items():
            idx_backend, this_data_backend = self.frontend.convert_write(idx_user=idx, data_user=v)
            data_backend[k] = this_data_backend
            assert idx_last is None or idx_backend == idx_last
            idx_last = idx_backend
        assert idx_backend is not None
        self.write_with_procs(idx=idx_backend, data=data_backend)
