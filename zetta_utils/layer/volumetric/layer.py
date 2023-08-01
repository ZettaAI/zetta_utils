from __future__ import annotations

from typing import Union

import attrs
import torch

from .. import DataProcessor, IndexProcessor, JointIndexDataProcessor, Layer
from . import (
    UserVolumetricIndex,
    VolumetricBackend,
    VolumetricFrontend,
    VolumetricIndex,
)

VolumetricDataProcT = Union[
    DataProcessor[torch.Tensor], JointIndexDataProcessor[torch.Tensor, VolumetricIndex]
]


@attrs.frozen
class VolumetricLayer(Layer[VolumetricIndex, torch.Tensor]):
    backend: VolumetricBackend[torch.Tensor]
    frontend: VolumetricFrontend
    readonly: bool = False

    index_procs: tuple[IndexProcessor[VolumetricIndex], ...] = ()
    read_procs: tuple[VolumetricDataProcT, ...] = ()
    write_procs: tuple[VolumetricDataProcT, ...] = ()

    def __getitem__(self, idx: UserVolumetricIndex) -> torch.Tensor:
        idx_backend = self.frontend.convert_idx(idx)
        return self.read_with_procs(idx=idx_backend)

    def __setitem__(self, idx: UserVolumetricIndex, data: torch.Tensor | float | int | bool):
        idx_backend, data_backend = self.frontend.convert_write(idx, data)
        self.write_with_procs(idx=idx_backend, data=data_backend)

    def pformat(self) -> str:  # pragma: no cover
        return self.backend.pformat()

    def with_changes(
        self,
        **kwargs,
    ):
        return attrs.evolve(self, **kwargs)  # pragma: no cover
