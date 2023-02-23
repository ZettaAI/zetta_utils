from typing import Union

import attrs
import torch

from zetta_utils import builder

from ... import DataProcessor, IndexProcessor, JointIndexDataProcessor, Layer
from .. import VolumetricIndex
from . import VolumetricLayerSetBackend

VolumetricSetDataProcT = Union[
    DataProcessor[dict[str, torch.Tensor]],
    JointIndexDataProcessor[dict[str, torch.Tensor], VolumetricIndex],
]


@builder.register("VolumetricLayerSet")
@attrs.frozen
class VolumetricLayerSet(Layer[VolumetricIndex, dict[str, torch.Tensor]]):
    backend: VolumetricLayerSetBackend

    readonly: bool = False

    index_procs: tuple[IndexProcessor[VolumetricIndex], ...] = ()
    read_procs: tuple[VolumetricSetDataProcT, ...] = ()
    write_procs: tuple[VolumetricSetDataProcT, ...] = ()

    # TODO: we could add custom user index formats by re-using converter functions from
    # the vanilla VolumetricLayer

    def __getitem__(self, idx: VolumetricIndex) -> dict[str, torch.Tensor]:
        return self.read_with_procs(idx=idx)

    def __setitem__(self, idx: VolumetricIndex, data: dict[str, torch.Tensor]):
        self.write_with_procs(idx=idx, data=data)
