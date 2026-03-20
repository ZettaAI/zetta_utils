from __future__ import annotations

from typing import Union

import attrs
import pandas as pd

from zetta_utils.layer.layer_base import Layer
from zetta_utils.layer.tools_base import (
    DataProcessor,
    IndexProcessor,
    JointIndexDataProcessor,
)
from zetta_utils.layer.volumetric.index import VolumetricIndex
from zetta_utils.layer.volumetric.tabular.backend import TabularBackend

TabularDataProcT = Union[
    DataProcessor[pd.DataFrame],
    JointIndexDataProcessor[pd.DataFrame, VolumetricIndex],
]


@attrs.frozen
class VolumetricTabularLayer(Layer[VolumetricIndex, pd.DataFrame, pd.DataFrame]):
    backend: TabularBackend
    readonly: bool = False

    index_procs: tuple[IndexProcessor[VolumetricIndex], ...] = ()
    read_procs: tuple[TabularDataProcT, ...] = ()
    write_procs: tuple[TabularDataProcT, ...] = ()

    def pformat(self) -> str:  # pragma: no cover
        return self.backend.pformat()

    def with_changes(self, **kwargs):
        return attrs.evolve(self, **kwargs)  # pragma: no cover
