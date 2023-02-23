from typing import TypeVar, Union

import attrs

from .. import DataProcessor, IndexProcessor, JointIndexDataProcessor, Layer
from . import LayerSetBackend

IndexT = TypeVar("IndexT")
DataT = TypeVar("DataT")

LayerSetDataProcT = Union[
    DataProcessor[dict[str, DataT]],
    JointIndexDataProcessor[dict[str, DataT], IndexT],
]


@attrs.frozen
class LayerSet(Layer[IndexT, dict[str, DataT]]):
    backend: LayerSetBackend[IndexT, DataT]

    readonly: bool = False

    index_procs: tuple[IndexProcessor[IndexT], ...] = ()
    read_procs: tuple[LayerSetDataProcT, ...] = ()
    write_procs: tuple[LayerSetDataProcT, ...] = ()

    def __getitem__(self, idx: IndexT) -> dict[str, DataT]:
        return self.read_with_procs(idx=idx)

    def __setitem__(self, idx: IndexT, data: dict[str, DataT]):
        self.write_with_procs(idx=idx, data=data)
