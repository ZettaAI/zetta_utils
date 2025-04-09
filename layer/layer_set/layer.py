from typing import TypeVar, Union

import attrs

from .. import DataProcessor, IndexProcessor, JointIndexDataProcessor, Layer
from . import LayerSetBackend

IndexT = TypeVar("IndexT")
DataT = TypeVar("DataT")
DataWriteT = TypeVar("DataWriteT")

LayerSetDataProcT = Union[
    DataProcessor[dict[str, DataT]],
    JointIndexDataProcessor[dict[str, DataT], IndexT],
]


@attrs.frozen
class LayerSet(Layer[IndexT, dict[str, DataT], dict[str, DataWriteT]]):
    backend: LayerSetBackend[IndexT, DataT, DataWriteT]

    readonly: bool = False

    index_procs: tuple[IndexProcessor[IndexT], ...] = ()
    read_procs: tuple[LayerSetDataProcT, ...] = ()
    write_procs: tuple[LayerSetDataProcT, ...] = ()

    def __getitem__(self, idx: IndexT) -> dict[str, DataT]:
        return self.read_with_procs(idx=idx)

    def __setitem__(self, idx: IndexT, data: dict[str, DataWriteT]):
        self.write_with_procs(idx=idx, data=data)
