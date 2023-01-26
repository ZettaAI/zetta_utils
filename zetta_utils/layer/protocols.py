from __future__ import annotations

from typing import Any, List, Protocol, TypeVar, Union, runtime_checkable

from . import DataProcessor, IndexProcessor, JointIndexDataProcessor

IndexT = TypeVar("IndexT")
DataT = TypeVar("DataT")


@runtime_checkable
class LayerWithIndexT(Protocol[IndexT]):
    index_procs: List[IndexProcessor[IndexT]]

    def read(self, idx_user: IndexT) -> Any:
        ...

    def write(self, idx_user: IndexT, data_user: Any):
        ...

    def __getitem__(self, idx_user: IndexT) -> Any:
        ...

    def __setitem__(self, idx_user: IndexT, data_user: Any):
        ...


@runtime_checkable
class LayerWithIndexDataT(Protocol[IndexT, DataT]):
    index_procs: List[IndexProcessor[IndexT]]
    read_procs: List[
        Union[
            DataProcessor[DataT],
            JointIndexDataProcessor[DataT, IndexT],
        ]
    ]

    write_procs: List[
        Union[
            DataProcessor[DataT],
            JointIndexDataProcessor[DataT, IndexT],
        ]
    ]

    def read(self, idx_user: IndexT) -> DataT:
        ...

    def write(self, idx_user: IndexT, data_user: DataT):
        ...

    def __getitem__(self, idx_user: IndexT) -> DataT:
        ...

    def __setitem__(self, idx_user: IndexT, data_user: DataT):
        ...
