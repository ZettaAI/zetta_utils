from typing import Any, List, Protocol, TypeVar, Union, runtime_checkable

from . import JointIndexDataProcessor, Processor

IndexT = TypeVar("IndexT")
DataT = TypeVar("DataT")


@runtime_checkable
class LayerWithIndexT(Protocol[IndexT]):
    index_procs: List[Processor[IndexT]]

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
    index_procs: List[Processor[IndexT]]
    read_procs: List[
        Union[
            Processor[DataT],
            JointIndexDataProcessor[DataT, IndexT],
        ]
    ]

    write_procs: List[
        Union[
            Processor[DataT],
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
