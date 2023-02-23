from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

IndexT_contra = TypeVar("IndexT_contra", contravariant=True)
DataT = TypeVar("DataT")


@runtime_checkable
class LayerWithIndexT(Protocol[IndexT_contra]):
    def read_with_procs(self, idx_user: IndexT_contra) -> Any:
        ...

    def write_with_procs(self, idx_user: IndexT_contra, data_user: Any):
        ...

    def __getitem__(self, idx_user: IndexT_contra) -> Any:
        ...

    def __setitem__(self, idx_user: IndexT_contra, data_user: Any):
        ...


@runtime_checkable
class LayerWithIndexDataT(Protocol[IndexT_contra, DataT]):
    def read_with_procs(self, idx_user: IndexT_contra) -> DataT:
        ...

    def write_with_procs(self, idx_user: IndexT_contra, data_user: DataT):
        ...

    def __getitem__(self, idx_user: IndexT_contra) -> DataT:
        ...

    def __setitem__(self, idx_user: IndexT_contra, data_user: DataT):
        ...
