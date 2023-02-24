from typing import Protocol, TypeVar, runtime_checkable

from . import VolumetricBackend, VolumetricIndex

DataT = TypeVar("DataT")


@runtime_checkable
class VolumetricBasedLayerProtocol(Protocol[DataT]):
    @property
    def backend(self) -> VolumetricBackend:
        ...

    def __setitem__(self, idx: VolumetricIndex, data: DataT):
        ...

    def __getitem__(self, idx: VolumetricIndex) -> DataT:
        ...
