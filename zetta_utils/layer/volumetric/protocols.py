from __future__ import annotations

from typing import Iterable, Optional, Protocol, TypeVar, Union, runtime_checkable

from .. import DataProcessor, IndexProcessor, JointIndexDataProcessor
from . import VolumetricBackend, VolumetricIndex

IndexT = TypeVar("IndexT", bound=VolumetricIndex)
DataT = TypeVar("DataT")
VolumetricBasedLayerProtocolT = TypeVar(
    "VolumetricBasedLayerProtocolT", bound="VolumetricBasedLayerProtocol"
)


@runtime_checkable
class VolumetricBasedLayerProtocol(Protocol[DataT, IndexT]):
    @property
    def backend(self) -> VolumetricBackend:
        ...

    def __setitem__(self, idx: IndexT, data: DataT):
        ...

    def __getitem__(self, idx: IndexT) -> DataT:
        ...

    def pformat(self) -> str:
        ...

    def with_procs(
        self: VolumetricBasedLayerProtocolT,
        index_procs: Optional[Iterable[IndexProcessor[IndexT]]] = None,
        read_procs: Optional[
            Iterable[Union[DataProcessor[DataT], JointIndexDataProcessor[DataT, IndexT]]]
        ] = None,
        write_procs: Optional[
            Iterable[Union[DataProcessor[DataT], JointIndexDataProcessor[DataT, IndexT]]]
        ] = None,
    ) -> VolumetricBasedLayerProtocolT:
        ...
