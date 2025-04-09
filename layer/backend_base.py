# pylint: disable=missing-docstring # pragma: no cover
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

IndexT = TypeVar("IndexT")
DataT = TypeVar("DataT")
DataWriteT = TypeVar("DataWriteT")


class Backend(ABC, Generic[IndexT, DataT, DataWriteT]):  # pylint: disable=too-few-public-methods
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def read(self, idx: IndexT) -> DataT:
        """Reads data from the given index"""

    @abstractmethod
    def write(self, idx: IndexT, data: DataWriteT):
        """Writes given data to the given index"""

    @abstractmethod
    def with_changes(self, **kwargs) -> Backend[IndexT, DataT, DataWriteT]:  # pragma: no cover
        """Remakes the Layer with the requested backend changes. The kwargs are not typed
        since the implementation is currently based on `attrs.evolve` and the
        base Backend class does not have any attrs, leaving all implementation to the inherited
        classes.
        In the future, Backend can be typed using a ParamSpec so that kwargs can be typed as
        `P.kwargs`."""
        # return attrs.evolve(self, **kwargs) # has to be implemented by the
        # child class, as it's not necesserily an `attrs` class
