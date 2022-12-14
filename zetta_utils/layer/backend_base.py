# pylint: disable=missing-docstring # pragma: no cover
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

IndexT = TypeVar("IndexT")
DataT = TypeVar("DataT")


class Backend(ABC, Generic[IndexT, DataT]):  # pylint: disable=too-few-public-methods
    @abstractmethod
    def read(self, idx: IndexT) -> DataT:
        """Reads data from the given index"""

    @abstractmethod
    def write(self, idx: IndexT, data: DataT):
        """Writes given data to the given index"""

    @abstractmethod
    def get_name(self) -> str:
        """Get name for the layer"""
