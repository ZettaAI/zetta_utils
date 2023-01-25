# pylint: disable=missing-docstring # pragma: no cover
from __future__ import annotations

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
    def with_changes(self, **kwargs) -> Backend[IndexT, DataT]:
        """Changes the Backend with the kwargs being passed to the backend.
        Currently untyped; see `Layer.with_backend_changes()` for the reason."""

    name: str
