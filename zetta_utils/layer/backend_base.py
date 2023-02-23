# pylint: disable=missing-docstring # pragma: no cover
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import attrs

IndexT = TypeVar("IndexT")
DataT = TypeVar("DataT")


class Backend(ABC, Generic[IndexT, DataT]):  # pylint: disable=too-few-public-methods
    name: str

    @abstractmethod
    def read(self, idx: IndexT) -> DataT:
        """Reads data from the given index"""

    @abstractmethod
    def write(self, idx: IndexT, data: DataT):
        """Writes given data to the given index"""

    def with_changes(self, **kwargs) -> Backend[IndexT, DataT]:  # pragma: no cover
        """Remakes the Layer with the requested backend changes. The kwargs are not typed
        since the implementation is currently based on `attrs.evolve` and the
        base Backend class does not have any attrs, leaving all implementation to the inherited
        classes.
        In the future, Backend can be typed using a ParamSpec so that kwargs can be typed as
        `P.kwargs`."""
        return attrs.evolve(self, **kwargs)
