# pylint: disable=missing-docstring # pragma: no cover
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from zetta_utils.layer import LayerIndex

IndexT = TypeVar("IndexT", bound=LayerIndex)
DataT = TypeVar("DataT")


class LayerBackend(Generic[IndexT, DataT], ABC):  # pylint: disable=too-few-public-methods
    @abstractmethod
    def read(self, idx: IndexT) -> DataT:
        """Reads data from the given index"""

    @abstractmethod
    def write(self, idx: IndexT, value: DataT):
        """Writes given value to the given index"""

    @abstractmethod
    def get_name(self) -> str:
        """Get name for the layer"""

    # Open problem:
    # This ugliness could be avoided with proper templating
    # CC https://stackoverflow.com/questions/55345608/instantiate-a-type-that-is-a-typevar
    @classmethod
    def get_index_type(cls):  # pragma: no cover
        result = cls.__orig_bases__[0].__args__[0]
        return result
