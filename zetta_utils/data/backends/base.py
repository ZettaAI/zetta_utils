# pylint: disable=missing-docstring # pragma: no cover
from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from zetta_utils.data.indexes import Index

IndexT = TypeVar("IndexT", bound=Index)


class DataBackend(ABC, Generic[IndexT]):  # pylint: disable=too-few-public-methods
    @abstractmethod
    def read(self, idx: IndexT):
        """Reads data from the given index"""

    @abstractmethod
    def write(self, idx: IndexT, value):
        """Writes given value to the given index"""

    @classmethod
    def get_index_type(cls):
        result = cls.__orig_bases__[0].__args__[0]  # pylint: disable=no-member
        return result
