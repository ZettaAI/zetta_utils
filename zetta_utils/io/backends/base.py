# pylint: disable=missing-docstring # pragma: no cover
from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from zetta_utils.io.indexes import Index

IndexT = TypeVar("IndexT", bound=Index)


class IOBackend(ABC, Generic[IndexT]):  # pylint: disable=too-few-public-methods
    @abstractmethod
    def read(self, idx: IndexT):
        """Reads data from the given index"""

    @abstractmethod
    def write(self, idx: IndexT, value):
        """Writes given value to the given index"""

    # Open problem:
    # This ugliness could be avoided with proper templating
    # CC https://stackoverflow.com/questions/55345608/instantiate-a-type-that-is-a-typevar
    @classmethod
    def get_index_type(cls):
        result = cls.__orig_bases__[0].__args__[0]  # pylint: disable=no-member
        return result
