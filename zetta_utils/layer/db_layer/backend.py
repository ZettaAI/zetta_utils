# pylint: disable=missing-docstring # pragma: no cover
from __future__ import annotations

from abc import abstractmethod
from typing import MutableMapping, Sequence, Union

from .. import Backend
from . import DBIndex

DBScalarValueT = Union[bool, int, float, str]
DBArrayValueT = list[DBScalarValueT]
DBValueT = Union[DBScalarValueT, DBArrayValueT]
DBRowDataT = MutableMapping[str, DBValueT]
DBDataT = Sequence[DBRowDataT]


class DBBackend(Backend[DBIndex, DBDataT]):  # pylint: disable=too-few-public-methods
    @abstractmethod
    def read(self, idx: DBIndex) -> DBDataT:
        ...

    @abstractmethod
    def exists(self, idx: DBIndex) -> bool:
        ...

    @abstractmethod
    def write(self, idx: DBIndex, data: DBDataT):
        ...

    @abstractmethod
    def query(self, column_filter: dict[str, list] | None = None) -> list[str]:
        ...
