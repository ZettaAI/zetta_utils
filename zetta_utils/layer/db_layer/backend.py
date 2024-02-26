# pylint: disable=missing-docstring # pragma: no cover
from __future__ import annotations

from abc import abstractmethod
from typing import MutableMapping, Sequence, Union

from .. import Backend
from . import DBIndex

DBValueT = Union[bool, int, float, str, list[bool | int | float | str]]
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
