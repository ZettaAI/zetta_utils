# pylint: disable=missing-docstring # pragma: no cover
from __future__ import annotations

from abc import abstractmethod
from typing import MutableMapping, Sequence, Union

from .. import Backend
from . import DBIndex

DBScalarValueT = Union[bool, int, float, str]
DBArrayValueT = list[DBScalarValueT]
DBValueT = Union[DBScalarValueT, DBArrayValueT]
DBRowDataT = MutableMapping[str, DBValueT | None]
DBDataT = Sequence[DBRowDataT]


class DBBackend(Backend[DBIndex, DBDataT, DBDataT]):  # pylint: disable=too-few-public-methods
    @abstractmethod
    def __contains__(self, idx: str) -> bool:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def read(self, idx: DBIndex) -> DBDataT:
        ...

    @abstractmethod
    def write(self, idx: DBIndex, data: DBDataT):
        ...

    @abstractmethod
    def clear(self, idx: DBIndex | None = None) -> None:
        ...

    @abstractmethod
    def keys(
        self,
        column_filter: dict[str, list] | None = None,
        union: bool = True,
    ) -> list[str]:
        ...

    @abstractmethod
    def query(
        self,
        column_filter: dict[str, list] | None = None,
        return_columns: tuple[str, ...] = (),
        union: bool = True,
    ) -> dict[str, DBRowDataT]:
        ...

    @abstractmethod
    def get_batch(
        self, batch_number: int, avg_rows_per_batch: int, return_columns: tuple[str, ...] = ()
    ) -> dict[str, DBRowDataT]:
        ...
