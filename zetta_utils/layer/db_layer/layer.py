# pylint: disable=missing-docstring,no-self-use,unused-argument
from __future__ import annotations

from typing import Any, List, Sequence, Tuple, Union, overload

import attrs
from typing_extensions import TypeGuard

from .. import DataProcessor, IndexProcessor, JointIndexDataProcessor, Layer
from . import DBBackend, DBDataT, DBIndex, DBRowDataT, DBValueT

RowIndex = Union[str, List[str]]
ColIndex = Union[str, Tuple[str, ...]]
RowColIndex = Tuple[RowIndex, ColIndex]

RawDBIndex = Union[RowIndex, RowColIndex]
UserDBIndex = Union[RawDBIndex, DBIndex]


DBDataProcT = Union[DataProcessor[DBDataT], JointIndexDataProcessor[DBDataT, DBIndex]]


def is_scalar_seq(values: Sequence[Any]) -> TypeGuard[Sequence[DBValueT]]:
    return all(isinstance(v, (bool, int, float, str)) for v in values) and len(values) > 0


def is_rowdata_seq(values: Sequence[Any]) -> TypeGuard[Sequence[DBRowDataT]]:
    return all(isinstance(v, dict) for v in values) and len(values) > 0


@attrs.mutable
class DBLayer(Layer[DBIndex, DBDataT, DBDataT]):
    backend: DBBackend
    readonly: bool = False

    index_procs: tuple[IndexProcessor[DBIndex], ...] = ()
    read_procs: tuple[DBDataProcT, ...] = ()
    write_procs: tuple[DBDataProcT, ...] = ()

    def _convert_idx(self, idx_user: UserDBIndex) -> DBIndex:
        if isinstance(idx_user, DBIndex):
            return idx_user

        if isinstance(idx_user, str):
            row_col_keys = {idx_user: ("value",)}
            return DBIndex(row_col_keys)

        if isinstance(idx_user, List):
            row_col_keys = {row_key: ("value",) for row_key in idx_user}
            return DBIndex(row_col_keys)

        row_keys, col_keys = idx_user
        if isinstance(row_keys, str):
            row_keys = [row_keys]
        if isinstance(col_keys, str):
            col_keys = (col_keys,)
        row_col_keys = {row_key: col_keys for row_key in row_keys}  # type: ignore
        return DBIndex(row_col_keys)

    @overload
    def _convert_read_data(self, idx_user: str, data: DBDataT) -> DBValueT | DBRowDataT:
        ...

    @overload
    def _convert_read_data(self, idx_user: List[str], data: DBDataT) -> Sequence[DBValueT]:
        ...

    @overload
    def _convert_read_data(self, idx_user: Tuple[str, str], data: DBDataT) -> DBValueT:
        ...

    @overload
    def _convert_read_data(
        self, idx_user: Tuple[List[str], str], data: DBDataT
    ) -> Sequence[DBValueT]:
        ...

    @overload
    def _convert_read_data(
        self, idx_user: Tuple[str, Tuple[str, ...]], data: DBDataT
    ) -> DBRowDataT:
        ...

    @overload
    def _convert_read_data(
        self, idx_user: Tuple[List[str], Tuple[str, ...]], data: DBDataT
    ) -> DBDataT:
        ...

    def _convert_read_data(self, idx_user: UserDBIndex, data: DBDataT):
        if isinstance(idx_user, str):
            return data[0]["value"] if "value" in data[0] else data[0]

        if isinstance(idx_user, list):
            return [d["value"] for d in data]
        if isinstance(idx_user, tuple):
            row_keys, col_keys = idx_user
            if isinstance(row_keys, str):
                if isinstance(col_keys, str):
                    return data[0][col_keys]
                return {col_key: data[0][col_key] for col_key in col_keys if col_key in data[0]}
            elif isinstance(col_keys, str):
                return [row[col_keys] for row in data]
        return data

    @overload
    def _convert_write(
        self,
        idx_user: str,
        data_user: DBValueT,
    ) -> Tuple[DBIndex, DBDataT]:
        ...

    @overload
    def _convert_write(
        self,
        idx_user: List[str],
        data_user: Sequence[DBValueT],
    ) -> Tuple[DBIndex, DBDataT]:
        ...

    @overload
    def _convert_write(
        self,
        idx_user: Tuple[str, ColIndex],
        data_user: DBRowDataT,
    ) -> Tuple[DBIndex, DBDataT]:
        ...

    @overload
    def _convert_write(
        self,
        idx_user: Tuple[List[str], ColIndex],
        data_user: DBDataT,
    ) -> Tuple[DBIndex, DBDataT]:
        ...

    def _convert_write(self, idx_user, data_user):
        idx = self._convert_idx(idx_user)
        if isinstance(data_user, (bool, int, float, str)):
            return idx, [{"value": data_user}]

        if isinstance(data_user, dict):
            return idx, [data_user]

        if is_scalar_seq(data_user):
            return idx, [{"value": d} for d in data_user]

        if is_rowdata_seq(data_user):
            return idx, data_user

        raise ValueError(f"Unsupported data type: {type(data_user)}")

    def __getitem__(self, idx):
        idx_backend = self._convert_idx(idx)
        data_backend = self.read_with_procs(idx=idx_backend)
        data_user = self._convert_read_data(idx_user=idx, data=data_backend)
        return data_user

    @overload
    def __setitem__(
        self,
        idx_user: str,
        data_user: DBValueT,
    ):
        ...

    @overload
    def __setitem__(
        self,
        idx_user: List[str],
        data_user: Sequence[DBValueT],
    ):
        ...

    @overload
    def __setitem__(
        self,
        idx_user: Tuple[str, ColIndex],
        data_user: DBRowDataT,
    ):
        ...

    @overload
    def __setitem__(
        self,
        idx_user: Tuple[List[str], ColIndex],
        data_user: DBDataT,
    ):
        ...

    def __setitem__(self, idx, data):
        idx_backend, data_backend = self._convert_write(idx_user=idx, data_user=data)
        self.write_with_procs(idx=idx_backend, data=data_backend)

    def __contains__(self, idx: str) -> bool:  # pragma: no cover # no logic
        return idx in self.backend

    def __len__(self) -> int:  # pragma: no cover # no logic
        return len(self.backend)

    def get(self, idx: str, default=None):
        """Read a single key from the database."""
        if idx in self.backend:
            return self[idx]
        return default

    def query(
        self, column_filter: dict[str, list] | None = None
    ) -> list[str]:  # pragma: no cover # no logic
        return self.backend.query(column_filter)
