# pylint: disable=missing-docstring,no-self-use,unused-argument
from __future__ import annotations

from typing import Any, Sequence, Tuple, Union, overload

import attrs
from typing_extensions import TypeGuard

from zetta_utils.layer.frontend_base import Frontend

from .. import DataProcessor, IndexProcessor, JointIndexDataProcessor, Layer
from .backend import DBDataT, DBIndex, DBRowDataT, DBValueT

RowIndex = Union[str, list[str]]
ColIndex = Union[str, Tuple[str, ...]]
RowColIndex = Tuple[RowIndex, ColIndex]

RawDBIndex = Union[RowIndex, RowColIndex]
UserDBIndex = Union[RawDBIndex, DBIndex]


DBDataProcT = Union[DataProcessor[DBDataT], JointIndexDataProcessor[DBDataT, DBIndex]]


def is_scalar_seq(values: Sequence[Any]) -> TypeGuard[Sequence[DBValueT]]:
    return all(isinstance(v, (bool, int, float, str)) for v in values) and len(values) > 0


def is_rowdata_seq(values: Sequence[Any]) -> TypeGuard[Sequence[DBRowDataT]]:
    return all(isinstance(v, dict) for v in values) and len(values) > 0


class DBFrontend(Frontend[UserDBIndex, DBIndex, DBDataT, DBDataT]):
    def convert_idx(self, idx_user: UserDBIndex) -> DBIndex:
        if isinstance(idx_user, DBIndex):
            return idx_user

        if isinstance(idx_user, (str, int)):
            row_col_keys = {idx_user: ("value",)}
            return DBIndex(row_col_keys)

        if isinstance(idx_user, list):
            row_col_keys = {row_key: ("value",) for row_key in idx_user}
            return DBIndex(row_col_keys)

        row_keys, col_keys = idx_user
        if isinstance(row_keys, (str, int)):
            row_keys = [row_keys]
        if isinstance(col_keys, str):
            col_keys = (col_keys,)
        row_col_keys = {row_key: col_keys for row_key in row_keys}  # type: ignore
        return DBIndex(row_col_keys, col_keys)

    @overload
    def _convert_read_data(self, idx_user: str, data: DBDataT) -> DBValueT | DBRowDataT:
        ...

    @overload
    def _convert_read_data(self, idx_user: list[str], data: DBDataT) -> Sequence[DBValueT]:
        ...

    @overload
    def _convert_read_data(self, idx_user: Tuple[str, str], data: DBDataT) -> DBValueT:
        ...

    @overload
    def _convert_read_data(
        self, idx_user: Tuple[list[str], str], data: DBDataT
    ) -> Sequence[DBValueT]:
        ...

    @overload
    def _convert_read_data(
        self, idx_user: Tuple[str, Tuple[str, ...]], data: DBDataT
    ) -> DBRowDataT:
        ...

    @overload
    def _convert_read_data(
        self, idx_user: Tuple[list[str], Tuple[str, ...]], data: DBDataT
    ) -> DBDataT:
        ...

    def _convert_read_data(self, idx_user: UserDBIndex, data: DBDataT):
        if isinstance(idx_user, (str, int)):
            return data[0]["value"] if "value" in data[0] else data[0]

        if isinstance(idx_user, list):
            return [d["value"] for d in data]
        if isinstance(idx_user, tuple):
            row_keys, col_keys = idx_user
            if isinstance(row_keys, (str, int)):
                if isinstance(col_keys, str):
                    return data[0][col_keys]
                return {col_key: data[0][col_key] for col_key in col_keys if col_key in data[0]}
            elif isinstance(col_keys, str):
                return [row[col_keys] for row in data]
        return data

    def convert_write(self, idx_user: UserDBIndex, data_user) -> Tuple[DBIndex, DBDataT]:
        idx = self.convert_idx(idx_user)
        if isinstance(data_user, (bool, int, float, str)):
            return idx, [{"value": data_user}]

        if isinstance(data_user, dict):
            return idx, [data_user]

        if is_scalar_seq(data_user):
            return idx, [{"value": d} for d in data_user]

        if is_rowdata_seq(data_user):
            return idx, data_user

        raise ValueError(f"Unsupported data type: {type(data_user)}")