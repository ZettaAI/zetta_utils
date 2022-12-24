# pylint: disable=missing-docstring,no-self-use,unused-argument
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union, overload

import attrs
from typing_extensions import TypeGuard

from zetta_utils import builder

from ..frontend_base import Frontend
from . import DBIndex

RowIndex = Union[str, List[str]]
ColIndex = Tuple[str, ...]
RowColIndex = Tuple[RowIndex, ColIndex]

RawDBIndex = Union[RowIndex, RowColIndex]
UserDBIndex = Union[RawDBIndex, DBIndex]


ValueT = Union[bool, int, float, str]
MultiValueT = List[ValueT]
RowDataT = Dict[str, ValueT]
DataT = List[RowDataT]


def is_scalar_list(values: List[Any]) -> TypeGuard[List[ValueT]]:
    return all(isinstance(v, (bool, int, float, str)) for v in values)


def is_rowdata_list(values: List[Any]) -> TypeGuard[List[RowDataT]]:
    return all(isinstance(v, dict) for v in values)


@builder.register("DBFrontend")
@attrs.mutable
class DBFrontend(Frontend):
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
        row_col_keys = {row_key: col_keys for row_key in row_keys}  # type: ignore
        return DBIndex(row_col_keys)

    def convert_read_idx(self, idx_user: UserDBIndex) -> DBIndex:
        return self._convert_idx(idx_user)

    @overload
    def convert_read_data(self, idx_user: str, data: DataT) -> ValueT:
        ...

    @overload
    def convert_read_data(self, idx_user: List[str], data: DataT) -> List[ValueT]:
        ...

    @overload
    def convert_read_data(self, idx_user: Tuple[str, ColIndex], data: DataT) -> RowDataT:
        ...

    @overload
    def convert_read_data(self, idx_user: Tuple[List[str], ColIndex], data: DataT) -> DataT:
        ...

    def convert_read_data(self, idx_user: UserDBIndex, data: DataT):
        if isinstance(idx_user, str):
            return data[0]["value"]

        if isinstance(idx_user, list):
            return [d["value"] for d in data]

        if isinstance(idx_user, tuple):
            row_keys, col_keys = idx_user
            if isinstance(row_keys, str):
                return {col_key: data[0][col_key] for col_key in col_keys}
            return data

        return data

    @overload
    def convert_write(
        self,
        idx_user: str,
        data_user: ValueT,
    ) -> Tuple[DBIndex, DataT]:
        ...

    @overload
    def convert_write(
        self,
        idx_user: List[str],
        data_user: List[ValueT],
    ) -> Tuple[DBIndex, DataT]:
        ...

    @overload
    def convert_write(
        self,
        idx_user: Tuple[str, ColIndex],
        data_user: RowDataT,
    ) -> Tuple[DBIndex, DataT]:
        ...

    @overload
    def convert_write(
        self,
        idx_user: Tuple[List[str], ColIndex],
        data_user: DataT,
    ) -> Tuple[DBIndex, DataT]:
        ...

    def convert_write(
        self,
        idx_user: UserDBIndex,
        data_user: Union[ValueT, MultiValueT, RowDataT, DataT],
    ):
        idx = self._convert_idx(idx_user)
        if isinstance(data_user, (bool, int, float, str)):
            return idx, [{"value": data_user}]

        if isinstance(data_user, dict):
            return idx, [data_user]

        if is_scalar_list(data_user):
            return idx, [{"value": d} for d in data_user]

        if is_rowdata_list(data_user):
            return idx, data_user

        raise ValueError("Unsupported data type.")
