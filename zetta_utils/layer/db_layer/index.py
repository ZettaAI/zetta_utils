# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Mapping

import attrs

from zetta_utils import builder


@builder.register("DBIndex")
@attrs.frozen
class DBIndex:
    row_col_keys: Mapping[str, tuple[str, ...]]
    col_keys: tuple[str, ...] | None = None

    @property
    def row_keys(self) -> list[str]:
        return list(self.row_col_keys.keys())

    @property
    def rows_col_keys(self) -> list[tuple[str, ...]]:
        return list(self.row_col_keys.values())

    def __len__(self) -> int:
        return len(self.row_col_keys)

    def get_size(self):  # pragma: no cover
        return len(self.row_col_keys)
