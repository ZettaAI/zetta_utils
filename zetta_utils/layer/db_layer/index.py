# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Dict, Iterator, Tuple

import attrs

from zetta_utils import builder


@builder.register("DBIndex")
@attrs.frozen
class DBIndex:  # pragma: no cover
    row_col_keys: Dict[str, Tuple[str]]

    @property
    def row_keys_count(self) -> int:
        return len(self.row_col_keys)

    @property
    def row_keys(self) -> Iterator[str]:
        return iter(self.row_col_keys.keys())

    @property
    def col_keys(self) -> Iterator[Tuple[str]]:
        return iter(self.row_col_keys.values())

    def get_size(self):  # pragma: no cover
        return len(self.row_col_keys)
