# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Tuple, Optional, Union, Any

import attrs
from typeguard import typechecked

from zetta_utils.io.indexes.base import (
    Index,
)

RawSetSelectionIndex = Union[
    Tuple[None, Any],
    Tuple[str, Any],
    Tuple[Tuple[str, ...], Any],
]


@typechecked
@attrs.frozen
class SetSelectionIndex(Index):  # pylint: disable=too-few-public-methods
    layer_selection: Optional[Tuple[str, ...]]
    layer_idx: Any

    @classmethod
    def convert(
        cls,
        idx_raw: RawSetSelectionIndex,
    ):
        selection_raw = idx_raw[0]
        layer_idx = idx_raw[1]

        if isinstance(selection_raw, tuple):
            layer_selection = selection_raw
        elif selection_raw is not None:
            layer_selection = (selection_raw,)
        else:
            layer_selection = None

        result = cls(
            layer_selection=layer_selection,
            layer_idx=layer_idx,
        )

        return result
