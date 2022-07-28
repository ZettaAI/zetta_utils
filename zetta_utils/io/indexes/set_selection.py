# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Tuple, Optional, Union

import attrs
from typeguard import typechecked

from zetta_utils.io.indexes.base import (
    Index,
)

RawSetIndex = Union[
    Index,
    Tuple[None, Index],
    Tuple[str, Index],
    Tuple[Tuple[str], Index],
]


@typechecked
@attrs.frozen
class SetSelectionIndex(Index):  # pylint: disable=too-few-public-methods
    selected_layers: Optional[Tuple[str]]
    layer_idx: Index

    @classmethod
    def convert(
        cls,
        idx_raw: RawSetIndex,
    ):
        if isinstance(idx_raw, tuple):
            selection_raw = idx_raw[0]
            layer_idx = idx_raw[1]

            if isinstance(selection_raw, tuple):
                selected_layers = selection_raw
            elif selection_raw is not None:
                selected_layers = (selection_raw,)
            else:
                selected_layers = None

        result = cls(
            selected_layers=selected_layers,
            layer_idx=layer_idx,
        )

        return result
