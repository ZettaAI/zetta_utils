# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Tuple, Optional, Union, Any

import attrs
from typeguard import typechecked

from zetta_utils.io.indexes.base import (
    Index,
)

RawSetSelectionIndex = Union[
    Tuple[Tuple[str, ...], Any],  # Specificeation
    Tuple[Any, ...],  # All Layers
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
        # Check if the first element is a suitable layer name spec
        # If so, assume that's what it's meant to be
        if isinstance(idx_raw[0], tuple) and all(isinstance(e, str) for e in idx_raw[0]):
            layer_selection = idx_raw[0]
            layer_idx = idx_raw[1:]
        else:
            layer_selection = None
            layer_idx = idx_raw

        result = cls(
            layer_selection=layer_selection,
            layer_idx=layer_idx,
        )

        return result
