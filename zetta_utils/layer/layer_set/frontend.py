# pylint: disable=missing-docstring,no-self-use
from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import attrs

from ..frontend_base import Frontend
from . import LayerSetIndex

AllLayerSetIndex = Tuple[Any, ...]
SpecificLayerSetIndex = Tuple[Tuple[str, ...], Any]
UnconvertedLayerSetIndex = Union[
    SpecificLayerSetIndex,
    AllLayerSetIndex,
]
UserLayerSetIndex = Union[UnconvertedLayerSetIndex, LayerSetIndex]


@attrs.frozen
class LayerSetFrontend(Frontend):
    def convert_read_idx(
        self, idx_user: UserLayerSetIndex  # pylint: disable=unused-argument
    ) -> LayerSetIndex:
        return self._convert_idx(idx_user)

    def convert_read_data(
        self,
        idx_user: UserLayerSetIndex,  # pylint: disable=unused-argument
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return data

    def convert_write(
        self,
        idx_user: UserLayerSetIndex,  # pylint: disable=unused-argument
        data_user: Dict[str, Any],
    ) -> Tuple[LayerSetIndex, Dict[str, Any]]:
        return self._convert_idx(idx_user), data_user

    def _convert_idx(
        self,
        idx_raw: UserLayerSetIndex,
    ) -> LayerSetIndex:
        # Check if the first element is a suitable layer name spec
        # If so, assume that's what it is a set selection.
        if isinstance(idx_raw, LayerSetIndex):
            result = idx_raw
        else:
            if isinstance(idx_raw[0], tuple) and all(isinstance(e, str) for e in idx_raw[0]):
                layer_selection = idx_raw[0]
                if len(idx_raw) == 2:
                    layer_idx = idx_raw[1]
                else:
                    layer_idx = idx_raw[1:]
            else:
                layer_selection = None
                layer_idx = idx_raw

            result = LayerSetIndex(
                layer_selection=layer_selection,
                layer_idx=layer_idx,
            )

        return result
