# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Dict, Any
import attrs

from zetta_utils import spec_parser
from zetta_utils import io
from zetta_utils.io.backends.base import IOBackend
from zetta_utils.io.indexes import SetSelectionIndex


@spec_parser.register("LayerSetBackend")
@attrs.mutable()
class LayerSetBackend(IOBackend[SetSelectionIndex]):  # pylint: disable=too-few-public-methods
    layers: Dict[str, io.layers.Layer]

    def read(self, idx: SetSelectionIndex) -> Dict[str, Any]:
        if idx.selected_layers is None:
            selected_layers = tuple(self.layers.keys())
        else:
            selected_layers = idx.selected_layers

        # TODO: can be parallelized
        result = {k: self.layers[k].read(idx.layer_idx) for k in selected_layers}

        return result

    def write(self, idx: SetSelectionIndex, value: Dict[str, Any]):
        if idx.selected_layers is None:
            selected_layers = tuple(self.layers.keys())
        else:
            selected_layers = idx.selected_layers

        # TODO: can be parallelized
        for k in selected_layers:
            self.layers[k][idx.layer_idx] = value[k]
