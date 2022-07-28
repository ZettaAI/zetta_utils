# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Dict, Any, Tuple
import attrs

from zetta_utils import spec_parser
from zetta_utils import io
from zetta_utils.io.backends.base import IOBackend
from zetta_utils.io.indexes import SetSelectionIndex


@spec_parser.register("LayerSetBackend")
@attrs.mutable()
class LayerSetBackend(IOBackend[SetSelectionIndex]):  # pylint: disable=too-few-public-methods
    layers: Dict[str, io.layers.Layer]

    def _get_layer_selection(self, idx: SetSelectionIndex) -> Tuple[str, ...]:
        if idx.layer_selection is None:
            result = tuple(self.layers.keys())
        else:
            result = idx.layer_selection
        return result

    def read(self, idx: SetSelectionIndex) -> Dict[str, Any]:
        layer_selection = self._get_layer_selection(idx)

        # TODO: can be parallelized
        result = {k: self.layers[k].read(idx.layer_idx) for k in layer_selection}

        return result

    def write(self, idx: SetSelectionIndex, value: Dict[str, Any]):
        layer_selection = self._get_layer_selection(idx)

        # TODO: can be parallelized
        for k in layer_selection:
            self.layers[k].write(idx.layer_idx, value[k])
