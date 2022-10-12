# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Dict, Any, Tuple
import attrs

from zetta_utils import builder
from . import SetSelectionIndex
from .. import LayerBackend, Layer


@builder.register("LayerSetBackend")
@attrs.mutable()
class LayerSetBackend(LayerBackend[SetSelectionIndex]):  # pylint: disable=too-few-public-methods
    layer: Dict[str, Layer]

    def _get_layer_selection(self, idx: SetSelectionIndex) -> Tuple[str, ...]:
        if idx.layer_selection is None:
            result = tuple(self.layer.keys())
        else:
            result = idx.layer_selection

        return result

    def read(self, idx: SetSelectionIndex) -> Dict[str, Any]:
        layer_selection = self._get_layer_selection(idx)

        # TODO: can be parallelized
        result = {k: self.layer[k].read(idx.layer_idx) for k in layer_selection}
        return result

    def write(self, idx: SetSelectionIndex, value: Dict[str, Any]):
        layer_selection = self._get_layer_selection(idx)

        # TODO: can be parallelized
        for k in layer_selection:
            self.layer[k].write(idx.layer_idx, value[k])
