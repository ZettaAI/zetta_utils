# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Any, Dict, Tuple

import attrs

from zetta_utils import builder

from .. import Backend, Layer
from . import LayerSetIndex


# TODO: type LayerSet
@builder.register("build_layer_setBackend")
@attrs.mutable()
class LayerSetBackend(
    Backend[LayerSetIndex, Dict[str, Any]]
):  # pylint: disable=too-few-public-methods
    layer: Dict[str, Layer]

    def _get_layer_selection(self, idx: LayerSetIndex) -> Tuple[str, ...]:
        if idx.layer_selection is None:
            result = tuple(self.layer.keys())
        else:
            result = idx.layer_selection

        return result

    def read(self, idx: LayerSetIndex) -> Dict[str, Any]:
        layer_selection = self._get_layer_selection(idx)

        # TODO: can be parallelized
        result = {k: self.layer[k].read(idx.layer_idx) for k in layer_selection}
        return result

    def write(self, idx: LayerSetIndex, data: Dict[str, Any]):
        layer_selection = self._get_layer_selection(idx)

        # TODO: can be parallelized
        for k in layer_selection:
            self.layer[k].write(idx.layer_idx, data[k])

    def get_name(self) -> str:  # pragma: no cover
        return ", ".join([l.get_name() for l in self.layer.values()])
