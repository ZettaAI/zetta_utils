# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Any, Dict, Tuple

import attrs

from zetta_utils import builder

from .. import Backend, Layer
from . import LayerSetIndex


# TODO: type LayerSet
@builder.register("build_layer_set_backend")
@attrs.mutable()
class LayerSetBackend(
    Backend[LayerSetIndex, Dict[str, Any]]
):  # pylint: disable=too-few-public-methods
    layers: Dict[str, Layer]
    name: str = attrs.field(init=False)

    def __attrs_post_init__(self):
        names = ", ".join([l.name for l in self.layers.values()])
        self.name = f"Set({names})"

    def _get_layer_selection(self, idx: LayerSetIndex) -> Tuple[str, ...]:
        if idx.layer_selection is None:
            result = tuple(self.layers.keys())
        else:
            result = idx.layer_selection

        return result

    def read(self, idx: LayerSetIndex) -> Dict[str, Any]:
        layer_selection = self._get_layer_selection(idx)

        # TODO: can be parallelized
        result = {k: self.layers[k].read(idx.layer_idx) for k in layer_selection}
        return result

    def write(self, idx: LayerSetIndex, data: Dict[str, Any]):
        layer_selection = self._get_layer_selection(idx)

        # TODO: can be parallelized
        for k in layer_selection:
            self.layers[k].write(idx.layer_idx, data[k])

    def with_changes(self, **kwargs) -> LayerSetBackend:  # pragma: no cover
        """Changes the backends for all layers with the parameters in a
        dictionary of dictionaries"""
        for k in kwargs:
            if k not in self.layers:
                raise KeyError(f"key `{k}` not found in the LayerSet")
        new_layers: Dict[str, Layer] = {}
        for k in self.layers:
            if k in kwargs:
                new_layers[k] = self.layers[k].with_backend_changes(**(kwargs[k]))
            else:
                new_layers[k] = self.layers[k].with_backend_changes()
        res = attrs.evolve(self, layer=new_layers)
        return res
