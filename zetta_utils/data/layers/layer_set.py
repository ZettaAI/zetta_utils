# pylint: disable=missing-docstring
"""Volumetric layers are referenced by [(MIP), z, x, y] and
support a `data_mip` parameter which can allow reading
raw data from a different MIP, followed by (up/down)sampling."""
from __future__ import annotations

from zetta_utils.data.layers.common import Layer


class LayerSet(Layer):
    def __init__(self, layers: Dict[str, Layer]):
        self.layers = layers

    def _read(self, idx):
        pass

    def _write(self, idx, value):
        pass
