from typing import TypeVar, Any, Dict
import attrs
from typeguard import typechecked
from zetta_utils import builder
from zetta_utils.layer import Layer, LayerIndex
from . import LayerProcessor

IndexT = TypeVar("IndexT", bound=LayerIndex)

@builder.register("CopyLayer")
@typechecked
@attrs.frozen()
class CopyLayer(LayerProcessor[IndexT]):
    def __call__(self, layers: Dict[str, Layer[Any, IndexT]], idx: IndexT):
        layers['dst'][idx] = layers['src'][idx]
