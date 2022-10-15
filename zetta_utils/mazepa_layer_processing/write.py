from typing import TypeVar, Any, Dict
import attrs
import mazepa
from typeguard import typechecked
from zetta_utils import builder
from zetta_utils.layer import Layer, LayerIndex
from . import LayerProcessor

IndexT = TypeVar("IndexT", bound=LayerIndex)


@builder.register("WriteProcessor")
@mazepa.task_maker_cls
@typechecked
@attrs.frozen()
class WriteProcessor(LayerProcessor[IndexT]):
    def __call__(self, layers: Dict[str, Layer[Any, IndexT]], idx: IndexT):
        layers["dst"][idx] = layers["src"][idx]
