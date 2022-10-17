from typing import TypeVar, Any, Generic
import attrs
import mazepa
from typeguard import typechecked
from zetta_utils import builder
from zetta_utils.layer import Layer, LayerIndex

IndexT = TypeVar("IndexT", bound=LayerIndex)


@builder.register("WriteProcessor")
@mazepa.task_maker_cls
@typechecked
@attrs.frozen()
class WriteProcessor(Generic[IndexT]):
    def __call__(self, src: Layer[Any, IndexT], dst: Layer[Any, IndexT], idx: IndexT):
        dst[idx] = src[idx]
