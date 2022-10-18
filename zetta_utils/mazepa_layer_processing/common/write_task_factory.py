from typing import TypeVar, Any, Generic
import attrs
import mazepa
from zetta_utils import builder
from zetta_utils.layer import Layer, LayerIndex

IndexT = TypeVar("IndexT", bound=LayerIndex)


@builder.register("WriteTaskFactory")
@mazepa.task_factory_cls
@attrs.frozen()
class _WriteTaskFactory(Generic[IndexT]):
    def __call__(self, idx: IndexT, src: Layer[Any, IndexT], dst: Layer[Any, IndexT]):
        dst[idx] = src[idx]

WriteTaskFactory = mazepa.task_factory_cls(_WriteTaskFactory)
