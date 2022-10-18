from typing import TypeVar, Any, Generic, Callable
import attrs
import mazepa
from zetta_utils import builder
from zetta_utils.layer import Layer, LayerIndex

IndexT = TypeVar("IndexT", bound=LayerIndex)


@builder.register("SimpleCallableTaskFactory")
@mazepa.task_factory_cls
@attrs.frozen()
class _SimpleCallableTaskFactory(Generic[IndexT]):
    """
    Naive Wrapper that converts a callalbe to a task.
    No type checking will be performed on the callable.
    """
    fn: Callable
    # download_layers: bool = True # Could be made optoinal

    def __call__(self, idx: IndexT, dst: Layer[Any, IndexT], **kwargs):
        processed_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, Layer):
                processed_kwargs[f"{k}_data"] = v[idx]
            else:
                processed_kwargs[k] = v

        result = self.fn(**processed_kwargs)
        dst[idx] = result

SimpleCallableTaskFactory = mazepa.task_factory_cls(_SimpleCallableTaskFactory)

