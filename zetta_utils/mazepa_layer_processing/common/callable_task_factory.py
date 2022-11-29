from typing import Any, Callable, Generic, TypeVar

import attrs
from typing_extensions import Concatenate, ParamSpec

from zetta_utils import builder, mazepa
from zetta_utils.layer import IndexChunker, Layer, LayerIndex

from . import ChunkedApplyFlowType

IndexT = TypeVar("IndexT", bound=LayerIndex)
R = TypeVar("R")
P = ParamSpec("P")


@builder.register("CallableTaskFactory")
@mazepa.task_factory_cls
@attrs.mutable
class CallableTaskFactory(Generic[P, IndexT, R]):
    """
    Simple Wrapper that converts a callalbe to a task factory by.
    """

    fn: Callable[P, R]

    def __call__(
        self, idx: IndexT, dst: Layer[Any, IndexT, R], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        assert len(args) == 0
        fn_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, Layer):
                fn_kwargs[k] = v[idx]
            else:
                fn_kwargs[k] = v

        result = self.fn(**fn_kwargs)
        dst[idx] = result


@builder.register("build_chunked_callable_flow_type")
def build_chunked_callable_flow_type(
    fn: Callable[P, R], chunker: IndexChunker[IndexT]
) -> ChunkedApplyFlowType[Concatenate[Layer[Any, IndexT, R], P], IndexT, None,]:
    factory = CallableTaskFactory[P, IndexT, R](fn=fn)

    return ChunkedApplyFlowType[Concatenate[Layer[Any, IndexT, R], P], IndexT, None](
        chunker=chunker,
        task_factory=factory,  # type: ignore
    )
