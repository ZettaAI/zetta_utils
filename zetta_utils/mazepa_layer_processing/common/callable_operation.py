from typing import Any, Callable, Generic, TypeVar

import attrs
from typing_extensions import Concatenate, ParamSpec

from zetta_utils import builder, mazepa
from zetta_utils.layer import IndexChunker, Layer, LayerIndex

from . import ChunkedApplyFlowSchema

IndexT = TypeVar("IndexT", bound=LayerIndex)
R = TypeVar("R")
P = ParamSpec("P")


@builder.register("CallableOperation")
@mazepa.taskable_operation_cls
@attrs.mutable
class CallableOperation(Generic[P, IndexT, R]):
    """
    Simple Wrapper that converts a callalbe to a taskable operation.
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


@builder.register("build_chunked_callable_flow_schema")
def build_chunked_callable_flow_schema(
    fn: Callable[P, R], chunker: IndexChunker[IndexT]
) -> ChunkedApplyFlowSchema[Concatenate[Layer[Any, IndexT, R], P], IndexT, None,]:
    operation = CallableOperation[P, IndexT, R](fn=fn)

    return ChunkedApplyFlowSchema[Concatenate[Layer[Any, IndexT, R], P], IndexT, None](
        chunker=chunker,
        operation=operation,  # type: ignore
    )
