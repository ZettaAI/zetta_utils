import collections
from typing import Callable, Generic, TypeVar

import attrs
from typing_extensions import Concatenate, ParamSpec

from zetta_utils import builder, mazepa
from zetta_utils.layer import IndexChunker, Layer
from zetta_utils.layer.protocols import LayerWithIndexT

from . import ChunkedApplyFlowSchema

IndexT = TypeVar("IndexT")
R = TypeVar("R")
P = ParamSpec("P")


def _process_callable_kwargs(idx: IndexT, kwargs: dict) -> dict:
    result = {}

    for k, v in kwargs.items():
        if isinstance(v, Layer):
            result[k] = v[idx]
        elif isinstance(v, dict) and all(isinstance(vv, Layer) for vv in v.values()):
            result[k] = {kk: vv[idx] for kk, vv in v.items()}
        elif isinstance(v, collections.abc.Iterable) and all(isinstance(vv, Layer) for vv in v):
            result[k] = [vv[idx] for vv in v]
        else:
            result[k] = v
    return result


@builder.register("CallableOperation")
@mazepa.taskable_operation_cls
@attrs.mutable
class CallableOperation(Generic[P, IndexT, R]):
    """
    Simple Wrapper that converts a callable to a taskable operation.
    """

    fn: Callable[P, R]

    def __call__(
        self, idx: IndexT, dst: LayerWithIndexT[IndexT], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        assert len(args) == 0
        fn_kwargs = _process_callable_kwargs(idx, kwargs)
        result = self.fn(**fn_kwargs)
        dst[idx] = result


@builder.register("build_chunked_callable_flow_schema")
def build_chunked_callable_flow_schema(
    fn: Callable[P, R], chunker: IndexChunker[IndexT]
) -> ChunkedApplyFlowSchema[Concatenate[LayerWithIndexT[IndexT], P], IndexT, None,]:
    operation = CallableOperation[P, IndexT, R](fn=fn)

    return ChunkedApplyFlowSchema[Concatenate[LayerWithIndexT[IndexT], P], IndexT, None](
        chunker=chunker,
        operation=operation,  # type: ignore
    )
