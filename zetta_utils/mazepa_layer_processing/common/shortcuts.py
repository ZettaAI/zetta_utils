from typing import Any, Callable, TypeVar

from typing_extensions import ParamSpec

from zetta_utils import builder, mazepa
from zetta_utils.layer import IndexChunker, Layer, LayerIndex

from . import ChunkedApplyFlow, SimpleCallableTaskFactory

IndexT = TypeVar("IndexT", bound=LayerIndex)
P = ParamSpec("P")


@builder.register("build_chunked_apply_flow")
def build_chunked_apply_flow(
    task_factory: mazepa.TaskFactory[P, Any],
    chunker: IndexChunker[IndexT],
    *args: P.args,
    **kwargs: P.kwargs,
) -> mazepa.Flow:
    flow_type = ChunkedApplyFlow[IndexT, P, None](
        chunker=chunker,
        task_factory=task_factory,
    )
    flow = flow_type(*args, **kwargs)

    return flow


@builder.register("build_chunked_apply_callable_flow_type")
def build_chunked_apply_callable_flow_type(
    fn: Callable[P, Any], chunker: IndexChunker[IndexT]
) -> ChunkedApplyFlow[IndexT, P, None]:
    factory = SimpleCallableTaskFactory[P](fn=fn)
    return ChunkedApplyFlow[IndexT, P, None](
        chunker=chunker,
        task_factory=factory,
    )


def _write_callable(src_data):
    return src_data


@builder.register("build_chunked_write_flow")
def build_chunked_write_flow_type(
    chunker: IndexChunker[IndexT],
) -> ChunkedApplyFlow:
    return build_chunked_apply_callable_flow_type(
        fn=_write_callable,
        chunker=chunker,
    )


@builder.register("chunked_write")
def chunked_write(
    chunker: IndexChunker[IndexT],
    idx: IndexT,
    dst: Layer[Any, IndexT, Any],
    src: Layer[Any, IndexT, Any],
) -> mazepa.Flow:
    result = build_chunked_write_flow_type(chunker=chunker)(idx=idx, dst=dst, src=src)
    return result
