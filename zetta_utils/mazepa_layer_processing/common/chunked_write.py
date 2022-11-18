from typing import Any, TypeVar

from zetta_utils import builder, mazepa
from zetta_utils.layer import IndexChunker, Layer, LayerIndex

from . import build_chunked_apply_callable_flow_type

IndexT = TypeVar("IndexT", bound=LayerIndex)


def _write_callable(src_data):
    return src_data


@builder.register("chunked_write")
def chunked_write(
    chunker: IndexChunker[IndexT],
    idx: IndexT,
    dst: Layer[Any, IndexT, Any],
    src: Layer[Any, IndexT, Any],
) -> mazepa.Flow:

    flow_type = build_chunked_apply_callable_flow_type(
        fn=_write_callable,
        chunker=chunker,
    )
    result = flow_type(idx=idx, dst=dst, src=src)
    return result
