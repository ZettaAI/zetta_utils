from typing import Generic, TypeVar

import attrs
from typing_extensions import ParamSpec

from zetta_utils import builder, log, mazepa
from zetta_utils.layer import IndexChunker, LayerIndex

from .chunkable_protocols import ChunkableOperation

logger = log.get_logger("zetta_utils")

IndexT = TypeVar("IndexT", bound=LayerIndex)
P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


@builder.register("ChunkedApplyFlowSchema")
@mazepa.flow_schema_cls
@attrs.mutable
class ChunkedApplyFlowSchema(Generic[P, IndexT, R_co]):
    operation: ChunkableOperation[P, IndexT, R_co]
    chunker: IndexChunker[IndexT]

    def flow(
        self,
        idx: IndexT,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> mazepa.FlowFnReturnType:
        # can't figure out how to force mypy to check this
        assert len(args) == 0
        logger.info(f"Breaking {idx} into chunks with {self.chunker}.")
        idx_chunks = self.chunker(idx)
        tasks = [
            self.operation.make_task(
                idx=idx_chunk,
                **kwargs,
            )
            for idx_chunk in idx_chunks
        ]
        logger.info(f"Submitting {len(tasks)} processing tasks from operation {self.operation}.")
        yield tasks


@builder.register("build_chunked_apply_flow")
def build_chunked_apply_flow(
    operation: ChunkableOperation[P, IndexT, R_co],
    chunker: IndexChunker[IndexT],
    idx: IndexT,
    *args: P.args,
    **kwargs: P.kwargs,
) -> mazepa.Flow:
    flow_schema = ChunkedApplyFlowSchema(
        chunker=chunker,
        operation=operation,
    )
    flow = flow_schema(idx, *args, **kwargs)

    return flow
