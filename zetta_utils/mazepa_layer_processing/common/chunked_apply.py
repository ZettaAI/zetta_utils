from typing import TypeVar, Any, Generic, Callable
from typing_extensions import ParamSpec, Concatenate
import attrs
import mazepa
from zetta_utils import builder
from zetta_utils.layer import LayerIndex, IndexChunker
from zetta_utils.log import logger
from . import SimpleCallableTaskFactory

IndexT = TypeVar("IndexT", bound=LayerIndex)
P = ParamSpec("P")

@builder.register("ChunkedApply")
@attrs.mutable
class _ChunkedApply(Generic[IndexT, P]):
    # Want to keep the same call signature as the task factory, so define it as class
    chunker: IndexChunker[IndexT]
    task_factory: mazepa.TaskFactory[Concatenate[IndexT, P], Any]

    def __call__(
        self,
        idx: IndexT,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> mazepa.FlowFnReturnType:
        logger.info(f"Breaking {idx} into chunks with {self.chunker}.")
        idx_chunks = self.chunker(idx)
        tasks = [
            self.task_factory.make_task(
                idx_chunk, # type: ignore # My protocols aren't right. Fixes wellcome.
                *args, **kwargs
            )
            for idx_chunk in idx_chunks
        ]
        logger.info(f"Submitting {len(tasks)} processing tasks from factory {self.task_factory}.")
        yield tasks


ChunkedApply = mazepa.flow_type_cls(_ChunkedApply)


@builder.register("chunked_apply_simple_callable")
def chunked_apply_simple_callable(
    fn: Callable,
    chunker: IndexChunker[IndexT]
) -> ChunkedApply:
    return ChunkedApply(
        chunker=chunker,
        task_factory=SimpleCallableTaskFactory(fn=fn)
    )
