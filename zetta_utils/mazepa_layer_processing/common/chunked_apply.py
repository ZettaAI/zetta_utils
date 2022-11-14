from typing import Generic, TypeVar

import attrs
from typing_extensions import ParamSpec

from zetta_utils import builder, mazepa
from zetta_utils.layer import IndexChunker, LayerIndex
from zetta_utils.log import logger

IndexT = TypeVar("IndexT", bound=LayerIndex)
P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


@builder.register("ChunkedApplyFlow")
@mazepa.flow_type_cls
@attrs.mutable(init=False)
class ChunkedApplyFlow(Generic[IndexT, P, R_co]):
    # TODO: re-enable attrs init after pyright figures this out
    # cc: https://github.com/microsoft/pyright/issues/4183
    def __init__(
        self,
        task_factory: mazepa.TaskFactory[P, R_co],
        chunker: IndexChunker[IndexT],
    ):
        self.task_factory = task_factory
        self.chunker = chunker

    def flow(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> mazepa.FlowFnReturnType:
        # can't figure out how to force mypy to check this
        idx: IndexT
        assert len(args) == 0
        assert "idx" in kwargs
        idx = kwargs["idx"]  # type: ignore
        # task_args = args
        task_kwargs = {k: v for k, v in kwargs.items() if k not in ["idx"]}

        logger.info(f"Breaking {idx} into chunks with {self.chunker}.")
        idx_chunks = self.chunker(idx)
        tasks = [
            self.task_factory.make_task(
                idx=idx_chunk,  # type: ignore
                # *task_args,
                **task_kwargs,  # type: ignore
            )
            for idx_chunk in idx_chunks
        ]
        logger.info(f"Submitting {len(tasks)} processing tasks from factory {self.task_factory}.")
        yield tasks


# from zetta_utils.mazepa.flows import RawFlowType

# def foo(x: RawFlowType):
#    pass

# foo = foo
# foo(ChunkedApplyFlow)
