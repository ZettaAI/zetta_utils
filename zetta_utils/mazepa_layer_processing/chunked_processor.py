from typing import TypeVar, Dict, Any, Generator, List, Generic
import attrs
import mazepa
from typeguard import typechecked
from zetta_utils import builder
from zetta_utils.layer import Layer, LayerIndex, IndexChunker, IdentityIndexChunker
from zetta_utils.log import logger

from . import LayerProcessor

IndexT = TypeVar("IndexT", bound=LayerIndex)


@mazepa.job
@typechecked
def _chunked_processing_job(
    layers: Dict[str, Layer[Any, IndexT]],
    idx: IndexT,
    chunker: IndexChunker[IndexT],
    processor: LayerProcessor,
) -> Generator[List[mazepa.Task], None, Any]:
    idx_chunks = chunker(idx)
    logger.info(f"Breaking {idx} into chunks with {chunker}.")
    tasks = [processor.make_task(layers=layers, idx=idx_chunk) for idx_chunk in idx_chunks]
    logger.info(f"Submitting {len(tasks)} processing tasks of type {processor}.")
    yield tasks


@builder.register("ChunkedProcessor")
@typechecked
@mazepa.task_maker_cls
@attrs.frozen()
class ChunkedProcessor(LayerProcessor, Generic[IndexT]):
    inner_processor: LayerProcessor
    chunker: IndexChunker = IdentityIndexChunker()
    executor: mazepa.Executor = mazepa.Executor()

    def __call__(
        self,
        layers: Dict[str, Layer[Any, IndexT]],
        idx: IndexT,
    ):
        job = _chunked_processing_job(
            layers=layers, idx=idx, chunker=self.chunker, processor=self.inner_processor
        )
        self.executor(job)
