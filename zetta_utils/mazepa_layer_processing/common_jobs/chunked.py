from typing import TypeVar, Dict, Any, List, Generator
import mazepa
from typeguard import typechecked
from zetta_utils import builder
from zetta_utils.layer import Layer, LayerIndex, IndexChunker
from zetta_utils.log import logger

from .. import LayerProcessor

IndexT = TypeVar("IndexT", bound=LayerIndex)


@builder.register("mazepa_chunked_job")
@mazepa.job
@typechecked
def chunked_job(
    layers: Dict[str, Layer[Any, IndexT]],
    idx: IndexT,
    processor: LayerProcessor,
    chunker: IndexChunker[IndexT],
) -> Generator[List[mazepa.Task], None, Any]:
    idx_chunks = chunker(idx)
    logger.info(f"Breaking {idx} into chunks with {chunker}.")
    tasks = [processor.make_task(layers=layers, idx=idx_chunk) for idx_chunk in idx_chunks]
    logger.info(f"Submitting {len(tasks)} processing tasks of type {processor}.")
    yield tasks
