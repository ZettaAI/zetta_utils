import mazepa
from zetta_utils import builder

from .processor_base import LayerProcessor
from .write import WriteProcessor
from .chunked_processor import ChunkedProcessor

builder.register("MazepaExecutor")(mazepa.Executor)
builder.register("MazepaLocalExecutionQueue")(mazepa.LocalExecutionQueue)
builder.register("MazepaExecutionMultiQueue")(mazepa.ExecutionMultiQueue)
builder.register("MazepaSQSExecutionQueue")(mazepa.remote_execution_queues.SQSExecutionQueue)
