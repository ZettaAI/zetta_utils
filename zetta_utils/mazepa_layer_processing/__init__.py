import mazepa
from zetta_utils import builder

from .processor_base import LayerProcessor
from .common_processors import WriteProcessor
from .common_jobs import chunked_job

builder.register("MazepaExecutor")(mazepa.Executor)
builder.register("MazepaLocalExecutionQueue")(mazepa.LocalExecutionQueue)
builder.register("MazepaExecutionMultiQueue")(mazepa.ExecutionMultiQueue)
builder.register("MazepaSQSExecutionQueue")(mazepa.remote_execution_queues.SQSExecutionQueue)
builder.register("run_mazepa_worker")(mazepa.run_worker)
