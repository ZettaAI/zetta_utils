from zetta_utils import mazepa
from zetta_utils import builder

from .common import (
    SimpleCallableTaskFactory,
    ChunkedApplyFlow,
    build_chunked_apply_callable_flow_type,
    #    ChunkedApplySimpleCallableFlow,
)
from . import alignment

builder.register("MazepaExecutor")(mazepa.Executor)
builder.register("mazepa_execute")(mazepa.execute)
builder.register("MazepaLocalExecutionQueue")(mazepa.LocalExecutionQueue)
builder.register("MazepaExecutionMultiQueue")(mazepa.ExecutionMultiQueue)
builder.register("MazepaSQSExecutionQueue")(mazepa.remote_execution_queues.SQSExecutionQueue)
builder.register("mazepa_run_worker")(mazepa.run_worker)
builder.register("MazepaSubflowTask")(mazepa.SubflowTask)
