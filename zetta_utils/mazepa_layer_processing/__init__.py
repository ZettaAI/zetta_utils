import mazepa
from zetta_utils import builder

from .common import WriteTaskFactory, SimpleCallableTaskFactory, ChunkedApply, chunked_apply_simple_processor

builder.register("Executor")(mazepa.Executor)
builder.register("LocalExecutionQueue")(mazepa.LocalExecutionQueue)
builder.register("ExecutionMultiQueue")(mazepa.ExecutionMultiQueue)
builder.register("SQSExecutionQueue")(mazepa.remote_execution_queues.SQSExecutionQueue)
builder.register("run_worker")(mazepa.run_worker)
builder.register("SubflowTask")(mazepa.SubflowTask)
