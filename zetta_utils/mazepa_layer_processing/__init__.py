from zetta_utils import builder, mazepa

from .operation_protocols import (
    MultiresOpProtocol,
    ComputeFieldOpProtocol,
    ChunkableOpProtocol,
    VolumetricOpProtocol,
)
from . import alignment
from . import segmentation
from .common import (
    ChunkedApplyFlowSchema,
    CallableOperation,
    VolumetricApplyFlowSchema,
    build_chunked_callable_flow_schema,
    build_chunked_apply_flow,
)

builder.register("mazepa.Executor")(mazepa.Executor)
builder.register("mazepa.execute")(mazepa.execute)
builder.register("mazepa.LocalExecutionQueue")(mazepa.LocalExecutionQueue)
builder.register("mazepa.ExecutionMultiQueue")(mazepa.ExecutionMultiQueue)
builder.register("mazepa.SQSExecutionQueue")(mazepa.remote_execution_queues.SQSExecutionQueue)
builder.register("mazepa.run_worker")(mazepa.run_worker)
builder.register("mazepa.SubflowTask")(mazepa.SubflowTask)
builder.register("mazepa.seq_flow")(mazepa.seq_flow)
builder.register("mazepa.concurrent_flow")(mazepa.concurrent_flow)
