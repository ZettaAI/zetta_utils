from zetta_utils import builder, mazepa

from .operation_protocols import (
    MultiresOpProtocol,
    ComputeFieldOpProtocol,
    ChunkableOpProtocol,
    VolumetricOpProtocol,
)

from .common import (
    ChunkedApplyFlowSchema,
    CallableOperation,
    VolumetricApplyFlowSchema,
    build_chunked_callable_flow_schema,
    build_chunked_apply_flow,
)

from . import annotation_postprocessing

builder.register("mazepa.Executor")(mazepa.Executor)
builder.register("mazepa.execute")(mazepa.execute)
builder.register("mazepa.TaskRouter")(mazepa.TaskRouter)
builder.register("mazepa.AutoexecuteTaskQueue")(mazepa.AutoexecuteTaskQueue)
builder.register("mazepa.run_worker")(mazepa.run_worker)
builder.register("mazepa.sequential_flow")(mazepa.sequential_flow)
builder.register("mazepa.concurrent_flow")(mazepa.concurrent_flow)
