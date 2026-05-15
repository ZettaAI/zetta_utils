"""Mazepa layer processing exports — lazily resolved via zetta_utils.common.lazy."""
from typing import TYPE_CHECKING

from zetta_utils import builder, mazepa
from zetta_utils.common.lazy import make_lazy_module

_LAZY_SUBPACKAGES = ("annotation_postprocessing",)

_LAZY_REEXPORTS = {
    ".operation_protocols": (
        "MultiresOpProtocol",
        "ComputeFieldOpProtocol",
        "ChunkableOpProtocol",
        "VolumetricOpProtocol",
        "StackableVolumetricOpProtocol",
    ),
    ".common": (
        "ChunkedApplyFlowSchema",
        "CallableOperation",
        "VolumetricApplyFlowSchema",
        "build_chunked_callable_flow_schema",
        "build_chunked_apply_flow",
    ),
}

__getattr__, __dir__ = make_lazy_module(
    __name__, globals(), _LAZY_SUBPACKAGES, _LAZY_REEXPORTS
)

if TYPE_CHECKING:
    from . import annotation_postprocessing
    from .common import (
        CallableOperation,
        ChunkedApplyFlowSchema,
        VolumetricApplyFlowSchema,
        build_chunked_apply_flow,
        build_chunked_callable_flow_schema,
    )
    from .operation_protocols import (
        ChunkableOpProtocol,
        ComputeFieldOpProtocol,
        MultiresOpProtocol,
        StackableVolumetricOpProtocol,
        VolumetricOpProtocol,
    )

# Builder registrations for mazepa primitives must fire at package-load time
# so the names are discoverable to the static index scanner.
builder.register("mazepa.Executor")(mazepa.Executor)
builder.register("mazepa.execute")(mazepa.execute)  # type: ignore[has-type,unused-ignore]
builder.register("mazepa.TaskRouter")(mazepa.TaskRouter)
builder.register("mazepa.AutoexecuteTaskQueue")(mazepa.AutoexecuteTaskQueue)
builder.register("mazepa.sequential_flow")(mazepa.sequential_flow)
builder.register("mazepa.concurrent_flow")(mazepa.concurrent_flow)
