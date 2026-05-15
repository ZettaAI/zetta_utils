"""Mazepa subpackage exports — lazily resolved via zetta_utils.common.lazy."""
from typing import TYPE_CHECKING

from zetta_utils.common.lazy import make_lazy_module

_LAZY_SUBPACKAGES = ("constants", "exceptions", "dryrun")

_LAZY_REEXPORTS = {
    ".task_outcome": ("TaskOutcome", "TaskStatus"),
    ".transient_errors": ("TransientErrorCondition",),
    ".tasks": (
        "Task",
        "TaskableOperation",
        "taskable_operation",
        "taskable_operation_cls",
    ),
    ".flows": (
        "Flow",
        "FlowFnReturnType",
        "FlowSchema",
        "flow_schema",
        "flow_schema_cls",
        "Dependency",
        "sequential_flow",
        "concurrent_flow",
    ),
    ".task_router": ("TaskRouter",),
    ".autoexecute_task_queue": ("AutoexecuteTaskQueue",),
    ".execution_state": ("ExecutionState", "InMemoryExecutionState"),
    ".progress_tracker": ("progress_ctx_mngr",),
    ".execution": ("Executor", "execute"),
    ".worker": ("run_worker",),
    ".semaphores": ("SemaphoreType", "configure_semaphores", "semaphore"),
}

__getattr__, __dir__ = make_lazy_module(
    __name__, globals(), _LAZY_SUBPACKAGES, _LAZY_REEXPORTS
)

if TYPE_CHECKING:
    # Static-analysis re-exports (invisible at runtime). Mirrors
    # _LAZY_REEXPORTS so pylint / mypy / IDE autocomplete see the public API.
    from . import constants, dryrun, exceptions
    from .autoexecute_task_queue import AutoexecuteTaskQueue
    from .execution import Executor, execute
    from .execution_state import ExecutionState, InMemoryExecutionState
    from .flows import (
        Dependency,
        Flow,
        FlowFnReturnType,
        FlowSchema,
        concurrent_flow,
        flow_schema,
        flow_schema_cls,
        sequential_flow,
    )
    from .progress_tracker import progress_ctx_mngr
    from .semaphores import SemaphoreType, configure_semaphores, semaphore
    from .task_outcome import TaskOutcome, TaskStatus
    from .task_router import TaskRouter
    from .tasks import Task, TaskableOperation, taskable_operation, taskable_operation_cls
    from .transient_errors import TransientErrorCondition
    from .worker import run_worker
