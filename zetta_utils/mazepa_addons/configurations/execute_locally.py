# pylint: disable=too-many-locals
from __future__ import annotations

from contextlib import ExitStack
from typing import Callable, Optional, Union

from typeguard import typechecked

from zetta_utils import builder, log
from zetta_utils.common import (
    ComparablePartial,
    SemaphoreType,
    configure_semaphores,
    setup_persistent_process_pool,
)
from zetta_utils.mazepa import Flow, Task, execute
from zetta_utils.mazepa.autoexecute_task_queue import AutoexecuteTaskQueue
from zetta_utils.mazepa.execution_state import ExecutionState, InMemoryExecutionState

logger = log.get_logger("mazepa")


@typechecked
@builder.register("mazepa.execute_locally")
def execute_locally(
    target: Union[Task, Flow, ExecutionState, ComparablePartial, Callable],
    task_queue: AutoexecuteTaskQueue | None = None,
    max_batch_len: int = 1000,
    batch_gap_sleep_sec: float = 0.5,
    state_constructor: Callable[..., ExecutionState] = InMemoryExecutionState,
    execution_id: Optional[str] = None,
    raise_on_failed_task: bool = True,
    do_dryrun_estimation: bool = True,
    show_progress: bool = True,
    checkpoint: Optional[str] = None,
    checkpoint_interval_sec: Optional[float] = 150,
    raise_on_failed_checkpoint: bool = True,
    num_procs: int = 1,
    semaphores_spec: dict[SemaphoreType, int] | None = None,
):
    with ExitStack() as stack:
        logger.info(
            "Configuring for local execution: "
            "allocating semaphores and persistent process pool as needed."
        )
        stack.enter_context(configure_semaphores(semaphores_spec))
        stack.enter_context(setup_persistent_process_pool(num_procs))
        execute(
            target=target,
            task_queue=task_queue,
            outcome_queue=task_queue,
            max_batch_len=max_batch_len,
            batch_gap_sleep_sec=batch_gap_sleep_sec,
            state_constructor=state_constructor,
            execution_id=execution_id,
            raise_on_failed_task=raise_on_failed_task,
            do_dryrun_estimation=do_dryrun_estimation,
            show_progress=show_progress,
            checkpoint=checkpoint,
            checkpoint_interval_sec=checkpoint_interval_sec,
            raise_on_failed_checkpoint=raise_on_failed_checkpoint,
        )
