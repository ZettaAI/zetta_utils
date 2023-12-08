# pylint: disable=too-many-locals
from __future__ import annotations

import os
from contextlib import ExitStack
from typing import Callable, Optional, Union

from typeguard import typechecked

from zetta_utils import builder, log
from zetta_utils.common import ComparablePartial
from zetta_utils.mazepa import Flow, SemaphoreType, Task, configure_semaphores, execute
from zetta_utils.mazepa.execution_state import ExecutionState, InMemoryExecutionState
from zetta_utils.message_queues import FileQueue

from .worker_pool import setup_local_worker_pool

logger = log.get_logger("mazepa")


@typechecked
@builder.register("mazepa.execute_locally")
def execute_locally(
    target: Union[Task, Flow, ExecutionState, ComparablePartial, Callable],
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
    debug: bool = False,
):

    with ExitStack() as stack:
        logger.info(
            "Configuring for local execution: "
            "creating local queues, allocating semaphores, and starting local workers."
        )
        stack.enter_context(configure_semaphores(semaphores_spec))

        if debug:
            logger.info("Debug mode: Using single process execution without local queues.")
            task_queue = None
            outcome_queue = None
        else:
            task_queue_name = f"{os.getpid()}_task_queue"
            outcome_queue_name = f"{os.getpid()}_outcome_queue"
            task_queue = stack.enter_context(FileQueue(task_queue_name))
            outcome_queue = stack.enter_context(FileQueue(outcome_queue_name))
            stack.enter_context(
                setup_local_worker_pool(num_procs, task_queue_name, outcome_queue_name)
            )
        execute(
            target=target,
            task_queue=task_queue,
            outcome_queue=outcome_queue,
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
