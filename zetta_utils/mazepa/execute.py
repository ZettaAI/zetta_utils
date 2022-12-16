# pylint: disable=too-many-locals
from __future__ import annotations

import time
from typing import Callable, Optional, Union

import attrs

from zetta_utils.common import ComparablePartial
from zetta_utils.log import get_logger

from . import Flow, Task, ctx_vars, seq_flow
from .execution_queue import ExecutionQueue, LocalExecutionQueue
from .execution_state import ExecutionState, InMemoryExecutionState
from .id_generation import get_unique_id
from .task_outcome import TaskStatus
from .tasks import _TaskableOperation

logger = get_logger("mazepa")


@attrs.mutable
class Executor:  # pragma: no cover # single statement, pure delegation
    exec_queue: Optional[ExecutionQueue] = None
    batch_gap_sleep_sec: float = 4.0
    max_batch_len: int = 10000
    state_constructor: Callable[..., ExecutionState] = InMemoryExecutionState
    upkeep_fn: Optional[Callable[[str], bool]] = None
    raise_on_failed_task: bool = True

    def __call__(self, target: Union[Task, Flow, ExecutionState, ComparablePartial, Callable]):
        return execute(
            target=target,
            exec_queue=self.exec_queue,
            batch_gap_sleep_sec=self.batch_gap_sleep_sec,
            max_batch_len=self.max_batch_len,
            state_constructor=self.state_constructor,
            upkeep_fn=self.upkeep_fn,
            raise_on_failed_task=self.raise_on_failed_task,
        )


def execute(
    target: Union[Task, Flow, ExecutionState, ComparablePartial, Callable],
    exec_queue: Optional[ExecutionQueue] = None,
    max_batch_len: int = 10000,
    batch_gap_sleep_sec: float = 4.0,
    state_constructor: Callable[..., ExecutionState] = InMemoryExecutionState,
    upkeep_fn: Optional[Callable[[str], bool]] = None,
    execution_id: Optional[str] = None,
    raise_on_failed_task: bool = True,
    max_task_retry: int = 1,
):
    """
    Executes a target until completion using the given execution queue.
    Execution is performed by making an execution state from the target and passing new task
    batches and completed task ids between the state and the execution queue.

    Implementation: this function performs misc setup and delegates to _execute_from_state.
    """
    if execution_id is None:
        execution_id_final = get_unique_id(prefix="execution")
    else:
        execution_id_final = execution_id

    ctx_vars.execution_id.set(execution_id_final)
    logger.debug(f"Starting execution '{execution_id_final}'")

    if isinstance(target, ExecutionState):
        state = target
        logger.debug(f"Loaded execution state {state}.")
    else:
        if isinstance(target, Task):
            flows = [seq_flow([target])]
        elif isinstance(target, Flow):
            flows = [target]
        else:  # isinstance(target, (ComparablePartial, Callable)):
            task = _TaskableOperation(
                fn=target,
                time_bound=False,
                max_retry=max_task_retry,
            ).make_task()
            flows = [seq_flow([task])]

        state = state_constructor(ongoing_flows=flows, raise_on_failed_task=raise_on_failed_task)
        logger.debug(f"Built initial execution state {state}.")

    if exec_queue is not None:
        exec_queue_built = exec_queue
    else:
        exec_queue_built = LocalExecutionQueue()

    logger.debug(f"STARTING: execution of {target}.")
    start_time = time.time()

    _execute_from_state(
        execution_id=execution_id_final,
        state=state,
        exec_queue=exec_queue_built,
        max_batch_len=max_batch_len,
        batch_gap_sleep_sec=batch_gap_sleep_sec,
        upkeep_fn=upkeep_fn,
    )

    end_time = time.time()
    logger.debug(f"DONE: mazepa execution of {target}.")
    logger.info(f"Total execution time: {end_time - start_time:.1f}secs")


def _execute_from_state(
    execution_id: str,
    state: ExecutionState,
    exec_queue: ExecutionQueue,
    max_batch_len: int,
    batch_gap_sleep_sec: float,
    upkeep_fn: Optional[Callable[[str], bool]],
):
    while True:
        if len(state.get_ongoing_flow_ids()) == 0:
            logger.debug("No ongoing flows left.")
            break

        execution_should_continue = True
        if upkeep_fn is not None:
            execution_should_continue = upkeep_fn(execution_id)

        if not execution_should_continue:
            logger.debug(f"Stopping execution by decision of {upkeep_fn}")
            break

        process_ready_tasks(exec_queue, state, execution_id, max_batch_len)

        if not isinstance(exec_queue, LocalExecutionQueue):
            logger.debug(f"Sleeping for {batch_gap_sleep_sec} between batches...")
            time.sleep(batch_gap_sleep_sec)
            logger.debug("Awake.")


def process_ready_tasks(
    queue: ExecutionQueue, state: ExecutionState, execution_id: str, max_batch_len: int
):
    logger.debug("Pulling task outcomes...")
    task_outcomes = queue.pull_task_outcomes()
    if len(task_outcomes) > 0:
        logger.info(f"Received {len(task_outcomes)} completed task outcomes.")
        logger.debug("Updating execution state with taks outcomes.")
        state.update_with_task_outcomes(task_outcomes)

    logger.debug("Getting next ready task batch.")
    task_batch = state.get_task_batch(max_batch_len=max_batch_len)
    logger.debug(f"A batch of {len(task_batch)} tasks ready for execution.")

    for task in task_batch:
        task.execution_id = execution_id
    logger.debug("Pushing task batch to queue.")
    queue.push_tasks(task_batch)
    for task in task_batch:
        task.status = TaskStatus.SUBMITTED
