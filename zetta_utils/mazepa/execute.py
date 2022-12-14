from __future__ import annotations

import time
from contextlib import AbstractContextManager, ExitStack
from typing import Callable, Iterable, Optional, Union

import attrs

from zetta_utils.log import get_logger

from . import ExecutionCtxManager, ctx_vars
from .execution_queue import ExecutionQueue, LocalExecutionQueue
from .execution_state import ExecutionState, InMemoryExecutionState
from .flows import Flow
from .id_generation import get_unique_id
from .task_outcome import TaskStatus

logger = get_logger("mazepa")


@attrs.mutable
class Executor:  # pragma: no cover # single statement, pure delegation
    exec_queue: Optional[ExecutionQueue] = None
    batch_gap_sleep_sec: float = 4.0
    max_batch_len: int = 10000
    state_constructor: Callable[..., ExecutionState] = InMemoryExecutionState
    upkeep_fn: Optional[Callable[[str], bool]] = None

    def __call__(self, target: Union[Flow, Iterable[Flow], ExecutionState]):
        return execute(
            target=target,
            exec_queue=self.exec_queue,
            batch_gap_sleep_sec=self.batch_gap_sleep_sec,
            max_batch_len=self.max_batch_len,
            state_constructor=self.state_constructor,
            upkeep_fn=self.upkeep_fn,
        )


def execute(
    target: Union[Flow, Iterable[Flow], ExecutionState],
    exec_queue: Optional[ExecutionQueue] = None,
    max_batch_len: int = 10000,
    batch_gap_sleep_sec: float = 4.0,
    state_constructor: Callable[..., ExecutionState] = InMemoryExecutionState,
    upkeep_fn: Optional[Callable[[str], bool]] = None,
    ctx_managers: Iterable[Union[AbstractContextManager, ExecutionCtxManager]] = (),
):
    """
    Executes a target until completion using the given execution queue.
    Execution is performed by making an execution state from the target and passing new task
    batches and completed task ids between the state and the execution queue.

    Implementation: this function performs misc setup and delegates to _execute_from_state.
    """
    execution_id = get_unique_id(prefix="execution")
    ctx_vars.execution_id.set(execution_id)
    logger.debug(f"Starting execution '{execution_id}'")

    if isinstance(target, ExecutionState):
        state = target
        logger.debug(f"Given execution state {state}.")
    else:
        state = state_constructor(ongoing_flows=[target])
        logger.debug(f"Constructed execution state {state}.")

    if exec_queue is not None:
        exec_queue_built = exec_queue
    else:
        exec_queue_built = LocalExecutionQueue()

    logger.debug(f"STARTING: execution of {target}.")
    start_time = time.time()

    with ExitStack() as stack:
        for mgr in ctx_managers:
            if isinstance(mgr, ExecutionCtxManager):
                stack.enter_context(mgr(execution_id=execution_id))
            else:
                stack.enter_context(mgr)

        _execute_from_state(
            execution_id=execution_id,
            state=state,
            exec_queue=exec_queue_built,
            max_batch_len=max_batch_len,
            batch_gap_sleep_sec=batch_gap_sleep_sec,
            upkeep_fn=upkeep_fn,
        )

    end_time = time.time()
    logger.debug(f"DONE: mazepa execution of {target}.")
    logger.debug(f"Total execution time: {end_time - start_time:.1f}secs")


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
    logger.debug(f"Received {len(task_outcomes)} taks outcomes.")
    logger.debug("Updating execution state with taks outcomes.")
    state.update_with_task_outcomes(task_outcomes)

    logger.debug("Getting next ready task batch.")
    task_batch = state.get_task_batch(max_batch_len=max_batch_len)
    logger.debug(f"Got a batch of {len(task_batch)} tasks.")

    for task in task_batch:
        task.execution_id = execution_id
    logger.debug("Pushing task batch to queue.")
    queue.push_tasks(task_batch)
    for task in task_batch:
        task.status = TaskStatus.SUBMITTED
