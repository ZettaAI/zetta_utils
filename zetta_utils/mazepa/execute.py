from __future__ import annotations

import time
from typing import Callable, Iterable, Optional, Union

import attrs

from zetta_utils.log import get_logger

from .execution_queue import ExecutionQueue, LocalExecutionQueue
from .execution_state import ExecutionState, InMemoryExecutionState
from .flows import Flow

logger = get_logger("mazepa")


@attrs.mutable
class Executor:  # pragma: no cover # single statement, pure delegation
    exec_queue: Optional[ExecutionQueue] = None
    batch_gap_sleep_sec: float = 4.0
    purge_at_start: bool = False
    max_batch_len: int = 10000
    state_constructor: Callable[..., ExecutionState] = InMemoryExecutionState

    def __call__(self, target: Union[Flow, Iterable[Flow], ExecutionState]):
        return execute(
            target=target,
            exec_queue=self.exec_queue,
            batch_gap_sleep_sec=self.batch_gap_sleep_sec,
            purge_at_start=self.purge_at_start,
            max_batch_len=self.max_batch_len,
            state_constructor=self.state_constructor,
        )


def execute(
    target: Union[Flow, Iterable[Flow], ExecutionState],
    exec_queue: Optional[ExecutionQueue] = None,
    batch_gap_sleep_sec: float = 4.0,
    purge_at_start: bool = False,
    max_batch_len: int = 10000,
    state_constructor: Callable[..., ExecutionState] = InMemoryExecutionState,
):
    """
    Executes a target until completion using the given execution queue.
    Execution is performed by making an execution state from the target and passing new task
    batches and completed task ids between the state and the execution queue.
    """
    logger.debug("Mazepa execute invoked.")

    if isinstance(target, ExecutionState):
        state = target
        logger.debug(f"Given execution state {state}.")
    else:
        if not isinstance(target, Flow):
            flows = target
        else:
            flows = [target]
        state = state_constructor(ongoing_flows=flows)
        logger.debug(f"Constructed execution state {state}.")

    if exec_queue is None:
        queue = LocalExecutionQueue()  # type: ExecutionQueue
    else:
        queue = exec_queue

    if purge_at_start:
        queue.purge()
        logger.info(f"Purged queue {queue}.")

    logger.debug(f"STARTING: mazepa execution of {target}.")
    while True:
        if len(state.get_ongoing_flow_ids()) == 0:
            logger.debug("No ongoing flows left.")
            break

        task_batch = state.get_task_batch(max_batch_len=max_batch_len)
        logger.debug(f"Got a batch of {len(task_batch)} tasks.")
        queue.push_tasks(task_batch)
        logger.debug("DONE: Pushing tasks to queue.")
        if not isinstance(queue, LocalExecutionQueue):
            logger.debug(f"Sleeping for {batch_gap_sleep_sec} between batches...")
            time.sleep(batch_gap_sleep_sec)
            logger.debug("Awake.")
        logger.debug("Pulling task outcomes...")
        task_outcomes = queue.pull_task_outcomes()
        logger.debug(f"Received {len(task_outcomes)} taks outcomes.")
        logger.debug("STARTING: Updating with taks outcomes.")
        state.update_with_task_outcomes(task_outcomes)
        logger.debug("DONE: Updating with taks outcomes.")

    logger.debug(f"DONE: mazepa execution of {target}.")
