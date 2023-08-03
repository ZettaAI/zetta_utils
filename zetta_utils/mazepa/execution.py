# pylint: disable=too-many-locals
from __future__ import annotations

import time
from contextlib import ExitStack
from datetime import datetime
from typing import Callable, Optional, Union

import attrs
from pebble import ProcessPool
from typeguard import typechecked

from zetta_utils import log
from zetta_utils.common import ComparablePartial
from zetta_utils.mazepa.autoexecute_task_queue import AutoexecuteTaskQueue
from zetta_utils.message_queues.base import PullMessageQueue, PushMessageQueue

from . import Flow, Task, dryrun, sequential_flow
from .execution_checkpoint import record_execution_checkpoint
from .execution_state import ExecutionState, InMemoryExecutionState
from .id_generation import get_unique_id
from .progress_tracker import progress_ctx_mngr
from .task_outcome import OutcomeReport, TaskStatus
from .tasks import _TaskableOperation

logger = log.get_logger("mazepa")


@attrs.mutable
class Executor:
    task_queue: PushMessageQueue[Task] | None = None
    outcome_queue: PullMessageQueue[OutcomeReport] | None = None
    batch_gap_sleep_sec: float = 4.0
    max_batch_len: int = 1000
    state_constructor: Callable[..., ExecutionState] = InMemoryExecutionState
    raise_on_failed_task: bool = True
    do_dryrun_estimation: bool = True
    show_progress: bool = True
    checkpoint: Optional[str] = None
    checkpoint_interval_sec: Optional[float] = None
    raise_on_failed_checkpoint: bool = True

    def __call__(self, target: Union[Task, Flow, ExecutionState, ComparablePartial, Callable]):
        assert (self.task_queue is None and self.outcome_queue is None) or (
            self.task_queue is not None and self.outcome_queue is not None
        )
        return execute(
            target=target,
            task_queue=self.task_queue,
            outcome_queue=self.outcome_queue,
            batch_gap_sleep_sec=self.batch_gap_sleep_sec,
            max_batch_len=self.max_batch_len,
            state_constructor=self.state_constructor,
            raise_on_failed_task=self.raise_on_failed_task,
            show_progress=self.show_progress,
            do_dryrun_estimation=self.do_dryrun_estimation,
            checkpoint=self.checkpoint,
            checkpoint_interval_sec=self.checkpoint_interval_sec,
            raise_on_failed_checkpoint=self.raise_on_failed_checkpoint,
        )


@typechecked
def execute(
    target: Union[Task, Flow, ExecutionState, ComparablePartial, Callable],
    task_queue: PushMessageQueue[Task] | None = None,
    outcome_queue: PullMessageQueue[OutcomeReport] | None = None,
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
):
    """
    Executes a target until completion using the given execution queue.
    Execution is performed by making an execution state from the target and passing new task
    batches and completed task ids between the state and the execution queue.

    Implementation: this function performs misc setup and delegates to _execute_from_state.
    """
    if execution_id is None:
        execution_id_final = get_unique_id(
            prefix="default-exec", slug_len=4, add_uuid=False, max_len=50
        )
    else:
        execution_id_final = execution_id

    with log.logging_tag_ctx("execution_id", execution_id_final):
        logger.debug(f"Starting execution '{execution_id_final}'")

        if isinstance(target, ExecutionState):
            state = target
            logger.debug(f"Loaded execution state {state}.")
        else:
            if isinstance(target, Task):
                flows = [sequential_flow([target])]
            elif isinstance(target, Flow):
                flows = [target]
            else:  # isinstance(target, (ComparablePartial, Callable)):
                task = _TaskableOperation(fn=target).make_task()
                flows = [sequential_flow([task])]

            state = state_constructor(
                ongoing_flows=flows,
                raise_on_failed_task=raise_on_failed_task,
                checkpoint=checkpoint,
            )
            logger.debug(f"Built initial execution state {state}.")

        if task_queue is not None:
            assert outcome_queue is not None
            task_queue_ = task_queue
            outcome_queue_ = outcome_queue
        else:
            assert outcome_queue is None
            task_queue_ = AutoexecuteTaskQueue(debug=True)
            outcome_queue_ = task_queue_

        logger.debug(f"STARTING: execution of {target}.")
        start_time = time.time()

        _execute_from_state(
            execution_id=execution_id_final,
            state=state,
            task_queue=task_queue_,
            outcome_queue=outcome_queue_,
            max_batch_len=max_batch_len,
            batch_gap_sleep_sec=batch_gap_sleep_sec,
            do_dryrun_estimation=do_dryrun_estimation,
            show_progress=show_progress,
            checkpoint_interval_sec=checkpoint_interval_sec,
            raise_on_failed_checkpoint=raise_on_failed_checkpoint,
        )

        end_time = time.time()
        logger.debug(f"DONE: mazepa execution of {target}.")
        logger.info(f"Total execution time: {end_time - start_time:.1f}secs")


def _execute_from_state(
    execution_id: str,
    state: ExecutionState,
    task_queue: PushMessageQueue[Task],
    outcome_queue: PullMessageQueue[OutcomeReport],
    max_batch_len: int,
    batch_gap_sleep_sec: float,
    do_dryrun_estimation: bool,
    show_progress: bool,
    checkpoint_interval_sec: Optional[float],
    raise_on_failed_checkpoint: bool,
):
    if do_dryrun_estimation:
        expected_operation_counts = dryrun.get_expected_operation_counts(state.get_ongoing_flows())
    else:
        expected_operation_counts = {}

    last_backup_ts = time.time()

    with ExitStack() as stack:
        if show_progress:
            progress_updater = stack.enter_context(progress_ctx_mngr(expected_operation_counts))
        else:
            progress_updater = lambda *args, **kwargs: None

        while True:
            progress_updater(state.get_progress_reports())
            if len(state.get_ongoing_flow_ids()) == 0:
                logger.debug("No ongoing flows left.")
                break

            submit_ready_tasks(task_queue, outcome_queue, state, execution_id, max_batch_len)

            if not isinstance(task_queue, AutoexecuteTaskQueue):
                logger.debug(f"Sleeping for {batch_gap_sleep_sec} between batches...")
                time.sleep(batch_gap_sleep_sec)
                logger.debug("Awake.")

            if (
                checkpoint_interval_sec is not None
                and time.time() > last_backup_ts + checkpoint_interval_sec
            ):
                backup_completed_tasks(
                    state, execution_id, raise_on_error=raise_on_failed_checkpoint
                )
                last_backup_ts = time.time()


def backup_completed_tasks(state: ExecutionState, execution_id: str, raise_on_error: bool = False):
    completed_ids = list(state.get_completed_ids())
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ckpt_name = f"{timestamp}_{len(completed_ids)}"
    record_execution_checkpoint(
        execution_id, ckpt_name, completed_ids, raise_on_error=raise_on_error
    )


def submit_ready_tasks(
    task_queue: PushMessageQueue[Task],
    outcome_queue: PullMessageQueue[OutcomeReport],
    state: ExecutionState,
    execution_id: str,
    max_batch_len: int,
    num_procs: int = 10,
):
    logger.debug("Pulling task outcomes...")
    task_outcomes = outcome_queue.pull(max_num=100)

    if len(task_outcomes) > 0:
        logger.debug(f"Received {len(task_outcomes)} completed task outcomes.")
        logger.debug("Updating execution state with taks outcomes.")
        state.update_with_task_outcomes(
            task_outcomes={e.payload.task_id: e.payload.outcome for e in task_outcomes}
        )

        # it's not important when to acknowledge outcomes, since
        # only the single manager node will be pulling from the
        # outcome queue
        with ProcessPool(max_workers=num_procs) as pool:
            for e in task_outcomes:
                pool.schedule(e.acknowledge_fn)
    logger.debug("Getting next ready task batch.")
    task_batch = state.get_task_batch(max_batch_len=max_batch_len)
    logger.debug(f"A batch of {len(task_batch)} tasks ready for execution.")

    for task in task_batch:
        task.execution_id = execution_id
    logger.debug("Pushing task batch to queue.")
    task_queue.push(task_batch)
    for task in task_batch:
        task.status = TaskStatus.SUBMITTED
