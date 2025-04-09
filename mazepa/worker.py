from __future__ import annotations

import math
import sys
import time
import traceback
from typing import Any, Callable, Optional

import tenacity

from zetta_utils import builder, log
from zetta_utils.common import RepeatTimer
from zetta_utils.mazepa import constants, exceptions
from zetta_utils.mazepa.exceptions import MazepaCancel, MazepaTimeoutError
from zetta_utils.mazepa.task_outcome import OutcomeReport, TaskOutcome
from zetta_utils.mazepa.transient_errors import (
    MAX_TRANSIENT_RETRIES,
    TRANSIENT_ERROR_CONDITIONS,
)
from zetta_utils.message_queues.base import MessageQueue, ReceivedMessage

from . import Task


class AcceptAllTasks:
    def __call__(self, task: Task):
        return True


logger = log.get_logger("mazepa")


@builder.register("run_worker")
def run_worker(
    task_queue: MessageQueue[Task],
    outcome_queue: MessageQueue[OutcomeReport],
    sleep_sec: float = 4.0,
    max_pull_num: int = 1,
    max_runtime: Optional[float] = None,
    task_filter_fn: Callable[[Task], bool] = AcceptAllTasks(),
    debug: bool = False,
    idle_timeout: int | None = None,
):
    start_time = time.time()
    time_slept = 0.0
    while True:
        try:
            task_msgs = task_queue.pull(max_num=max_pull_num)
        except (exceptions.MazepaException, SystemExit, KeyboardInterrupt) as e:
            raise e  # pragma: no cover
        except Exception as e:  # pylint: disable=broad-except
            # The broad except here is OK because it will be propagated to the outcome
            # queue and reraise the exception
            logger.error("Failed pulling tasks from the queue:")
            logger.exception(e)
            exc_type, exception, tb = sys.exc_info()
            traceback_text = "".join(traceback.format_exception(exc_type, exception, tb))

            outcome = TaskOutcome[Any](
                exception=exception,
                traceback_text=traceback_text,
                execution_sec=0,
                return_value=None,
            )
            outcome_report = OutcomeReport(task_id=constants.UNKNOWN_TASK_ID, outcome=outcome)
            outcome_queue.push([outcome_report])
            raise e

        logger.info(f"Got {len(task_msgs)} tasks.")

        if len(task_msgs) == 0:
            logger.info(f"Sleeping for {sleep_sec} secs.")
            time.sleep(sleep_sec)
            time_slept += sleep_sec
        else:
            logger.info("STARTING: task batch execution.")
            time_slept = 0
            time_start = time.time()
            for msg in task_msgs:
                task = msg.payload
                with log.logging_tag_ctx("task_id", task.id_):
                    with log.logging_tag_ctx("execution_id", task.execution_id):
                        if task_filter_fn(task):
                            ack_task, outcome = process_task_message(msg=msg, debug=debug)
                        else:
                            ack_task = True
                            outcome = TaskOutcome(exception=MazepaCancel())

                        if ack_task:
                            outcome_report = OutcomeReport(
                                task_id=msg.payload.id_, outcome=outcome
                            )
                            outcome_queue.push([outcome_report])
                            msg.acknowledge_fn()

            time_end = time.time()
            logger.info(f"DONE: task batch execution ({time_end - time_start:.2f}sec).")

        if max_runtime is not None and time.time() - start_time > max_runtime:
            break

        if idle_timeout and time_slept > idle_timeout:
            break


def process_task_message(
    msg: ReceivedMessage[Task], debug: bool, handle_exceptions: bool = True
) -> tuple[bool, TaskOutcome]:
    task = msg.payload
    if task.upkeep_settings.perform_upkeep:
        outcome = _run_task_with_upkeep(
            task, msg.extend_lease_fn, debug=debug, handle_exceptions=handle_exceptions
        )
    else:
        outcome = task(debug=debug, handle_exceptions=handle_exceptions)

    finished_processing: bool
    if outcome.exception is None:
        finished_processing = True
    else:
        if msg.approx_receive_count < MAX_TRANSIENT_RETRIES and any(
            e.does_match(outcome.exception) for e in TRANSIENT_ERROR_CONDITIONS
        ):
            logger.debug(f"Task {task.id_} transient error: {outcome.exception}")
            finished_processing = False
        elif isinstance(outcome.exception, MazepaTimeoutError):
            logger.debug(f"Task {task.id_} execution timed out")
            finished_processing = False
        else:
            finished_processing = True

    return finished_processing, outcome


def _run_task_with_upkeep(
    task: Task, extend_lease_fn: Callable, debug: bool, handle_exceptions: bool
) -> TaskOutcome:
    def _perform_upkeep_callbacks():
        assert task.upkeep_settings.interval_sec is not None
        try:
            extend_lease_fn(math.ceil(task.upkeep_settings.interval_sec * 5))
        except tenacity.RetryError as e:  # pragma: no cover
            logger.info(f"Couldn't perform upkeep: {e}")

    assert task.upkeep_settings.interval_sec is not None
    upkeep = RepeatTimer(task.upkeep_settings.interval_sec, _perform_upkeep_callbacks)
    upkeep.start()
    try:
        result = task(debug=debug, handle_exceptions=handle_exceptions)
    except Exception as e:  # pragma: no cover
        raise e from None
    finally:
        upkeep.cancel()

    return result
