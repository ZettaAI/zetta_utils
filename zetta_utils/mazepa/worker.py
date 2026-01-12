from __future__ import annotations

import ctypes
import math
import sys
import threading
import time
import traceback
from typing import Any, Callable, Optional

from zetta_utils import builder, log
from zetta_utils.common import RepeatTimer, monitor_resources
from zetta_utils.mazepa import constants, exceptions
from zetta_utils.mazepa.exceptions import MazepaCancel, MazepaTimeoutError
from zetta_utils.mazepa.pool_activity import PoolActivityTracker
from zetta_utils.mazepa.task_outcome import OutcomeReport, TaskOutcome
from zetta_utils.mazepa.transient_errors import (
    MAX_TRANSIENT_RETRIES,
    TRANSIENT_ERROR_CONDITIONS,
    ExplicitTransientError,
)
from zetta_utils.message_queues.base import MessageQueue, ReceivedMessage

from . import Task


class AcceptAllTasks:
    def __call__(self, task: Task):
        return True


logger = log.get_logger("mazepa")


def _pull_tasks_with_error_handling(
    task_queue: MessageQueue[Task],
    outcome_queue: MessageQueue[OutcomeReport],
    max_pull_num: int,
) -> list[ReceivedMessage[Task]]:
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
    return task_msgs


def _handle_idle_state(
    sleep_sec: float,
    idle_timeout: float | None,
    activity_tracker: PoolActivityTracker | None,
) -> None:
    if activity_tracker:
        last_activity, active_count = activity_tracker.get_activity_data()
        time_since_activity = time.time() - last_activity
        idle_line = (
            "Idle timeout not set."
            if idle_timeout is None
            else f"Idle timeout {idle_timeout:.1f}s."
        )
        logger.info(
            f"Sleeping {sleep_sec}s. Idle: {time_since_activity:.1f}s, "
            f"Active workers: {active_count}. {idle_line}"
        )
    else:
        logger.info(f"Sleeping {sleep_sec}s.")
    time.sleep(sleep_sec)


def _process_task_batch(
    task_msgs: list[ReceivedMessage[Task]],
    task_filter_fn: Callable[[Task], bool],
    outcome_queue: MessageQueue[OutcomeReport],
    activity_tracker: PoolActivityTracker | None,
    debug: bool,
) -> None:
    logger.info("STARTING: task batch execution.")

    # Update activity time and increment active worker count
    if activity_tracker:
        activity_tracker.update_activity_time()
        activity_tracker.increment_active_workers()

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
                    outcome_report = OutcomeReport(task_id=msg.payload.id_, outcome=outcome)
                    outcome_queue.push([outcome_report])
                    msg.acknowledge_fn()

    time_end = time.time()
    logger.info(f"DONE: task batch execution ({time_end - time_start:.2f}sec).")

    # Update activity time on completion and decrement active worker count
    if activity_tracker:
        activity_tracker.update_activity_time()
        activity_tracker.decrement_active_workers()


def _check_exit_conditions(
    start_time: float,
    max_runtime: float | None,
    idle_timeout: float | None,
    activity_tracker: PoolActivityTracker | None,
) -> tuple[bool, str | None]:
    if max_runtime is not None and time.time() - start_time > max_runtime:
        return True, "max_runtime_exceeded"

    if idle_timeout and activity_tracker:
        if activity_tracker.check_idle_timeout(idle_timeout):
            return True, "idle_timeout_exceeded"

    return False, None


@builder.register("run_worker")
def run_worker(
    task_queue: MessageQueue[Task],
    outcome_queue: MessageQueue[OutcomeReport],
    sleep_sec: float = 4.0,
    max_pull_num: int = 1,
    max_runtime: Optional[float] = None,
    task_filter_fn: Callable[[Task], bool] = AcceptAllTasks(),
    resource_monitor_interval: float | None = None,
    debug: bool = False,
    idle_timeout: float | None = None,
    pool_name: str | None = None,
) -> str:
    with monitor_resources(resource_monitor_interval):
        start_time = time.time()
        activity_tracker = PoolActivityTracker(pool_name) if pool_name else None

        while True:
            task_msgs = _pull_tasks_with_error_handling(task_queue, outcome_queue, max_pull_num)

            if len(task_msgs) == 0:
                _handle_idle_state(sleep_sec, idle_timeout, activity_tracker)
            else:
                _process_task_batch(
                    task_msgs, task_filter_fn, outcome_queue, activity_tracker, debug
                )

            should_exit, reason = _check_exit_conditions(
                start_time, max_runtime, idle_timeout, activity_tracker
            )
            if should_exit:
                assert reason is not None
                logger.info(f"Worker exiting: {reason}")
                return reason


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


def _raise_exception_in_thread(thread_id: int, exception_type: type[BaseException]):
    """Raise an exception in another thread using ctypes."""
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id), ctypes.py_object(exception_type)
    )
    if ret == 0:
        raise ValueError(f"Invalid thread id: {thread_id}")
    if ret > 1:  # pragma: no cover
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)
        raise SystemError("Exception raise affected multiple threads")


def _run_task_with_upkeep(
    task: Task, extend_lease_fn: Callable, debug: bool, handle_exceptions: bool
) -> TaskOutcome:
    main_thread_id = threading.current_thread().ident
    assert main_thread_id is not None

    def _perform_upkeep_callbacks():
        assert task.upkeep_settings.interval_sec is not None
        try:
            extend_lease_fn(math.ceil(task.upkeep_settings.interval_sec * 10))
            print("Upkeep successful.")
            logger.info("Upkeep successful.")
        except Exception as e:  # pragma: no cover # pylint: disable=broad-except
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            logger.error(f"Couldn't perform upkeep: {e}\n{tb_str}")
            _raise_exception_in_thread(main_thread_id, ExplicitTransientError)

    assert task.upkeep_settings.interval_sec is not None
    upkeep = RepeatTimer(task.upkeep_settings.interval_sec, _perform_upkeep_callbacks)
    upkeep.start()
    try:
        result = task(debug=debug, handle_exceptions=handle_exceptions)
    except Exception as e:  # pragma: no cover # pylint: disable=broad-except
        raise e
    finally:
        upkeep.cancel()

    return result
