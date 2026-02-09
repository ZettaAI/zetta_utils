from __future__ import annotations

import math
import multiprocessing
import os
import sys
import time
import traceback
from typing import Any, Callable, Optional

from zetta_utils import builder, log, try_load_train_inference
from zetta_utils.common import RepeatTimer, monitor_resources, reset_signal_handlers
from zetta_utils.mazepa import constants, exceptions
from zetta_utils.mazepa.exceptions import MazepaCancel, MazepaTimeoutError
from zetta_utils.mazepa.pool_activity import PoolActivityTracker
from zetta_utils.mazepa.task_outcome import OutcomeReport, TaskOutcome
from zetta_utils.mazepa.transient_errors import (
    MAX_TRANSIENT_RETRIES,
    TRANSIENT_ERROR_CONDITIONS,
)
from zetta_utils.mazepa.upkeep_handlers import (
    SQSUpkeepHandlerManager,
    perform_direct_upkeep,
)
from zetta_utils.message_queues.base import MessageQueue, ReceivedMessage
from zetta_utils.message_queues.sqs.utils import SQSReceivedMsg

from . import Task


class DummyBuffer:
    def read(self, data):
        pass

    def write(self, data):
        pass

    def flush(self):
        pass


def redirect_buffers() -> None:  # pragma: no cover
    sys.stdin = DummyBuffer()  # type: ignore
    sys.stdout = DummyBuffer()  # type: ignore
    sys.stderr = DummyBuffer()  # type: ignore


def worker_init(
    log_level: str,
    suppress_logs: bool = False,
    set_start_method: bool = False,
    multiprocessing_start_method: str = "spawn",
    load_train_inference: bool = False,
) -> None:
    """
    Initialize a worker process with proper logging and signal handling.

    Args:
        log_level: Log level string (e.g., "INFO", "DEBUG")
        suppress_logs: If True, redirect stdout/stderr to dummy buffers
        set_start_method: If True, set multiprocessing start method (for worker pools)
        multiprocessing_start_method: The start method to use if set_start_method is True
        load_train_inference: If True, try to load train/inference modules (for worker pools)
    """
    # Reset signal handlers inherited from parent to default behavior
    reset_signal_handlers()
    # For Kubernetes compatibility, ensure unbuffered output
    os.environ["PYTHONUNBUFFERED"] = "1"

    if suppress_logs:
        redirect_buffers()
    else:
        log.configure_logger(level=log_level, force=True)

    if set_start_method:
        multiprocessing.set_start_method(multiprocessing_start_method, force=True)

    if load_train_inference:
        try_load_train_inference()


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
    upkeep_handler: SQSUpkeepHandlerManager | None = None,
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
                    ack_task, outcome = process_task_message(
                        msg=msg, debug=debug, upkeep_handler=upkeep_handler
                    )
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
    # Start SQS upkeep handler process for handling visibility extensions
    upkeep_handler = SQSUpkeepHandlerManager()
    upkeep_handler.start()

    # For single worker case (k8s), create shared memory for activity tracking
    # For multi-worker case, parent already created it - we'll get FileExistsError which is fine
    owns_activity_tracker = False
    if pool_name and idle_timeout is not None:
        activity_tracker = PoolActivityTracker(pool_name)
        try:
            activity_tracker.create_shared_memory().close()
            owns_activity_tracker = True
            logger.info(f"Created activity tracker for pool: {pool_name}")
        except FileExistsError:
            logger.debug(f"Activity tracker already exists for pool: {pool_name}")
    else:
        activity_tracker = None

    try:
        with monitor_resources(resource_monitor_interval):
            start_time = time.time()

            while True:
                task_msgs = _pull_tasks_with_error_handling(
                    task_queue, outcome_queue, max_pull_num
                )

                if len(task_msgs) == 0:
                    _handle_idle_state(sleep_sec, idle_timeout, activity_tracker)
                else:
                    _process_task_batch(
                        task_msgs,
                        task_filter_fn,
                        outcome_queue,
                        activity_tracker,
                        debug,
                        upkeep_handler,
                    )

                should_exit, reason = _check_exit_conditions(
                    start_time, max_runtime, idle_timeout, activity_tracker
                )
                if should_exit:
                    assert reason is not None
                    logger.info(f"Worker exiting: {reason}")
                    return reason
    finally:
        upkeep_handler.shutdown()
        if owns_activity_tracker and activity_tracker:
            activity_tracker.unlink()


def _try_get_sqs_msg(extend_lease_fn: Callable) -> SQSReceivedMsg | None:
    """Try to extract SQSReceivedMsg from extend_lease_fn if available."""
    msg = getattr(extend_lease_fn, "kwargs", {}).get("msg")
    if isinstance(msg, SQSReceivedMsg):
        return msg
    return None


def process_task_message(
    msg: ReceivedMessage[Task],
    debug: bool,
    handle_exceptions: bool = True,
    upkeep_handler: SQSUpkeepHandlerManager | None = None,
) -> tuple[bool, TaskOutcome]:
    task = msg.payload
    if task.upkeep_settings.perform_upkeep:
        sqs_msg = _try_get_sqs_msg(msg.extend_lease_fn)
        outcome = _run_task_with_upkeep(
            task,
            msg.extend_lease_fn,
            sqs_msg=sqs_msg,
            debug=debug,
            handle_exceptions=handle_exceptions,
            upkeep_handler=upkeep_handler,
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
    task: Task,
    extend_lease_fn: Callable,
    sqs_msg: SQSReceivedMsg | None,
    debug: bool,
    handle_exceptions: bool,
    upkeep_handler: SQSUpkeepHandlerManager | None = None,
) -> TaskOutcome:
    task_start_time = time.time()
    assert task.upkeep_settings.interval_sec is not None
    extend_duration = math.ceil(task.upkeep_settings.interval_sec * 10)

    use_process_handler = sqs_msg is not None and upkeep_handler is not None

    if use_process_handler:
        assert sqs_msg is not None
        assert upkeep_handler is not None
        # Handler process manages its own timer - completely isolated from main process GIL
        logger.debug(
            f"UPKEEP: Starting upkeep via handler process: "
            f"interval={task.upkeep_settings.interval_sec}s, extend_by={extend_duration}s"
        )
        upkeep_handler.start_upkeep(
            task_id=task.id_,
            receipt_handle=sqs_msg.receipt_handle,
            visibility_timeout=extend_duration,
            interval_sec=task.upkeep_settings.interval_sec,
            queue_name=sqs_msg.queue_name,
            region_name=sqs_msg.region_name,
            endpoint_url=sqs_msg.endpoint_url,
        )
        try:
            logger.info("Task execution starting")
            result = task(debug=debug, handle_exceptions=handle_exceptions)
            elapsed = time.time() - task_start_time
            logger.info(f"Task execution completed successfully after {elapsed:.2f}s")
        except Exception as e:  # pragma: no cover # pylint: disable=broad-except
            elapsed = time.time() - task_start_time
            logger.error(
                f"Task execution failed with {type(e).__name__}: {e} after {elapsed:.2f}s"
            )
            raise e
        finally:
            logger.debug("UPKEEP: Stopping upkeep via handler process")
            upkeep_handler.stop_upkeep(task.id_)
    else:
        # Fallback: use RepeatTimer in main process for non-SQS queues
        def upkeep_callback():
            perform_direct_upkeep(extend_lease_fn, extend_duration, task_start_time)

        upkeep = RepeatTimer(task.upkeep_settings.interval_sec, upkeep_callback)
        upkeep.start()
        try:
            logger.info("Task execution starting")
            result = task(debug=debug, handle_exceptions=handle_exceptions)
            elapsed = time.time() - task_start_time
            logger.info(f"Task execution completed successfully after {elapsed:.2f}s")
        except Exception as e:  # pragma: no cover # pylint: disable=broad-except
            elapsed = time.time() - task_start_time
            logger.error(
                f"Task execution failed with {type(e).__name__}: {e} after {elapsed:.2f}s"
            )
            raise e
        finally:
            upkeep.cancel()

    return result
