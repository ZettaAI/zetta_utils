from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable

import tenacity

from zetta_utils import log

logger = log.get_logger("mazepa")


def perform_direct_upkeep(
    extend_lease_fn: Callable,
    extend_duration: int,
    task_start_time: float,
) -> None:
    """
    Perform upkeep by directly calling extend_lease_fn.

    Used as a fallback for non-SQS queues where the process-based handler
    cannot be used.
    """
    current_time = time.time()
    elapsed_since_start = current_time - task_start_time
    logger.debug(
        f"UPKEEP: [T+{elapsed_since_start:.1f}s] Timer fired, calling extend_lease_fn directly"
    )
    try:
        start_time = time.time()
        extend_lease_fn(extend_duration)
        api_duration = time.time() - start_time
        logger.debug(
            f"UPKEEP: [T+{elapsed_since_start:.1f}s] Successfully extended lease by "
            f"{extend_duration}s (API call took {api_duration:.1f}s)"
        )
    except tenacity.RetryError as e:  # pragma: no cover
        logger.error(f"UPKEEP: Failed to extend lease after retries: {e}")
    except Exception as e:  # pragma: no cover  # pylint: disable=broad-except
        logger.error(f"UPKEEP: Unexpected error: {type(e).__name__}: {e}")


@dataclass
class UpkeepCommand:
    """Command sent to the SQS upkeep handler process."""

    action: str  # "start_upkeep", "stop_upkeep", or "shutdown"
    # Required for start_upkeep
    task_id: str | None = None
    receipt_handle: str | None = None
    visibility_timeout: int | None = None
    interval_sec: float | None = None
    queue_name: str | None = None
    region_name: str | None = None
    endpoint_url: str | None = None


def _is_parent_alive(parent_pid: int) -> bool:  # pragma: no cover
    """Check if the parent process is still alive."""
    try:
        os.kill(parent_pid, 0)  # Signal 0 just checks if process exists
        return True
    except OSError:
        return False


def run_sqs_upkeep_handler(  # pylint: disable=too-many-statements
    command_queue: multiprocessing.Queue,
    log_level: str = "INFO",
    parent_pid: int | None = None,
    interval_sec: float = 5.0,
) -> None:
    """
    Main loop for the SQS upkeep handler process.

    Runs in a separate process to handle SQS visibility extensions. This isolates
    SQS operations from the main worker process's GIL, ensuring that heavy CPU work
    in the main process doesn't delay upkeep operations.

    The handler manages its own timer, so timing is not affected by the main
    process's CPU usage.

    If parent_pid is provided, the handler will exit if the parent process dies.
    This prevents orphaned handler processes when workers are force-killed.
    """
    # pylint: disable=import-outside-toplevel
    from zetta_utils.mazepa.worker import worker_init
    from zetta_utils.message_queues.sqs import utils

    # Initialize the process (logging, signal handlers, etc.)
    # Don't set start method or load train/inference for the upkeep handler
    worker_init(log_level=log_level)

    logger.info(
        "SQS_HANDLER: Upkeep handler process started (PID: %d, parent: %s)",
        multiprocessing.current_process().pid,
        parent_pid,
    )

    # Track active upkeep tasks: task_id -> (stop_event, thread)
    active_upkeeps: dict[str, tuple[threading.Event, threading.Thread]] = {}

    def _run_upkeep_loop(
        task_id: str,
        stop_event: threading.Event,
        receipt_handle: str,
        visibility_timeout: int,
        interval_sec: float,
        queue_name: str,
        region_name: str,
        endpoint_url: str | None,
    ):
        """Timer loop that extends visibility at regular intervals."""
        task_start_time = time.time()
        logger.info(
            f"SQS_HANDLER: [{task_id}] Starting upkeep loop: interval={interval_sec}s, "
            f"extend_by={visibility_timeout}s"
        )

        while not stop_event.wait(timeout=interval_sec):
            elapsed = time.time() - task_start_time
            logger.debug(
                f"SQS_HANDLER: [{task_id}] [T+{elapsed:.1f}s] Extending visibility to "
                f"{visibility_timeout}s for queue '{queue_name}'"
            )
            try:
                api_start = time.time()
                utils.change_message_visibility(
                    receipt_handle=receipt_handle,
                    visibility_timeout=visibility_timeout,
                    queue_name=queue_name,
                    region_name=region_name,
                    endpoint_url=endpoint_url,
                )
                api_duration = time.time() - api_start
                logger.debug(
                    f"SQS_HANDLER: [{task_id}] [T+{elapsed:.1f}s] Successfully extended "
                    f"(API took {api_duration:.1f}s)"
                )
            except Exception as e:  # pylint: disable=broad-except # pragma: no cover
                logger.error(
                    f"SQS_HANDLER: [{task_id}] Failed to extend visibility: "
                    f"{type(e).__name__}: {e}"
                )

        elapsed = time.time() - task_start_time
        logger.info(f"SQS_HANDLER: [{task_id}] Upkeep loop stopped after {elapsed:.1f}s")

    while True:
        try:
            # Use timeout so we can periodically check if parent is alive
            try:
                cmd = command_queue.get(timeout=interval_sec)
            except:  # pylint: disable=bare-except # pragma: no cover
                # Check if parent is still alive
                if parent_pid is not None and not _is_parent_alive(parent_pid):
                    logger.warning(
                        "SQS_HANDLER: Parent process %d died, exiting handler", parent_pid
                    )
                    # Stop all active upkeeps before exiting
                    for task_id, (stop_event, thread) in active_upkeeps.items():
                        stop_event.set()
                        thread.join(timeout=1.0)
                    break
                continue

            if cmd.action == "shutdown":
                logger.info("SQS_HANDLER: Received shutdown command")
                # Stop all active upkeeps
                for task_id, (stop_event, thread) in active_upkeeps.items():
                    logger.info(f"SQS_HANDLER: Stopping upkeep for task {task_id}")
                    stop_event.set()
                    thread.join(timeout=2.0)
                break

            if cmd.action == "start_upkeep":
                if cmd.task_id in active_upkeeps:
                    logger.warning(
                        f"SQS_HANDLER: Upkeep already active for task {cmd.task_id}, ignoring"
                    )
                    continue

                stop_event = threading.Event()
                thread = threading.Thread(
                    target=_run_upkeep_loop,
                    args=(
                        cmd.task_id,
                        stop_event,
                        cmd.receipt_handle,
                        cmd.visibility_timeout,
                        cmd.interval_sec,
                        cmd.queue_name,
                        cmd.region_name,
                        cmd.endpoint_url,
                    ),
                    daemon=True,
                    name=f"upkeep-{cmd.task_id[:8]}",
                )
                thread.start()
                active_upkeeps[cmd.task_id] = (stop_event, thread)
                logger.info(f"SQS_HANDLER: Started upkeep for task {cmd.task_id}")

            elif cmd.action == "stop_upkeep":
                if cmd.task_id not in active_upkeeps:
                    logger.warning(
                        f"SQS_HANDLER: No active upkeep for task {cmd.task_id}, ignoring"
                    )
                    continue

                stop_event, thread = active_upkeeps.pop(cmd.task_id)
                stop_event.set()
                thread.join(timeout=2.0)
                logger.info(f"SQS_HANDLER: Stopped upkeep for task {cmd.task_id}")

            else:
                logger.warning(f"SQS_HANDLER: Unknown action: {cmd.action}")

        except Exception as e:  # pylint: disable=broad-except # pragma: no cover
            logger.error(f"SQS_HANDLER: Error processing command: {type(e).__name__}: {e}")

    logger.info("SQS_HANDLER: Handler process exiting")


class SQSUpkeepHandlerManager:
    """
    Manages the lifecycle of an SQS upkeep handler process.

    The handler process manages its own timers for visibility extensions,
    completely isolated from the main process's GIL.

    Usage:
        manager = SQSUpkeepHandlerManager()
        manager.start()
        try:
            manager.start_upkeep(task_id, ...)  # Handler starts its own timer
            # ... task runs ...
            manager.stop_upkeep(task_id)  # Handler stops the timer
        finally:
            manager.shutdown()
    """

    def __init__(self):
        self._command_queue: multiprocessing.Queue | None = None
        self._handler_process: multiprocessing.Process | None = None

    def start(self) -> None:
        """Start the handler process."""
        if self._command_queue is not None:
            return  # Already running

        # Get current log level to pass to handler process
        current_log_level = logging.getLevelName(logging.getLogger("mazepa").getEffectiveLevel())

        # Pass parent PID so handler can detect if parent dies
        parent_pid = os.getpid()

        self._command_queue = multiprocessing.Queue()
        self._handler_process = multiprocessing.Process(
            target=run_sqs_upkeep_handler,
            args=(self._command_queue, current_log_level, parent_pid),
            daemon=True,
            name="sqs-upkeep-handler",
        )
        self._handler_process.start()
        logger.info(f"Started SQS upkeep handler process (PID: {self._handler_process.pid})")

    def shutdown(self, timeout: float = 10.0) -> None:
        """Shutdown the handler process gracefully."""
        if self._command_queue is None:
            return  # Not running

        logger.info("Shutting down SQS upkeep handler process...")
        self._command_queue.put(UpkeepCommand(action="shutdown"))

        if self._handler_process is not None:
            self._handler_process.join(timeout=timeout)
            if self._handler_process.is_alive():
                logger.warning("Handler process did not stop gracefully, terminating...")
                self._handler_process.terminate()
                self._handler_process.join(timeout=1.0)

        self._command_queue = None
        self._handler_process = None
        logger.info("SQS upkeep handler process stopped.")

    def start_upkeep(
        self,
        task_id: str,
        receipt_handle: str,
        visibility_timeout: int,
        interval_sec: float,
        queue_name: str,
        region_name: str,
        endpoint_url: str | None = None,
    ) -> None:
        """
        Start upkeep for a task. The handler process will manage its own timer
        and extend visibility at regular intervals.
        """
        if self._command_queue is None:
            logger.warning("SQS_HANDLER: Handler not running, start_upkeep ignored")
            return

        cmd = UpkeepCommand(
            action="start_upkeep",
            task_id=task_id,
            receipt_handle=receipt_handle,
            visibility_timeout=visibility_timeout,
            interval_sec=interval_sec,
            queue_name=queue_name,
            region_name=region_name,
            endpoint_url=endpoint_url,
        )
        self._command_queue.put_nowait(cmd)

    def stop_upkeep(self, task_id: str) -> None:
        """Stop upkeep for a task."""
        if self._command_queue is None:
            logger.warning("SQS_HANDLER: Handler not running, stop_upkeep ignored")
            return

        self._command_queue.put_nowait(UpkeepCommand(action="stop_upkeep", task_id=task_id))

    @property
    def is_running(self) -> bool:
        """Check if the handler process is running."""
        return self._handler_process is not None and self._handler_process.is_alive()
