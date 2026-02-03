from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import sys
import time
from contextlib import ExitStack

import pebble

from zetta_utils import builder, get_mp_context, log, setup_environment
from zetta_utils.common import monitor_resources, reset_signal_handlers
from zetta_utils.mazepa import SemaphoreType, Task, configure_semaphores, run_worker
from zetta_utils.mazepa.pool_activity import PoolActivityTracker
from zetta_utils.mazepa.task_outcome import OutcomeReport
from zetta_utils.message_queues import FileQueue, SQSQueue

logger = log.get_logger("mazepa")


class DummyBuffer:
    def read(self, data):
        pass

    def write(self, data):
        pass

    def flush(self):
        pass


def redirect_buffers() -> None:  # Do not need to implement 14 passes for typing.FileIO
    sys.stdin = DummyBuffer()  # type: ignore
    sys.stdout = DummyBuffer()  # type: ignore
    sys.stderr = DummyBuffer()  # type: ignore


def worker_init(
    suppress_worker_logs: bool, multiprocessing_start_method: str, log_level: str
) -> None:
    # Reset signal handlers inherited from parent to default behavior
    # This prevents parent's signal handlers from interfering with worker cleanup
    reset_signal_handlers()
    # For Kubernetes compatibility, ensure unbuffered output
    os.environ["PYTHONUNBUFFERED"] = "1"
    if suppress_worker_logs:
        redirect_buffers()
    else:
        # Reconfigure logging in worker process with parent's log level
        log.configure_logger(level=log_level, force=True)

    # Inherit the start method from the calling process
    multiprocessing.set_start_method(multiprocessing_start_method, force=True)
    setup_environment("try")


def run_local_worker(
    task_queue_name: str,
    outcome_queue_name: str,
    local: bool = True,
    sleep_sec: float = 0.1,
    idle_timeout: int | None = None,
    pool_name: str | None = None,
) -> str:
    logger.info("Creating a local worker in this process....")
    queue_type = FileQueue if local else SQSQueue
    task_queue = queue_type(name=task_queue_name)
    outcome_queue = queue_type(name=outcome_queue_name, pull_wait_sec=1.0)
    exit_reason = run_worker(
        task_queue=task_queue,
        outcome_queue=outcome_queue,
        sleep_sec=sleep_sec,
        max_pull_num=1,
        idle_timeout=idle_timeout,
        pool_name=pool_name,
    )
    logger.info(f"Local worker returned: {exit_reason}")
    return exit_reason


@contextlib.contextmanager
def setup_local_worker_pool(
    num_procs: int,
    task_queue_name: str,
    outcome_queue_name: str,
    local: bool,
    sleep_sec: float,
    idle_timeout: int | None,
    suppress_worker_logs: bool,
    resource_monitor_interval: float | None,
):
    """
    Context manager for creating task/outcome queues, alongside a persistent pool of workers.
    Note that worker pools will inherit the current process' multiprocessing start method.
    """
    with monitor_resources(resource_monitor_interval):
        # Create pool activity tracker for coordinated idle timeout
        pool_name = f"{task_queue_name}_{outcome_queue_name}"
        activity_tracker = None
        if idle_timeout is not None:
            activity_tracker = PoolActivityTracker(pool_name)
            activity_tracker.create_shared_memory().close()
            logger.info(f"Created pool activity tracker for idle timeout management: {pool_name}")

        try:
            current_log_level = logging.getLevelName(
                logging.getLogger("mazepa").getEffectiveLevel()
            )
            pool = pebble.ProcessPool(
                max_workers=num_procs,
                context=get_mp_context(),
                initializer=worker_init,
                initargs=[
                    suppress_worker_logs,
                    multiprocessing.get_start_method(),
                    current_log_level,
                ],
            )
            try:
                futures = []
                for _ in range(num_procs):
                    future = pool.schedule(
                        run_local_worker,
                        args=[
                            task_queue_name,
                            outcome_queue_name,
                            local,
                            sleep_sec,
                            idle_timeout,
                            pool_name,
                        ],
                    )
                    futures.append(future)

                idle_line = (
                    "Idle timeout not set."
                    if idle_timeout is None
                    else f"Idle timeout {idle_timeout:.1f}s (pool-wide coordination)."
                )
                logger.info(
                    f"Created {num_procs} local workers attached to queues "
                    f"`{task_queue_name}` / `{outcome_queue_name}`. " + idle_line
                )
                yield futures
            finally:
                pool.stop()
                pool.join()
                logger.info(
                    f"Cleaned up {num_procs} local workers that were attached to queues "
                    f"`{task_queue_name}` / `{outcome_queue_name}`."
                )
        finally:
            if activity_tracker:
                activity_tracker.unlink()


@builder.register("mazepa.run_worker_manager")
def run_worker_manager(
    task_queue: SQSQueue[Task],
    outcome_queue: SQSQueue[OutcomeReport],
    sleep_sec: float = 1.0,
    num_procs: int = 1,
    semaphores_spec: dict[SemaphoreType, int] | None = None,
    idle_timeout: int | None = None,
    suppress_worker_logs: bool = False,
    resource_monitor_interval: float | None = 1.0,
):
    with ExitStack() as stack:
        stack.enter_context(configure_semaphores(semaphores_spec))
        worker_pool_ctx = setup_local_worker_pool(
            num_procs,
            task_queue.name,
            outcome_queue.name,
            local=False,
            sleep_sec=sleep_sec,
            idle_timeout=idle_timeout,
            suppress_worker_logs=suppress_worker_logs,
            resource_monitor_interval=resource_monitor_interval,
        )
        pool_futures = stack.enter_context(worker_pool_ctx)

        while True:
            for i, future in enumerate(pool_futures):
                if future.done():
                    try:
                        exit_reason = future.result(timeout=0)
                        logger.info(
                            f"Worker {i} has returned (reason: {exit_reason}), "
                            f"cleaning the worker pool..."
                        )
                    except Exception as e:  # pylint: disable=broad-except
                        logger.info(
                            f"Worker {i} has returned with exception ({type(e).__name__}: {e}), "
                            f"cleaning the worker pool..."
                        )
                    break
            else:
                time.sleep(sleep_sec)
                continue
            break
        logger.info("Exiting worker manager.")
