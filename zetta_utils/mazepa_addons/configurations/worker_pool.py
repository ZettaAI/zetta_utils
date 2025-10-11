from __future__ import annotations

import contextlib
import multiprocessing
import os
import sys
import time
from contextlib import ExitStack
from itertools import repeat

import pebble

from zetta_utils import builder, log, try_load_train_inference
from zetta_utils.common import monitor_resources
from zetta_utils.mazepa import SemaphoreType, Task, configure_semaphores, run_worker
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


def worker_init(suppress_worker_logs: bool) -> None:
    # For Kubernetes compatibility, ensure unbuffered output
    os.environ["PYTHONUNBUFFERED"] = "1"
    if suppress_worker_logs:
        redirect_buffers()
    try_load_train_inference()


def run_local_worker(
    task_queue_name: str,
    outcome_queue_name: str,
    local: bool = True,
    sleep_sec: float = 0.1,
    idle_timeout: int | None = None,
) -> None:
    logger.info("Creating a local worker in this process....")
    queue_type = FileQueue if local else SQSQueue
    task_queue = queue_type(name=task_queue_name)
    outcome_queue = queue_type(name=outcome_queue_name, pull_wait_sec=1.0)
    run_worker(
        task_queue=task_queue,
        outcome_queue=outcome_queue,
        sleep_sec=sleep_sec,
        max_pull_num=1,
        idle_timeout=idle_timeout,
    )
    logger.info("Local worker returned.")


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
    """
    with monitor_resources(resource_monitor_interval):
        pool = pebble.ProcessPool(
            max_workers=num_procs,
            context=multiprocessing.get_context("fork"),
            initializer=worker_init,
            initargs=[
                suppress_worker_logs,
            ],
        )
        try:
            future = pool.map(
                run_local_worker,
                repeat(task_queue_name, num_procs),
                repeat(outcome_queue_name, num_procs),
                repeat(local, num_procs),
                repeat(sleep_sec, num_procs),
                repeat(idle_timeout, num_procs),
            )
            idle_line = (
                "Idle timeout not set."
                if idle_timeout is None
                else "Idle timeout {idle_timeout:.1f}s."
            )
            logger.info(
                f"Created {num_procs} local workers attached to queues "
                f"`{task_queue_name}` / `{outcome_queue_name}`. " + idle_line
            )
            yield future
        finally:
            pool.stop()
            pool.join()
            logger.info(
                f"Cleaned up {num_procs} local workers that were attached to queues "
                f"`{task_queue_name}` / `{outcome_queue_name}`."
            )


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
        pool_future = stack.enter_context(worker_pool_ctx)

        while True:
            if pool_future and pool_future.done():
                logger.info("All worker processes have returned, cleaning the worker pool...")
                break
            time.sleep(sleep_sec)
        logger.info("Exiting worker manager.")
