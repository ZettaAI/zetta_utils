from __future__ import annotations

import contextlib
import multiprocessing
import os
import time
from contextlib import ExitStack
from itertools import repeat

import pebble

from zetta_utils import builder, log, try_load_train_inference
from zetta_utils.common import RepeatTimer
from zetta_utils.common.resource_monitor import ResourceMonitor
from zetta_utils.mazepa import SemaphoreType, Task, configure_semaphores, run_worker
from zetta_utils.mazepa.task_outcome import OutcomeReport
from zetta_utils.message_queues import FileQueue, SQSQueue

logger = log.get_logger("mazepa")


def worker_init() -> None:
    # For Kubernetes compatibility, ensure unbuffered output
    os.environ["PYTHONUNBUFFERED"] = "1"

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
    monitor_resources: bool,
    resource_monitor_interval: float,
    resource_monitor_summary: bool,
):
    """
    Context manager for creating task/outcome queues, alongside a persistent pool of workers.
    """
    # Start resource monitoring
    resource_timer = None
    resource_monitor = None
    if monitor_resources:
        try:
            resource_monitor = ResourceMonitor(
                log_interval_seconds=resource_monitor_interval,
                collect_summary=resource_monitor_summary,
            )
            resource_timer = RepeatTimer(resource_monitor_interval, resource_monitor.log_usage)
            resource_timer.start()
            mode = "summary" if resource_monitor_summary else "continuous"
            logger.info(
                f"Started resource monitoring ({mode} mode) with {resource_monitor_interval}s interval"
            )
        except Exception as e:
            logger.warning(f"Failed to start resource monitoring: {e}")

    try:
        pool = pebble.ProcessPool(
            max_workers=num_procs,
            context=multiprocessing.get_context("fork"),
            initializer=worker_init,
        )
        future = pool.map(
            run_local_worker,
            repeat(task_queue_name, num_procs),
            repeat(outcome_queue_name, num_procs),
            repeat(local, num_procs),
            repeat(sleep_sec, num_procs),
            repeat(idle_timeout, num_procs),
        )
        idle_line = (
            "Idle timeout not set." if idle_timeout is None else "Idle timeout {idle_timeout}s."
        )
        logger.info(
            f"Created {num_procs} local workers attached to queues "
            f"`{task_queue_name}` / `{outcome_queue_name}`. " + idle_line
        )
        yield future
    finally:
        # Stop resource monitoring and log summary if enabled
        if resource_timer is not None:
            resource_timer.cancel()
            logger.info("Stopped resource monitoring")

        if resource_monitor is not None and resource_monitor_summary:
            resource_monitor.log_summary()

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
    monitor_resources: bool = True,
    resource_monitor_interval: float = 1.0,
    resource_monitor_summary: bool = True,
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
            monitor_resources=monitor_resources,
            resource_monitor_interval=resource_monitor_interval,
            resource_monitor_summary=resource_monitor_summary,
        )
        pool_future = stack.enter_context(worker_pool_ctx)

        while True:
            if pool_future and pool_future.done():
                logger.info("All worker processes have returned, cleaning the worker pool...")
                break
            time.sleep(1)
        logger.info("Exiting worker manager.")
