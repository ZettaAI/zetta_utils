from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import signal
import threading
import time
from contextlib import ExitStack

import pebble
import psutil

from zetta_utils import builder, get_mp_context, log
from zetta_utils.common import monitor_resources
from zetta_utils.mazepa import SemaphoreType, Task, configure_semaphores, run_worker
from zetta_utils.mazepa.pool_activity import PoolActivityTracker
from zetta_utils.mazepa.task_outcome import OutcomeReport
from zetta_utils.mazepa.worker import worker_init
from zetta_utils.message_queues import FileQueue, SQSQueue

logger = log.get_logger("mazepa")


def run_local_worker(
    task_queue_name: str,
    outcome_queue_name: str,
    local: bool = True,
    sleep_sec: float = 0.1,
    activity_tracker: PoolActivityTracker | None = None,
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
        activity_tracker=activity_tracker,
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
        pool_name = f"{task_queue_name}_{outcome_queue_name}"
        activity_tracker: PoolActivityTracker | None = None
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
                    current_log_level,  # log_level
                    suppress_worker_logs,  # suppress_logs
                    True,  # set_start_method
                    multiprocessing.get_start_method(),  # multiprocessing_start_method
                    True,  # load_train_inference
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
                            activity_tracker,
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
                # SIGKILL direct children so workers exit immediately on
                # unwind. For the Ctrl-C confirm path this is the only
                # signal workers receive — instant exit, no waiting for
                # the current task. For the k8s SIGTERM path,
                # run_worker_manager's polling loop already forwarded
                # SIGTERM before breaking, giving workers a brief grace
                # window; SIGKILL here is the backstop that prevents
                # pool.join() from wedging on long-running tasks.
                for child in psutil.Process(os.getpid()).children(recursive=False):
                    try:
                        child.send_signal(signal.SIGKILL)
                    except psutil.NoSuchProcess:
                        pass
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
    # k8s sends SIGTERM to PID 1 only (this process). Flip a flag the
    # polling loop consults so we break and trigger ExitStack unwind,
    # which runs setup_local_worker_pool's finally — SIGKILL to workers.
    shutdown_requested = threading.Event()

    def _handle_sigterm(*_):  # pragma: no cover
        shutdown_requested.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGHUP, _handle_sigterm)

    # Pool-wide idle timeout is checked here (not in each worker) so the
    # decision is made once per pool. The tracker wraps the shared memory
    # segment created by setup_local_worker_pool below; constructing a
    # local reference is cheap (attrs.frozen around a single str name).
    activity_tracker: PoolActivityTracker | None = None
    if idle_timeout is not None:
        pool_name = f"{task_queue.name}_{outcome_queue.name}"
        activity_tracker = PoolActivityTracker(pool_name)

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
            if shutdown_requested.is_set():
                logger.info("SIGTERM received; exiting worker manager.")
                break
            if (
                idle_timeout is not None
                and activity_tracker is not None
                and activity_tracker.check_idle_timeout(idle_timeout)
            ):
                logger.info("Pool idle timeout exceeded; exiting worker manager.")
                break
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
