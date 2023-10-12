from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import time
from contextlib import ExitStack

import psutil

from zetta_utils import builder, log
from zetta_utils.mazepa import SemaphoreType, Task, configure_semaphores
from zetta_utils.mazepa.task_outcome import OutcomeReport
from zetta_utils.message_queues import SQSQueue

logger = log.get_logger("mazepa")


def detach_from_process_group() -> None:
    os.setpgrp()


def get_local_worker_command(
    task_queue_name: str, outcome_queue_name: str, local: bool = True, sleep_sec: float = 0.1
):
    queue_type = "LocalQueue" if local else "SQSQueue"

    task_queue_spec = {
        "@type": queue_type,
        "name": task_queue_name,
    }
    outcome_queue_spec = {"@type": queue_type, "name": outcome_queue_name, "pull_wait_sec": 1.0}
    result = (
        """
    zetta -vv -l try run -s '{
        "@type": "mazepa.run_worker"
    """
        + f"task_queue: {json.dumps(task_queue_spec)}\n"
        + f"outcome_queue: {json.dumps(outcome_queue_spec)}\n"
        + f"sleep_sec: {sleep_sec}\n"
        + """
        max_pull_num: 1
    }'
    """
    )
    return result


@contextlib.contextmanager
def setup_local_worker_pool(
    num_procs: int,
    task_queue_name: str,
    outcome_queue_name: str,
    local: bool = True,
    sleep_sec: float = 0.1,
):
    """
    Context manager for creating task/outcome queues, alongside a persistent pool of workers.
    """
    worker_procs = []
    with tempfile.TemporaryFile() as iofile:
        try:
            worker_procs = [
                psutil.Process(
                    subprocess.Popen(  # pylint: disable=subprocess-popen-preexec-fn
                        get_local_worker_command(
                            task_queue_name, outcome_queue_name, local=local, sleep_sec=sleep_sec
                        ),
                        shell=True,
                        stdin=iofile,
                        stdout=iofile,
                        stderr=iofile,
                        preexec_fn=detach_from_process_group,
                    ).pid
                )
                for _ in range(num_procs)
            ]
            logger.info(
                f"Created {num_procs} local workers attached to queues "
                f"`{task_queue_name}`/`{outcome_queue_name}`."
            )
            yield
        finally:
            for proc in worker_procs:
                for pid in proc.children(recursive=True):
                    pid.kill()
                proc.kill()
            logger.info(
                f"Cleaned up {num_procs} local workers that were attached to queues "
                f"`{task_queue_name}`/`{outcome_queue_name}`."
            )


@builder.register("mazepa.run_worker_manager")
def run_worker_manager(
    task_queue: SQSQueue[Task],
    outcome_queue: SQSQueue[OutcomeReport],
    sleep_sec: float = 1.0,
    num_procs: int = 1,
    semaphores_spec: dict[SemaphoreType, int] | None = None,
):
    with ExitStack() as stack:
        stack.enter_context(configure_semaphores(semaphores_spec))
        stack.enter_context(
            setup_local_worker_pool(
                num_procs, task_queue.name, outcome_queue.name, local=False, sleep_sec=sleep_sec
            )
        )
        while True:
            time.sleep(1)
