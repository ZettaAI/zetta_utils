from __future__ import annotations

import time
from typing import Callable, Optional

from zetta_utils import log

from . import ExecutionQueue, Task

logger = log.get_logger("mazepa")


class AcceptAllTasks:
    def __call__(self, task: Task):
        return True


def run_worker(
    exec_queue: ExecutionQueue,
    sleep_sec: float = 4.0,
    max_pull_num: int = 1,
    max_runtime: Optional[float] = None,
    task_filter_fn: Callable[[Task], bool] = AcceptAllTasks(),
    debug: bool = False,
):
    start_time = time.time()
    while True:
        tasks = exec_queue.pull_tasks(max_num=max_pull_num)
        logger.info(f"Got {len(tasks)} tasks.")
        if len(tasks) == 0:
            logger.info(f"Sleeping for {sleep_sec} secs.")
            time.sleep(sleep_sec)
        else:
            logger.info("STARTING: taks batch execution.")
            for task in tasks:
                if task_filter_fn(task):
                    with log.logging_tag_ctx("task_id", task.id_):
                        with log.logging_tag_ctx("execution_id", task.execution_id):
                            task(debug=debug)
                else:
                    task.cancel_without_starting()

            logger.info("DONE: taks batch execution.")

        if max_runtime is not None and time.time() - start_time > max_runtime:
            break
