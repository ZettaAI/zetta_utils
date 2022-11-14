import time

from zetta_utils.log import get_logger

from . import ExecutionQueue

logger = get_logger("mazepa")


def run_worker(
    exec_queue: ExecutionQueue, sleep_sec: int = 4, max_pull_num: int = 1
):  # pragma: no cover # TODO
    while True:
        tasks = exec_queue.pull_tasks(max_num=max_pull_num)
        logger.info(f"Got {len(tasks)} tasks.")

        if len(tasks) == 0:
            logger.info(f"Sleeping for {sleep_sec} secs.")
            time.sleep(sleep_sec)
        else:
            logger.info("STARTING: taks batch execution.")
            for e in tasks:
                e()
            logger.info("DONE: taks batch execution.")
