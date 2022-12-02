import time

from zetta_utils import builder, log

from . import ExecutionQueue

logger = log.get_logger("mazepa")


@builder.register("mazepa.run_worker")
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
                log.set_logging_label("task_id", e.id_)
                e()
                log.set_logging_label("task_id", None)
            logger.info("DONE: taks batch execution.")
