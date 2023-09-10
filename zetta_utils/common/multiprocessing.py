# pylint: disable=global-statement
from __future__ import annotations

import contextlib

from pebble import ProcessPool

from zetta_utils import log

logger = log.get_logger("zetta_utils")

PERSISTENT_PROCESS_POOL: ProcessPool | None = None


@contextlib.contextmanager
def setup_persistent_process_pool(num_procs: int):
    """
    Context manager for creating a persistent pool of workers.
    """

    global PERSISTENT_PROCESS_POOL
    try:
        if num_procs == 1:
            logger.info("Skipping creation because 1 process is requested.")
        elif PERSISTENT_PROCESS_POOL is not None:
            raise RuntimeError("Persistent process pool already exists.")
        else:
            logger.info(f"Creating a persistent process pool with {num_procs} processes.")
            PERSISTENT_PROCESS_POOL = ProcessPool(num_procs)
        yield
    finally:
        if num_procs == 1:
            pass
        elif PERSISTENT_PROCESS_POOL is None:
            raise RuntimeError("Persistent process pool does not exist.")
        else:
            PERSISTENT_PROCESS_POOL.stop()
            PERSISTENT_PROCESS_POOL.join()
            PERSISTENT_PROCESS_POOL = None
            logger.info("Cleaned up persistent process pool.")


def get_persistent_process_pool() -> ProcessPool | None:
    """
    Fetches and returns either the semaphore associated with the current process,
    or the semaphore associated with the parent process, in that order.
    """
    return PERSISTENT_PROCESS_POOL
