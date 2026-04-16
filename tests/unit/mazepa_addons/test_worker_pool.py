import os
import signal
import threading
import time
from unittest.mock import MagicMock

import pytest

from zetta_utils.mazepa.pool_activity import PoolActivityTracker
from zetta_utils.mazepa_addons.configurations.worker_pool import run_worker_manager

# pylint: disable=redefined-outer-name


@pytest.fixture(autouse=True)
def restore_signal_handlers():
    """Save/restore signal handlers so run_worker_manager's signal.signal
    calls don't leak into subsequent tests."""
    saved = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGHUP: signal.getsignal(signal.SIGHUP),
    }
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _mock_worker_pool_ctx(mocker, futures):
    """Patch setup_local_worker_pool and configure_semaphores to no-op
    context managers that yield the given futures list."""
    semaphores_mock = MagicMock()
    semaphores_mock.return_value.__enter__.return_value = None
    semaphores_mock.return_value.__exit__.return_value = False
    mocker.patch(
        "zetta_utils.mazepa_addons.configurations.worker_pool.configure_semaphores",
        semaphores_mock,
    )

    pool_mock = MagicMock()
    pool_mock.return_value.__enter__.return_value = futures
    pool_mock.return_value.__exit__.return_value = False
    mocker.patch(
        "zetta_utils.mazepa_addons.configurations.worker_pool.setup_local_worker_pool",
        pool_mock,
    )


def test_run_worker_manager_exits_on_idle_timeout(mocker):
    """Manager attaches to the pool activity tracker by name, polls it,
    and exits when check_idle_timeout returns True. Verifies the pool_name
    derivation and the polling-loop integration both work end-to-end."""
    _mock_worker_pool_ctx(mocker, futures=[])

    task_queue = MagicMock()
    task_queue.name = "idle_test_task"
    outcome_queue = MagicMock()
    outcome_queue.name = "idle_test_outcome"

    # Create the shm the manager will attach to. The pool_name derivation
    # must match run_worker_manager's: f"{task_queue.name}_{outcome_queue.name}".
    pool_name = "idle_test_task_idle_test_outcome"
    tracker = PoolActivityTracker(pool_name)
    tracker.create_shared_memory().close()

    try:
        start = time.time()
        run_worker_manager(
            task_queue=task_queue,
            outcome_queue=outcome_queue,
            sleep_sec=0.05,
            num_procs=1,
            idle_timeout=1,
        )
        elapsed = time.time() - start
        # idle_timeout=1 plus a polling cycle or two of slack; far less
        # than the default wedge time the old code was vulnerable to.
        assert elapsed < 3.0
    finally:
        tracker.unlink()


def test_run_worker_manager_exits_on_sigterm(mocker):
    """Manager's SIGTERM handler flips a flag that the polling loop
    consults. Verifies the flag path without requiring a real pool."""
    _mock_worker_pool_ctx(mocker, futures=[])

    task_queue = MagicMock()
    task_queue.name = "sigterm_test_task"
    outcome_queue = MagicMock()
    outcome_queue.name = "sigterm_test_outcome"

    def send_sigterm_soon():
        time.sleep(0.3)
        os.kill(os.getpid(), signal.SIGTERM)

    sender = threading.Thread(target=send_sigterm_soon)
    sender.start()
    try:
        start = time.time()
        run_worker_manager(
            task_queue=task_queue,
            outcome_queue=outcome_queue,
            sleep_sec=0.05,
            num_procs=1,
            idle_timeout=None,  # no idle timeout, only SIGTERM should cause exit
        )
        elapsed = time.time() - start
        # Should exit shortly after SIGTERM delivery (0.3s + a few poll cycles).
        assert elapsed < 2.0
    finally:
        sender.join(timeout=1.0)
