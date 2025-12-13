import time
from multiprocessing import shared_memory

import pytest

from zetta_utils.mazepa.pool_activity import PoolActivityTracker


@pytest.fixture(autouse=True)
def cleanup_pool_activity():
    yield
    try:
        tracker = PoolActivityTracker("test")
        tracker.unlink()
    except:  # pylint: disable=bare-except
        pass


def test_poolactivitytracker_create_and_unlink():
    tracker = PoolActivityTracker("test")
    shm = tracker.create_shared_memory()
    shm.close()
    tracker.unlink()


def test_poolactivitytracker_duplicate_shm_exc():
    tracker = PoolActivityTracker("test")
    tracker.create_shared_memory().close()
    with pytest.raises(FileExistsError):
        tracker.create_shared_memory()
    tracker.unlink()


def test_poolactivitytracker_duplicate_unlink():
    tracker = PoolActivityTracker("test")
    tracker.create_shared_memory().close()
    tracker.unlink()
    tracker.unlink()


def test_poolactivitytracker_update_activity_time_noshm():
    tracker = PoolActivityTracker("nonexistent")
    tracker.update_activity_time()


def test_poolactivitytracker_increment_active_workers_noshm():
    tracker = PoolActivityTracker("nonexistent")
    tracker.increment_active_workers()


def test_poolactivitytracker_decrement_active_workers_noshm():
    tracker = PoolActivityTracker("nonexistent")
    tracker.decrement_active_workers()


def test_poolactivitytracker_get_activity_data_noshm():
    tracker = PoolActivityTracker("nonexistent")
    last_activity, active_count = tracker.get_activity_data()
    assert active_count == 0
    assert isinstance(last_activity, float)


def test_poolactivitytracker_initial_state():
    tracker = PoolActivityTracker("test")
    tracker.create_shared_memory().close()

    last_activity, active_count = tracker.get_activity_data()
    assert active_count == 0
    assert isinstance(last_activity, float)

    tracker.unlink()


def test_poolactivitytracker_update_activity_time():
    tracker = PoolActivityTracker("test")
    tracker.create_shared_memory().close()

    last_activity1, _ = tracker.get_activity_data()
    time.sleep(0.1)
    tracker.update_activity_time()
    last_activity2, _ = tracker.get_activity_data()

    assert last_activity2 > last_activity1

    tracker.unlink()


def test_poolactivitytracker_increment_decrement_workers():
    tracker = PoolActivityTracker("test")
    tracker.create_shared_memory().close()

    _, active_count = tracker.get_activity_data()
    assert active_count == 0

    tracker.increment_active_workers()
    _, active_count = tracker.get_activity_data()
    assert active_count == 1

    tracker.increment_active_workers()
    _, active_count = tracker.get_activity_data()
    assert active_count == 2

    tracker.decrement_active_workers()
    _, active_count = tracker.get_activity_data()
    assert active_count == 1

    tracker.decrement_active_workers()
    _, active_count = tracker.get_activity_data()
    assert active_count == 0

    tracker.unlink()


def test_poolactivitytracker_check_idle_timeout_active_workers():
    tracker = PoolActivityTracker("test")
    tracker.create_shared_memory().close()

    tracker.increment_active_workers()
    time.sleep(0.2)

    is_idle = tracker.check_idle_timeout(0.1)
    assert not is_idle

    tracker.unlink()


def test_poolactivitytracker_check_idle_timeout_no_timeout():
    tracker = PoolActivityTracker("test")
    tracker.create_shared_memory().close()

    is_idle = tracker.check_idle_timeout(1.0)
    assert not is_idle

    tracker.unlink()


def test_poolactivitytracker_check_idle_timeout_exceeded():
    tracker = PoolActivityTracker("test")
    tracker.create_shared_memory().close()

    time.sleep(0.2)

    is_idle = tracker.check_idle_timeout(0.1)
    assert is_idle

    tracker.unlink()


def test_poolactivitytracker_check_idle_timeout_activity_resets():
    tracker = PoolActivityTracker("test")
    tracker.create_shared_memory().close()

    time.sleep(0.2)

    tracker.update_activity_time()

    is_idle = tracker.check_idle_timeout(0.1)
    assert not is_idle

    tracker.unlink()


def test_poolactivitytracker_cleanup_exc(mocker):
    original_unlink = shared_memory.SharedMemory.unlink

    def failing_unlink(self):
        raise PermissionError("Cannot unlink shared memory")

    mocker.patch.object(shared_memory.SharedMemory, "unlink", failing_unlink)

    tracker = PoolActivityTracker("test")
    tracker.create_shared_memory().close()

    tracker.unlink()

    mocker.patch.object(shared_memory.SharedMemory, "unlink", original_unlink)

    tracker.unlink()
