# pylint: disable=bare-except, c-extension-no-member

import os
from multiprocessing import shared_memory
from typing import List, get_args

import posix_ipc
import pytest

from zetta_utils.mazepa.semaphores import (
    DummySemaphore,
    SemaphoreType,
    TimingTracker,
    _log_timing_summary,
    configure_semaphores,
    get_semaphore_stats,
    name_to_posix_name,
    reset_all_timing_data,
    reset_timing_data,
    semaphore,
)


@pytest.fixture(autouse=True)
def cleanup_semaphores():
    yield
    sema_types: List[SemaphoreType] = ["read", "write", "cuda", "cpu"]
    for name in sema_types:
        try:
            # two unlinks in case grandparent semaphore exists
            semaphore(name).unlink()
            semaphore(name).unlink()
        except:
            pass


def test_default_semaphore_init():
    with configure_semaphores():
        assert not isinstance(semaphore("read"), DummySemaphore)


@pytest.mark.parametrize(
    "semaphore_spec",
    [
        {
            "read": 2,
            "write": 2,
            "cuda": 2,
            "cpu": 2,
        }
    ],
)
def test_width_2(semaphore_spec):
    with configure_semaphores(semaphore_spec):
        with semaphore("read"):
            with semaphore("read"):
                pass


def test_double_init_exc():
    with pytest.raises(RuntimeError):
        with configure_semaphores():
            with configure_semaphores():
                pass


def test_unlink_nonexistent_exc():
    with pytest.raises(RuntimeError):
        with configure_semaphores():
            semaphore("read").unlink()
        # exception on exiting context


def test_get_parent_semaphore():
    try:
        sema = posix_ipc.Semaphore(name_to_posix_name("read", os.getppid()))
        sema.unlink()
    except:
        pass
    sema = posix_ipc.Semaphore(name_to_posix_name("read", os.getppid()), flags=posix_ipc.O_CREX)
    assert sema.name == semaphore("read").semaphore.name
    sema.unlink()


@pytest.mark.parametrize(
    "name",
    [
        "read",
        "write",
        "cuda",
        "cpu",
    ],
)
def test_dummy_semaphores(name):
    with semaphore(name):
        pass


@pytest.mark.parametrize(
    "name",
    [
        "writing",
        "tests",
        "is",
        "hard",
    ],
)
def test_dummy_semaphores_exc(name):
    with pytest.raises(ValueError):
        with semaphore(name):
            pass


@pytest.mark.parametrize(
    "semaphore_spec",
    [
        {
            "read": 1,
            "write": 1,
            "cuda": 1,
            "cpu": 1,
            "nonsense": 1,
        }
    ],
)
def test_invalid_semaphore_type_exc(semaphore_spec):
    with pytest.raises(ValueError):
        with configure_semaphores(semaphore_spec):
            pass


@pytest.mark.parametrize(
    "semaphore_spec",
    [
        {
            "read": 1,
            "write": 1,
            "cuda": 1,
            "cpu": -1,
        }
    ],
)
def test_invalid_semaphore_width_exc(semaphore_spec):
    with pytest.raises(ValueError):
        with configure_semaphores(semaphore_spec):
            pass


@pytest.mark.parametrize(
    "semaphore_spec",
    [
        {
            "read": 1,
            "write": 1,
            "cuda": 1,
        }
    ],
)
def test_missing_semaphore_type_exc(semaphore_spec):
    with pytest.raises(ValueError):
        with configure_semaphores(semaphore_spec):
            pass


def test_timingtracker_reset():
    tracker = TimingTracker("test")
    tracker.create_shared_memory().close()
    tracker.reset_timing_data()
    tracker.unlink()


def test_timingtracker_duplicate_shm_exc():
    tracker = TimingTracker("test")
    tracker.create_shared_memory().close()
    with pytest.raises(FileExistsError):
        tracker.create_shared_memory()
    tracker.unlink()


def test_timingtracker_duplicate_unlink():
    tracker = TimingTracker("test")
    tracker.create_shared_memory().close()
    tracker.unlink()
    tracker.unlink()


def test_timingtracker_add_wait_time_noshm_exc():
    tracker = TimingTracker("read")
    with pytest.raises(RuntimeError):
        tracker.add_wait_time(1.0)


def test_timingtracker_add_lease_time_noshm_exc():
    tracker = TimingTracker("read")
    with pytest.raises(RuntimeError):
        tracker.add_lease_time(1.0)


def test_timingtracker_get_timing_data_noshm_exc():
    tracker = TimingTracker("read")
    with pytest.raises(RuntimeError):
        tracker.get_timing_data()


def test_timingtracker_reset_timing_data_noshm_exc():
    tracker = TimingTracker("read")
    with pytest.raises(RuntimeError):
        tracker.reset_timing_data()


def test_get_semaphore_stats_exc():
    with pytest.raises(ValueError):
        get_semaphore_stats("exc")  # type: ignore


def test_reset_all_timing_data():
    for name in get_args(SemaphoreType):
        tracker = TimingTracker(name)
        tracker.create_shared_memory().close()
    reset_all_timing_data()
    for name in get_args(SemaphoreType):
        tracker = TimingTracker(name)
        tracker.unlink()


def test_reset_timing_data():
    tracker = TimingTracker("read")
    tracker.create_shared_memory().close()
    reset_timing_data("read")
    tracker.unlink()


def test_reset_timing_data_wrongname_exc():
    with pytest.raises(ValueError):
        reset_timing_data("exc")  # type: ignore


def test_reset_timing_data_function_exc():
    with pytest.raises(RuntimeError):
        reset_timing_data("read")


def test_log_timing_summary_exc():
    with pytest.raises(RuntimeError):
        _log_timing_summary({"read": 1})


def test_tracker_cleanup_exc(mocker):

    original_unlink = shared_memory.SharedMemory.unlink

    def failing_unlink(self):
        raise PermissionError("Cannot unlink shared memory")

    mocker.patch.object(shared_memory.SharedMemory, "unlink", failing_unlink)

    tracker = TimingTracker("read")
    tracker.create_shared_memory().close()

    with pytest.raises(RuntimeError):
        tracker.unlink()

    mocker.patch.object(shared_memory.SharedMemory, "unlink", original_unlink)

    tracker.unlink()


def test_configure_tracker_cleanup_exc(mocker):

    original_unlink = shared_memory.SharedMemory.unlink

    def failing_unlink(self):
        raise PermissionError("Cannot unlink shared memory")

    mocker.patch.object(shared_memory.SharedMemory, "unlink", failing_unlink)

    with pytest.raises(RuntimeError):
        with configure_semaphores():
            pass

    mocker.patch.object(shared_memory.SharedMemory, "unlink", original_unlink)

    for name in get_args(SemaphoreType):
        semaphore(name).unlink()
        TimingTracker(name).unlink()


def test_configure_semaphores_existing_shm():
    tracker = TimingTracker("read")
    tracker.create_shared_memory().close()
    tracker.add_wait_time(10.0)
    with configure_semaphores():
        stats = get_semaphore_stats("read")
        assert stats["total_wait_time"] == 0.0


def test_semaphores_logging():
    with configure_semaphores(
        {
            "read": 1,
            "write": 0,
            "cuda": 0,
            "cpu": 0,
        }
    ):
        with semaphore("read"):
            pass
