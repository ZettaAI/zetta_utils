# pylint: disable=bare-except, c-extension-no-member, protected-access

import builtins
import os
from multiprocessing import shared_memory
from typing import List, get_args

import posix_ipc
import pytest

from zetta_utils.mazepa.semaphores import (
    PRIORITY_MAX_WAITERS,
    PRIORITY_SHM_SIZE,
    DummySemaphore,
    PriorityPidSemaphore,
    SemaphoreType,
    TimingTracker,
    _log_timing_summary,
    _priority_mutex_name,
    _priority_shm_name,
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
    sema_types: List[SemaphoreType] = list(get_args(SemaphoreType))
    for name in sema_types:
        try:
            # two unlinks in case grandparent semaphore exists
            semaphore(name).unlink()
            semaphore(name).unlink()
        except:
            pass
        try:
            TimingTracker(name, pid=os.getpid()).unlink()
        except:
            pass
        try:
            posix_ipc.Semaphore(_priority_mutex_name(name, os.getpid())).unlink()
        except:
            pass
        try:
            shm = shared_memory.SharedMemory(name=_priority_shm_name(name, os.getpid()))
            shm.close()
            shm.unlink()
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
    ppid = os.getppid()
    for cleanup in (
        lambda: posix_ipc.Semaphore(name_to_posix_name("read", ppid)).unlink(),
        lambda: posix_ipc.Semaphore(_priority_mutex_name("read", ppid)).unlink(),
        lambda: shared_memory.SharedMemory(name=_priority_shm_name("read", ppid)).unlink(),
    ):
        try:
            cleanup()
        except:
            pass
    sema = posix_ipc.Semaphore(name_to_posix_name("read", ppid), flags=posix_ipc.O_CREX)
    mutex = posix_ipc.Semaphore(
        _priority_mutex_name("read", ppid), flags=posix_ipc.O_CREX, initial_value=1
    )
    shm = shared_memory.SharedMemory(
        name=_priority_shm_name("read", ppid), create=True, size=PRIORITY_SHM_SIZE
    )
    try:
        inner = semaphore("read").semaphore
        assert isinstance(inner, PriorityPidSemaphore)
        assert sema.name == inner._slot_sem.name
    finally:
        shm.close()
        shm.unlink()
        mutex.unlink()
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
    tracker = TimingTracker("test", pid=os.getpid())
    tracker.create_shared_memory().close()
    tracker.reset_timing_data()
    tracker.unlink()


def test_timingtracker_duplicate_shm_exc():
    tracker = TimingTracker("test", pid=os.getpid())
    tracker.create_shared_memory().close()
    with pytest.raises(FileExistsError):
        tracker.create_shared_memory()
    tracker.unlink()


def test_timingtracker_duplicate_unlink():
    tracker = TimingTracker("test", pid=os.getpid())
    tracker.create_shared_memory().close()
    tracker.unlink()
    tracker.unlink()


def test_timingtracker_add_wait_time_noshm_exc():
    tracker = TimingTracker("read", pid=os.getpid())
    with pytest.raises(RuntimeError):
        tracker.add_wait_time(1.0)


def test_timingtracker_add_lease_time_noshm_exc():
    tracker = TimingTracker("read", pid=os.getpid())
    with pytest.raises(RuntimeError):
        tracker.add_lease_time(1.0)


def test_timingtracker_get_timing_data_noshm_exc():
    tracker = TimingTracker("read", pid=os.getpid())
    with pytest.raises(RuntimeError):
        tracker.get_timing_data()


def test_timingtracker_reset_timing_data_noshm_exc():
    tracker = TimingTracker("read", pid=os.getpid())
    with pytest.raises(RuntimeError):
        tracker.reset_timing_data()


def test_get_semaphore_stats_exc():
    with pytest.raises(ValueError):
        get_semaphore_stats("exc")  # type: ignore


def test_reset_all_timing_data():
    for name in get_args(SemaphoreType):
        tracker = TimingTracker(name, pid=os.getpid())
        tracker.create_shared_memory().close()
    reset_all_timing_data()
    for name in get_args(SemaphoreType):
        tracker = TimingTracker(name, pid=os.getpid())
        tracker.unlink()


def test_reset_timing_data():
    tracker = TimingTracker("read", pid=os.getpid())
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

    tracker = TimingTracker("read", pid=os.getpid())
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
        TimingTracker(name, pid=os.getpid()).unlink()


def test_configure_semaphores_existing_shm():
    tracker = TimingTracker("read", pid=os.getpid())
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


def test_priority_waiter_array_full_exc():
    with configure_semaphores():
        priority_sem = semaphore("read").semaphore
        assert isinstance(priority_sem, PriorityPidSemaphore)
        arr = priority_sem._arr()
        for i in range(PRIORITY_MAX_WAITERS):
            arr[i] = 99000 + i
        try:
            with pytest.raises(RuntimeError, match="waiter array is full"):
                priority_sem._add_waiter(os.getpid())
        finally:
            for i in range(PRIORITY_MAX_WAITERS):
                arr[i] = 0


class _AcquireFailsOnce:
    """Wrapper around a posix_ipc.Semaphore that raises BusyError on the first
    acquire(timeout=0) call, then forwards to the real sem."""

    def __init__(self, real_sem):
        self._real = real_sem
        self.calls = 0

    def acquire(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise posix_ipc.BusyError()
        return self._real.acquire(*args, **kwargs)

    def release(self):
        return self._real.release()

    @property
    def name(self):
        return self._real.name

    def unlink(self):
        return self._real.unlink()


def test_priority_sem_busy_then_acquire():
    with configure_semaphores():
        sema = semaphore("read")
        priority_sem = sema.semaphore
        assert isinstance(priority_sem, PriorityPidSemaphore)
        wrapper = _AcquireFailsOnce(priority_sem._slot_sem)
        real_sem = priority_sem._slot_sem
        priority_sem._slot_sem = wrapper
        try:
            with sema:
                pass
            assert wrapper.calls >= 2
        finally:
            priority_sem._slot_sem = real_sem


def test_configure_tracker_unlink_exc(mocker):
    original_unlink = shared_memory.SharedMemory.unlink

    def selective_failing_unlink(self):
        if "zetta_semaphore_timing_" in self.name:
            raise PermissionError("Cannot unlink tracker shm")
        return original_unlink(self)

    mocker.patch.object(shared_memory.SharedMemory, "unlink", selective_failing_unlink)

    with pytest.raises(
        RuntimeError, match="Failed to cleanup shared memory for tracking"
    ):
        with configure_semaphores():
            pass

    mocker.patch.object(shared_memory.SharedMemory, "unlink", original_unlink)

    for name in get_args(SemaphoreType):
        try:
            semaphore(name).unlink()
        except:
            pass
        try:
            TimingTracker(name, pid=os.getpid()).unlink()
        except:
            pass


def test_cuda_sem_no_torch(mocker):
    real_import = builtins.__import__

    def fail_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch in this test")
        return real_import(name, *args, **kwargs)

    with configure_semaphores():
        mocker.patch.object(builtins, "__import__", fail_torch)
        with semaphore("cuda"):
            pass
