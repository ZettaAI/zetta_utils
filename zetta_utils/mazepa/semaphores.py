from __future__ import annotations

import contextlib
import os
import struct
import time
from multiprocessing import resource_tracker, shared_memory
from typing import List, Literal, get_args

import attrs
import psutil
from posix_ipc import (  # pylint: disable=no-name-in-module
    O_CREX,
    BusyError,
    ExistentialError,
    Semaphore,
)

from zetta_utils import log
from zetta_utils.common.pprint import lrpad

logger = log.get_logger("mazepa")
SemaphoreType = Literal["read", "write", "cuda", "cpu", "tensorrt"]

DEFAULT_SEMA_COUNT = 1
TIMING_FORMAT = "dddd"  # wait_time, lease_time, lease_count, start_time
TIMING_SIZE = struct.calcsize(TIMING_FORMAT)

# Priority-by-lowest-PID coordination: how many simultaneous waiters can
# register against a single semaphore, and how often each waiter rechecks
# the queue while spinning. 64 is comfortably above any realistic procs/pod;
# the spin interval is tuned for sub-second contention windows where the
# CPU cost is negligible.
PRIORITY_MAX_WAITERS = 64
PRIORITY_SHM_SIZE = PRIORITY_MAX_WAITERS * 4  # int32 per slot
PRIORITY_SPIN_INTERVAL = 0.0005  # 0.5 ms


@attrs.frozen
class TimingTracker:
    """
    Manages global semaphore timing data using shared memory.
    """

    name: str
    pid: int

    def _get_shared_memory_name(self) -> str:
        return f"zetta_semaphore_timing_{self.pid}_{self.name}"

    def _get_shared_memory(self) -> shared_memory.SharedMemory:
        """Get existing shared memory (assumes it was already created by head node)."""
        name = self._get_shared_memory_name()
        shm = shared_memory.SharedMemory(name=name)
        # Workaround for https://bugs.python.org/issue38119: attaching processes get
        # registered with resource_tracker and will unlink the segment on exit, even
        # though they did not create it. Unregister so only the head node owns cleanup.
        # Guard: only unregister in non-creator processes. The creator (pid == self.pid)
        # must keep the registration so resource_tracker can clean up on abnormal exit.
        if os.getpid() != self.pid:  # pragma: no cover
            resource_tracker.unregister(f"/{name}", "shared_memory")
        return shm

    def create_shared_memory(self) -> shared_memory.SharedMemory:
        """Create new shared memory block (should only be called by head node)."""
        name = self._get_shared_memory_name()
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=TIMING_SIZE)
            # Initialize with zeros: wait_time, lease_time, lease_count, start_time
            shm.buf[:TIMING_SIZE] = struct.pack(TIMING_FORMAT, 0.0, 0.0, 0.0, time.time())
            return shm
        except FileExistsError as exc:
            raise FileExistsError(
                f"Shared memory `{self.name}` with name `{name}` already exists."
            ) from exc

    def add_wait_time(self, wait_time: float) -> None:
        """Add wait time to global tracking."""
        shm = None
        try:
            shm = self._get_shared_memory()
            # Read current data: wait_time, lease_time, lease_count, start_time
            current_data = struct.unpack(TIMING_FORMAT, shm.buf[:TIMING_SIZE])
            total_wait_time = current_data[0] + wait_time
            lease_time = current_data[1]  # unchanged
            lease_count = current_data[2]  # unchanged
            start_time = current_data[3]  # unchanged
            # Write updated data
            shm.buf[:TIMING_SIZE] = struct.pack(
                TIMING_FORMAT, total_wait_time, lease_time, lease_count, start_time
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Timing tracker for `{self.name}` " "has not been initialized."
            ) from e
        finally:
            if shm:
                shm.close()

    def add_lease_time(self, lease_time: float) -> None:
        """Add lease time to global tracking."""
        shm = None
        try:
            shm = self._get_shared_memory()
            current_data = struct.unpack(TIMING_FORMAT, shm.buf[:TIMING_SIZE])
            wait_time = current_data[0]
            total_lease_time = current_data[1] + lease_time
            lease_count = current_data[2] + 1.0
            start_time = current_data[3]
            shm.buf[:TIMING_SIZE] = struct.pack(
                TIMING_FORMAT, wait_time, total_lease_time, lease_count, start_time
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Timing tracker for `{self.name}` " "has not been initialized."
            ) from e
        finally:
            if shm:
                shm.close()

    def get_timing_data(self) -> tuple[float, float, int, float]:
        """Get timing data as (total_wait_time, total_lease_time, lease_count, start_time)."""
        shm = None
        try:
            shm = self._get_shared_memory()
            data = struct.unpack(TIMING_FORMAT, shm.buf[:TIMING_SIZE])
            return data[0], data[1], int(data[2]), data[3]
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Timing tracker for `{self.name}` " "has not been initialized."
            ) from e
        finally:
            if shm:
                shm.close()

    def reset_timing_data(self) -> None:
        """Reset timing data to zero."""
        shm = None
        try:
            shm = self._get_shared_memory()
            shm.buf[:TIMING_SIZE] = struct.pack(TIMING_FORMAT, 0.0, 0.0, 0.0, time.time())
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Timing tracker for `{self.name}` " "has not been initialized."
            ) from e
        finally:
            if shm:
                shm.close()

    def unlink(self) -> None:
        """Cleanup shared memory for this timing tracker."""
        shm_name = self._get_shared_memory_name()
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:  # pylint: disable=broad-except
            raise RuntimeError(f"Failed to cleanup shared memory for `{self.name}`") from e


def name_to_posix_name(name: SemaphoreType, pid: int) -> str:  # pragma: no cover
    return f"zetta_utils_{pid}_{name}_semaphore"


def _priority_mutex_name(name: SemaphoreType, pid: int) -> str:  # pragma: no cover
    return f"zetta_utils_{pid}_{name}_priority_mutex"


def _priority_shm_name(name: SemaphoreType, pid: int) -> str:  # pragma: no cover
    return f"zetta_semaphore_priority_{pid}_{name}"


@attrs.mutable
class PriorityPidSemaphore(contextlib.AbstractContextManager):
    """Cross-process semaphore that grants slots to the lowest-PID waiter first.

    Wraps a counting POSIX semaphore (`slot_sem`) with a shared-memory array of
    waiting PIDs guarded by a POSIX mutex semaphore. On acquire, each caller
    registers its PID and spins, locking the mutex to (a) reap dead waiters and
    (b) check whether its PID is the lowest live waiter and a slot is
    non-blockingly acquirable. The spin granularity (PRIORITY_SPIN_INTERVAL) is
    short enough that CPU cost is negligible at realistic concurrency, but long
    enough that mutex traffic stays bounded.

    The underlying primitives (slot sem, mutex sem, shm) are created/cleaned up
    by `configure_semaphores`; this class only opens existing handles.
    """

    name: SemaphoreType
    pid: int  # head pid that owns the underlying primitives
    _slot_sem: Semaphore = attrs.field(init=False, default=None)
    _mutex: Semaphore = attrs.field(init=False, default=None)
    _shm: shared_memory.SharedMemory | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        self._slot_sem = Semaphore(name_to_posix_name(self.name, self.pid))
        self._mutex = Semaphore(_priority_mutex_name(self.name, self.pid))
        self._shm = shared_memory.SharedMemory(name=_priority_shm_name(self.name, self.pid))
        if os.getpid() != self.pid:  # pragma: no cover
            resource_tracker.unregister(f"/{self._shm.name}", "shared_memory")

    def _arr(self) -> memoryview:
        assert self._shm is not None
        return self._shm.buf.cast("i")

    def _add_waiter(self, pid: int) -> None:
        # Caller holds self._mutex.
        arr = self._arr()
        for i in range(PRIORITY_MAX_WAITERS):
            if arr[i] == 0:
                arr[i] = pid
                return
        raise RuntimeError(
            f"Priority semaphore '{self.name}' waiter array is full "
            f"({PRIORITY_MAX_WAITERS} slots). Increase PRIORITY_MAX_WAITERS."
        )

    def _remove_waiter(self, pid: int) -> None:
        # Caller holds self._mutex.
        arr = self._arr()
        for i in range(PRIORITY_MAX_WAITERS):
            if arr[i] == pid:
                arr[i] = 0

    def _min_waiter(self) -> int | None:
        # Caller holds self._mutex.
        arr = self._arr()
        best: int | None = None
        for i in range(PRIORITY_MAX_WAITERS):
            p = int(arr[i])
            if p == 0:
                continue
            if best is None or p < best:
                best = p
        return best

    def __enter__(self):
        my_pid = os.getpid()
        self._mutex.acquire()
        try:
            self._add_waiter(my_pid)
        finally:
            self._mutex.release()

        while True:
            self._mutex.acquire()
            try:
                if self._min_waiter() == my_pid:
                    try:
                        self._slot_sem.acquire(timeout=0)
                    except BusyError:
                        pass
                    else:
                        self._remove_waiter(my_pid)
                        return self
            finally:
                self._mutex.release()
            time.sleep(PRIORITY_SPIN_INTERVAL)

    def __exit__(self, *args):
        try:
            self._slot_sem.release()
        finally:
            self._close_shm()

    def _close_shm(self) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:  # pylint: disable=broad-except  # pragma: no cover
                pass
            self._shm = None

    def __del__(self):  # pragma: no cover
        # Backup cleanup if __exit__ wasn't reached.
        self._close_shm()

    def unlink(self):
        try:
            self._slot_sem.unlink()
        except ExistentialError:  # pragma: no cover
            pass
        try:
            self._mutex.unlink()
        except ExistentialError:  # pragma: no cover
            pass
        try:
            shm = shared_memory.SharedMemory(name=_priority_shm_name(self.name, self.pid))
            shm.close()
            shm.unlink()
        except FileNotFoundError:  # pragma: no cover
            pass


@contextlib.contextmanager
def configure_semaphores(
    semaphores_spec: dict[SemaphoreType, int] | None = None
):  # pylint: disable=too-many-branches, too-many-statements
    """
    Context manager for creating and destroying semaphores.
    """

    sema_types_to_check: List[SemaphoreType] = ["read", "write", "cuda", "cpu", "tensorrt"]
    if semaphores_spec is not None:
        for name in semaphores_spec:
            if name not in get_args(SemaphoreType):
                raise ValueError(f"`{name}` is not a valid semaphore type.")
        try:
            _required: tuple[SemaphoreType, ...] = ("read", "write", "cuda", "cpu")
            for sema_type in _required:
                assert semaphores_spec[sema_type] >= 0
        except KeyError as e:
            raise ValueError(
                "`semaphores_spec` given to `execute_with_pool` must contain "
                "`read`, `write`, `cuda`, and `cpu`."
            ) from e
        except AssertionError as e:
            raise ValueError("Number of semaphores must be nonnegative.") from e
        semaphores_spec_ = {"tensorrt": DEFAULT_SEMA_COUNT, **semaphores_spec}
    else:
        semaphores_spec_ = {name: DEFAULT_SEMA_COUNT for name in sema_types_to_check}

    try:
        try:
            for name in semaphores_spec_:
                Semaphore(name_to_posix_name(name, os.getpid()), flags=0)
                raise RuntimeError(
                    f"Semaphore `{name}` with POSIX name "
                    f"`{name_to_posix_name(name, os.getpid())}` "
                    "already exists from the current process."
                )
        except ExistentialError:
            logger.info(f"Creating semaphores from within process {os.getpid()}.")
            summary = ""
            for name, width in semaphores_spec_.items():
                Semaphore(
                    name_to_posix_name(name, os.getpid()),
                    flags=O_CREX,
                    initial_value=width,
                )
                # Mutex semaphore guards the priority waiter array.
                Semaphore(
                    _priority_mutex_name(name, os.getpid()),
                    flags=O_CREX,
                    initial_value=1,
                )
                # Waiter array (zeros = empty slots; written PIDs as int32).
                priority_shm = shared_memory.SharedMemory(
                    name=_priority_shm_name(name, os.getpid()),
                    create=True,
                    size=PRIORITY_SHM_SIZE,
                )
                priority_shm.buf[:PRIORITY_SHM_SIZE] = b"\x00" * PRIORITY_SHM_SIZE
                priority_shm.close()
                summary += f"{name} semaphores: {width}\t\t"
            logger.info(summary)

            for sema_type in semaphores_spec_:
                try:
                    tracker = TimingTracker(name=sema_type, pid=os.getpid())
                    tracker.create_shared_memory().close()
                except FileExistsError as e:
                    logger.debug(
                        f"Shared memory already exists for tracking `{sema_type}`: {e}, resetting."
                    )
                    tracker = TimingTracker(name=sema_type, pid=os.getpid())
                    tracker.reset_timing_data()
            yield
    finally:
        _log_timing_summary(semaphores_spec_)

        for name in semaphores_spec_:
            try:
                Semaphore(name_to_posix_name(name, os.getpid())).unlink()
            except ExistentialError:  # pragma: no cover
                pass
            try:
                Semaphore(_priority_mutex_name(name, os.getpid())).unlink()
            except ExistentialError:  # pragma: no cover
                pass
            try:
                priority_shm = shared_memory.SharedMemory(
                    name=_priority_shm_name(name, os.getpid())
                )
                priority_shm.close()
                priority_shm.unlink()
            except FileNotFoundError:  # pragma: no cover
                pass
            except Exception as e:  # pylint: disable=broad-except
                raise RuntimeError(
                    f"Failed to cleanup priority shm for `{name}`."
                ) from e
            try:
                tracker = TimingTracker(name=name, pid=os.getpid())
                tracker.unlink()
            except Exception as e:  # pylint: disable=broad-except
                raise RuntimeError(
                    f"Failed to cleanup shared memory for tracking `{name}`."
                ) from e

        logger.info(f"Cleaned up semaphores created by process {os.getpid()}.")


@attrs.define
class DummySemaphore:  # pragma: no cover
    """
    Dummy semaphore class to be used if semaphores have not been configured.
    """

    name: str = ""

    def __enter__(self, *args):
        pass

    def __exit__(self, *args):
        pass

    def unlink(self):
        pass


@attrs.frozen
class DummyTimingTracker:  # pragma: no cover
    """
    Dummy timing tracker class to be used with dummy semaphores.
    """

    name: str

    def add_wait_time(self, wait_time: float) -> None:
        pass

    def add_lease_time(self, lease_time: float) -> None:
        pass

    def get_timing_data(self) -> tuple[float, float, int, float]:
        return 0.0, 0.0, 0, 0.0

    def reset_timing_data(self) -> None:
        pass

    def unlink(self) -> None:
        pass


@attrs.frozen
class TimedSemaphore(contextlib.AbstractContextManager):
    """
    Wrapper around a semaphore that tracks acquisition wait time and lease time globally.
    """

    semaphore: Semaphore | DummySemaphore | PriorityPidSemaphore
    name: str
    tracker_pid: int = 0
    timing_tracker: TimingTracker | DummyTimingTracker = attrs.field(init=False)
    _lease_start_time: float = attrs.field(init=False)

    def __attrs_post_init__(self):
        if isinstance(self.semaphore, DummySemaphore):
            object.__setattr__(self, "timing_tracker", DummyTimingTracker(name=self.name))
        else:
            object.__setattr__(
                self, "timing_tracker", TimingTracker(name=self.name, pid=self.tracker_pid)
            )
        object.__setattr__(self, "_lease_start_time", 0.0)

    def __enter__(self):
        start_time = time.perf_counter()
        result = self.semaphore.__enter__()
        wait_time = time.perf_counter() - start_time
        self.timing_tracker.add_wait_time(wait_time)

        object.__setattr__(self, "_lease_start_time", time.perf_counter())
        return result

    def __exit__(self, *args):
        # Sync queued GPU work before releasing the slot. Without this the
        # sem only gates kernel submission, not execution: kernels stay in
        # flight on the stream after release and the next holder's launches
        # interleave with ours instead of being serialized.
        if self.name == "cuda":
            try:
                import torch  # pylint: disable=import-outside-toplevel
            except ImportError:
                pass
            else:
                if torch.cuda.is_available():  # pragma: no cover
                    torch.cuda.synchronize()
        lease_time = time.perf_counter() - self._lease_start_time
        self.timing_tracker.add_lease_time(lease_time)
        return self.semaphore.__exit__(*args)

    def unlink(self):
        if hasattr(self.semaphore, "unlink"):
            self.semaphore.unlink()
        self.timing_tracker.unlink()


def semaphore(name: SemaphoreType) -> TimedSemaphore:
    """
    Fetches and returns the semaphore associated with the current process or
    any of its ancestors, walking up the process tree until one is found. Falls
    back to a dummy semaphore if none exists. Wraps in TimedSemaphore to track
    acquisition wait time.
    """
    if not name in get_args(SemaphoreType):
        raise ValueError(f"`{name}` is not a valid semaphore type.")
    candidate_pids = [os.getpid()]
    try:
        candidate_pids.extend(p.pid for p in psutil.Process().parents())
    except psutil.Error:  # pragma: no cover
        pass
    for pid in candidate_pids:
        try:
            # Probing the slot sem first so we don't construct the wider
            # priority wrapper unless the resource is actually configured here.
            Semaphore(name_to_posix_name(name, pid))
        except ExistentialError:
            continue
        return TimedSemaphore(
            semaphore=PriorityPidSemaphore(name=name, pid=pid),
            name=name,
            tracker_pid=pid,
        )
    dummy = DummySemaphore()
    return TimedSemaphore(semaphore=dummy, name=name)


def get_semaphore_stats(name: SemaphoreType) -> dict[str, float | int]:
    """Get usage stats for a semaphore type."""
    if name not in get_args(SemaphoreType):
        raise ValueError(f"`{name}` is not a valid semaphore type.")

    tracker = TimingTracker(name=name, pid=os.getpid())
    total_wait_time, total_lease_time, lease_count, start_time = tracker.get_timing_data()

    return {
        "total_wait_time": total_wait_time,
        "average_wait_time": total_wait_time / lease_count if lease_count > 0 else 0.0,
        "total_lease_time": total_lease_time,
        "lease_count": lease_count,
        "average_lease_time": total_lease_time / lease_count if lease_count > 0 else 0.0,
        "start_time": start_time,
    }


def reset_timing_data(name: SemaphoreType) -> None:
    """Reset timing data for a semaphore type."""
    if name not in get_args(SemaphoreType):
        raise ValueError(f"`{name}` is not a valid semaphore type.")

    tracker = TimingTracker(name=name, pid=os.getpid())
    tracker.reset_timing_data()


def reset_all_timing_data() -> None:
    """Reset timing data for all semaphore types."""
    for sema_type in get_args(SemaphoreType):
        reset_timing_data(sema_type)


def get_all_timing_data(
    semaphores_spec: dict[SemaphoreType, int]
) -> tuple[dict[SemaphoreType, dict[str, float | int]], dict[str, float | int]]:
    """Get timing data for all semaphore types with percentage breakdowns."""
    result = {}
    total_wait_time = 0.0
    total_lease_time = 0.0
    runtime = 0.0

    for sema_type, sema_count in semaphores_spec.items():
        stats = get_semaphore_stats(sema_type)
        result[sema_type] = stats
        total_wait_time += stats["total_wait_time"]
        total_lease_time += stats["total_lease_time"]
        result[sema_type]["semaphore_count"] = sema_count

        current_time = time.time()
        sema_runtime = current_time - stats["start_time"]
        if runtime == 0.0 or sema_runtime > runtime:
            runtime = sema_runtime

    for sema_type in semaphores_spec.keys():
        if sema_type in result:
            wait_time = result[sema_type]["total_wait_time"]
            lease_time = result[sema_type]["total_lease_time"]
            sema_count = int(result[sema_type]["semaphore_count"])

            wait_percentage = (wait_time / total_wait_time * 100) if total_wait_time > 0 else 0.0
            lease_percentage = (
                (lease_time / total_lease_time * 100) if total_lease_time > 0 else 0.0
            )
            result[sema_type]["wait_time_percentage"] = wait_percentage
            result[sema_type]["lease_time_percentage"] = lease_percentage

            total_potential_lease_time = sema_count * runtime
            runtime_utilization = (
                lease_time / total_potential_lease_time * 100
                if total_potential_lease_time > 0
                else 0.0
            )
            result[sema_type]["runtime_utilization_percentage"] = runtime_utilization

    total_lease_acquisitions = sum(
        stats["lease_count"]
        for stats in result.values()
        if isinstance(stats, dict) and "lease_count" in stats
    )

    summary = {
        "total_wait_time_all_semaphores": total_wait_time,
        "total_lease_time_all_semaphores": total_lease_time,
        "total_acquisitions": total_lease_acquisitions,
        "overall_average_wait": total_wait_time / total_lease_acquisitions
        if total_lease_acquisitions > 0
        else 0.0,
        "overall_average_lease": total_lease_time / total_lease_acquisitions
        if total_lease_acquisitions > 0
        else 0.0,
        "runtime": runtime,
    }

    return result, summary


def _log_timing_summary(  # pylint:disable=too-many-locals, too-many-statements
    semaphores_spec: dict[SemaphoreType, int]
) -> None:
    """Log a pretty-printed summary of semaphore timing data."""
    timing_data, summary_stats = get_all_timing_data(semaphores_spec)

    if not any(stats["lease_count"] > 0 for stats in timing_data.values()):
        logger.debug("No semaphore timing data available in this process.")
        return

    summary = ""
    summary += lrpad("  Semaphore Usage Summary  ", bounds="+", filler="=", length=80) + "\n"
    summary += lrpad("", bounds="|", length=80) + "\n"

    # Header for semaphore stats
    header = "Semaphore  Width   Acquis.   Wait Time  Lease Time   Avg Wait  Avg Lease"
    summary += lrpad(header, bounds="|", length=80) + "\n"

    summary += lrpad("", filler="-", bounds="|", length=80) + "\n"

    # Individual semaphore stats
    for sema_type, stats in timing_data.items():
        lease_count = stats["lease_count"]
        if lease_count > 0:
            # Format data row with proper alignment
            sema_count = stats["semaphore_count"]
            total_wait = stats["total_wait_time"]
            total_lease = stats["total_lease_time"]
            avg_wait = stats["average_wait_time"]
            avg_lease = stats["average_lease_time"]

            row = f"{sema_type:<9}{sema_count:>7}{lease_count:>9} "
            row += f"{total_wait:>11.3f}s{total_lease:>11.3f}s"
            row += f"{avg_wait:>10.3f}s{avg_lease:>10.3f}s"
            summary += lrpad(row, bounds="|", length=80) + "\n"

            # Add percentage info if meaningful
            wait_pct = stats["wait_time_percentage"]
            lease_pct = stats["lease_time_percentage"]
            if wait_pct > 0 or lease_pct > 0:
                pct_row = f"{'':<9}{'':<9}{'':<8}"
                pct_row += f"{f'({wait_pct:.1f}%)':>12}{f'({lease_pct:.1f}%)':>12}"
                summary += lrpad(pct_row, bounds="|", length=80) + "\n"

    # Summary totals
    if summary_stats:
        summary += lrpad("", filler="-", bounds="|", length=80) + "\n"
        total_acquisitions = summary_stats["total_acquisitions"]
        total_wait_all = summary_stats["total_wait_time_all_semaphores"]
        total_lease_all = summary_stats["total_lease_time_all_semaphores"]
        overall_avg_wait = summary_stats["overall_average_wait"]
        overall_avg_lease = summary_stats["overall_average_lease"]

        total_row = f"{'TOTAL':<12}{'':<4}{total_acquisitions:>9} "
        total_row += f"{total_wait_all:>11.3f}s{total_lease_all:>11.3f}s"
        total_row += f"{overall_avg_wait:>10.3f}s{overall_avg_lease:>10.3f}s"
        summary += lrpad(total_row, bounds="|", length=80) + "\n"

    # Runtime utilization section
    runtime = summary_stats["runtime"] if summary_stats else 0.0
    if runtime > 0:
        summary += lrpad("", bounds="|", length=80) + "\n"
        summary += lrpad("Runtime Utilization", bounds="|", length=80) + "\n"
        summary += lrpad("", filler="-", bounds="|", length=80) + "\n"

        util_header = "Semaphore  Width  Total Capacity  Utilization"
        summary += lrpad(util_header, bounds="|", length=80) + "\n"
        summary += lrpad("", filler="-", bounds="|", length=80) + "\n"

        for sema_type, stats in timing_data.items():
            if isinstance(stats, dict) and "lease_count" in stats:
                if stats["lease_count"] > 0:
                    sema_count = stats["semaphore_count"]
                    total_capacity = sema_count * runtime
                    utilization = stats["runtime_utilization_percentage"]

                    util_row = f"{sema_type:<9}{sema_count:>7}"
                    util_row += f"{total_capacity:>15.3f}s{utilization:>12.1f}%"
                    summary += lrpad(util_row, bounds="|", length=80) + "\n"

        runtime_info = f"Runtime: {runtime:.3f}s"
        summary += lrpad("", filler="-", bounds="|", length=80) + "\n"
        summary += lrpad(runtime_info, bounds="|", length=80) + "\n"

    summary += lrpad("", bounds="|", length=80) + "\n"
    summary += lrpad("", bounds="+", filler="=", length=80)

    logger.info(summary)
