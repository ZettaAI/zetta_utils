from __future__ import annotations

import contextlib
import os
import struct
import time
from multiprocessing import shared_memory
from typing import List, Literal, get_args

import attrs
from posix_ipc import (  # pylint: disable=no-name-in-module
    O_CREX,
    ExistentialError,
    Semaphore,
)

from zetta_utils import log
from zetta_utils.common.pprint import lrpad

logger = log.get_logger("mazepa")
SemaphoreType = Literal["read", "write", "cuda", "cpu"]

DEFAULT_SEMA_COUNT = 1
TIMING_FORMAT = "ddd"  # double for wait_time, lease_time, lease_count
TIMING_SIZE = struct.calcsize(TIMING_FORMAT)


@attrs.frozen
class TimingTracker:
    """
    Manages global semaphore timing data using shared memory.
    """

    name: str

    def _get_shared_memory_name(self) -> str:
        return f"zetta_semaphore_timing_{self.name}"

    def _get_shared_memory(self) -> shared_memory.SharedMemory:
        """Get existing shared memory (assumes it was already created by head node)."""
        name = self._get_shared_memory_name()
        return shared_memory.SharedMemory(name=name)

    def _create_shared_memory(self) -> shared_memory.SharedMemory:
        """Create new shared memory block (should only be called by head node)."""
        name = self._get_shared_memory_name()
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=TIMING_SIZE)
            # Initialize with zeros: wait_time, lease_time, lease_count
            shm.buf[:TIMING_SIZE] = struct.pack(TIMING_FORMAT, 0.0, 0.0, 0.0)
            return shm
        except FileExistsError:
            raise RuntimeError(f"Shared memory `{self.name}` with name `{name}` already exists.")

    def add_wait_time(self, wait_time: float) -> None:
        """Add wait time to global tracking."""
        shm = None
        try:
            shm = self._get_shared_memory()
            # Read current data: wait_time, lease_time, lease_count
            current_data = struct.unpack(TIMING_FORMAT, shm.buf[:TIMING_SIZE])
            total_wait_time = current_data[0] + wait_time
            lease_time = current_data[1]  # unchanged
            lease_count = current_data[2]  # unchanged
            # Write updated data
            shm.buf[:TIMING_SIZE] = struct.pack(
                TIMING_FORMAT, total_wait_time, lease_time, lease_count
            )
        finally:
            if shm:
                shm.close()

    def add_lease_time(self, lease_time: float) -> None:
        """Add lease time to global tracking."""
        shm = None
        try:
            shm = self._get_shared_memory()
            # Read current data: wait_time, lease_time, lease_count
            current_data = struct.unpack(TIMING_FORMAT, shm.buf[:TIMING_SIZE])
            wait_time = current_data[0]  # unchanged
            total_lease_time = current_data[1] + lease_time
            lease_count = current_data[2] + 1.0
            # Write updated data
            shm.buf[:TIMING_SIZE] = struct.pack(
                TIMING_FORMAT, wait_time, total_lease_time, lease_count
            )
        finally:
            if shm:
                shm.close()

    def get_timing_data(self) -> tuple[float, float, int]:
        """Get current timing data as (total_wait_time, total_lease_time, lease_count)."""
        shm = None
        try:
            shm = self._get_shared_memory()
            data = struct.unpack(TIMING_FORMAT, shm.buf[:TIMING_SIZE])
            return data[0], data[1], int(data[2])
        finally:
            if shm:
                shm.close()

    def reset_timing_data(self) -> None:
        """Reset timing data to zero."""
        shm = None
        try:
            shm = self._get_shared_memory()
            shm.buf[:TIMING_SIZE] = struct.pack(TIMING_FORMAT, 0.0, 0.0, 0.0)
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
            pass  # Already cleaned up
        except Exception as e:
            logger.warning(f"Failed to cleanup shared memory for {self.name}: {e}")


def name_to_posix_name(name: SemaphoreType, pid: int) -> str:  # pragma: no cover
    return f"zetta_utils_{pid}_{name}_semaphore"


@contextlib.contextmanager
def configure_semaphores(
    semaphores_spec: dict[SemaphoreType, int] | None = None
):  # pylint: disable=too-many-branches
    """
    Context manager for creating and destroying semaphores.
    """

    sema_types_to_check: List[SemaphoreType] = ["read", "write", "cuda", "cpu"]
    if semaphores_spec is not None:
        for name in semaphores_spec:
            if name not in get_args(SemaphoreType):
                raise ValueError(f"`{name}` is not a valid semaphore type.")
        try:
            for sema_type in sema_types_to_check:
                assert semaphores_spec[sema_type] >= 0
            semaphores_spec_ = semaphores_spec
        except KeyError as e:
            raise ValueError(
                "`semaphores_spec` given to `execute_with_pool` must contain "
                "`read`, `write`, `cuda`, and `cpu`."
            ) from e
        except AssertionError as e:
            raise ValueError("Number of semaphores must be nonnegative.") from e
        semaphores_spec_ = semaphores_spec
    else:
        semaphores_spec_ = {name: DEFAULT_SEMA_COUNT for name in sema_types_to_check}

    try:
        try:
            for name in semaphores_spec_:
                Semaphore(name_to_posix_name(name, os.getpid()), flags=0)
                raise RuntimeError(
                    f"Semaphore `{name}` with POSIX name "
                    "`{name_to_posix_name(name, os.getpid())}` "
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
                summary += f"{name} semaphores: {width}\t\t"
            logger.info(summary)

            for sema_type in semaphores_spec_:
                try:
                    tracker = TimingTracker(name=sema_type)
                    tracker._create_shared_memory().close()
                except Exception as e:
                    logger.debug(f"Failed to pre-create shared memory for {sema_type}: {e}")

            yield
    finally:
        try:
            _log_timing_summary(semaphores_spec_)

            for name in semaphores_spec_:
                try:
                    # Clean up POSIX semaphore
                    sema = Semaphore(name_to_posix_name(name, os.getpid()))
                    sema.unlink()

                    # Clean up shared memory (only head node should do this)
                    tracker = TimingTracker(name=name)
                    tracker.unlink()
                except ExistentialError:
                    # Semaphore doesn't exist, still try to cleanup shared memory
                    try:
                        tracker = TimingTracker(name=name)
                        tracker.unlink()
                    except Exception as e:
                        logger.debug(f"Failed to cleanup shared memory for {name}: {e}")

            logger.info(f"Cleaned up semaphores created by process {os.getpid()}.\n")
        except ExistentialError as e:
            raise RuntimeError(
                f"Trying to unlink semaphores created by process {os.getpid()} that do not exist."
            ) from e


@attrs.frozen
class DummySemaphore:  # pragma: no cover
    """
    Dummy semaphore class to be used if semaphores have not been configured.
    """

    def __enter__(self, *args):
        pass

    def __exit__(self, *args):
        pass

    def unlink(self):
        pass


@attrs.frozen
class TimedSemaphore:
    """
    Wrapper around a semaphore that tracks acquisition wait time and lease time globally.
    """

    semaphore: Semaphore | DummySemaphore
    name: str
    timing_tracker: TimingTracker = attrs.field(init=False)
    _lease_start_time: float = attrs.field(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "timing_tracker", TimingTracker(name=self.name))
        object.__setattr__(self, "_lease_start_time", 0.0)

    def __enter__(self):
        start_time = time.perf_counter()
        result = self.semaphore.__enter__()
        wait_time = time.perf_counter() - start_time
        self.timing_tracker.add_wait_time(wait_time)

        # Record lease start time
        object.__setattr__(self, "_lease_start_time", time.perf_counter())
        return result

    def __exit__(self, *args):
        # Calculate lease time and track it
        lease_time = time.perf_counter() - self._lease_start_time
        self.timing_tracker.add_lease_time(lease_time)
        return self.semaphore.__exit__(*args)

    def unlink(self):
        if hasattr(self.semaphore, "unlink"):
            self.semaphore.unlink()
        self.timing_tracker.unlink()


def semaphore(name: SemaphoreType) -> TimedSemaphore:
    """
    Fetches and returns either the semaphore associated with the current process,
    or the semaphore associated with the parent process, or a dummy semaphore,
    in that order. Wraps in TimedSemaphore to track acquisition wait time.
    """
    if not name in get_args(SemaphoreType):
        raise ValueError(f"`{name}` is not a valid semaphore type.")
    try:
        sema = Semaphore(name_to_posix_name(name, os.getpid()))
        return TimedSemaphore(semaphore=sema, name=name)
    except ExistentialError:
        try:
            sema = Semaphore(name_to_posix_name(name, os.getppid()))
            return TimedSemaphore(semaphore=sema, name=name)
        except ExistentialError:
            dummy = DummySemaphore()
            return TimedSemaphore(semaphore=dummy, name=name)


def _log_timing_summary(semaphores_spec: dict[str, int]) -> None:
    """Log a pretty-printed summary of semaphore timing data."""
    try:
        timing_data = get_all_timing_data(semaphores_spec)
        summary_stats = timing_data.pop("_summary", {})

        if not any(stats.get("lease_count", 0) > 0 for stats in timing_data.values()):
            logger.debug("No semaphore timing data available in this process.")
            return
    except Exception as e:
        logger.debug(f"Could not access semaphore timing data: {e}")
        return

    summary = ""
    summary += lrpad("  Semaphore Utilisation Summary  ", bounds="+", filler="=", length=80) + "\n"
    summary += lrpad("", bounds="|", length=80) + "\n"

    # Header for semaphore stats
    header = "Semaphore  Width   Acquis.   Wait Time  Lease Time   Avg Wait  Avg Lease"
    summary += lrpad(header, bounds="|", length=80) + "\n"

    summary += lrpad("", filler="-", bounds="|", length=80) + "\n"

    # Individual semaphore stats
    for sema_type, stats in timing_data.items():
        if isinstance(stats, dict) and "lease_count" in stats:
            lease_count = stats.get("lease_count", 0)
            if lease_count > 0:
                # Format data row with proper alignment
                row = f"{sema_type:<9}{stats.get('semaphore_count', 0):>7}{lease_count:>9} "
                row += f"{stats.get('total_wait_time', 0.0):>11.3f}s{stats.get('total_lease_time', 0.0):>11.3f}s"
                row += f"{stats.get('average_wait_time', 0.0):>10.3f}s{stats.get('average_lease_time', 0.0):>10.3f}s"
                summary += lrpad(row, bounds="|", length=80) + "\n"

                # Add percentage info if meaningful
                wait_pct = stats.get("wait_time_percentage", 0.0)
                lease_pct = stats.get("lease_time_percentage", 0.0)
                if wait_pct > 0 or lease_pct > 0:
                    pct_row = f"{'':<9}{'':<9}{'':<8}"
                    pct_row += f"{f'({wait_pct:.1f}%)':>12}{f'({lease_pct:.1f}%)':>12}"
                    summary += lrpad(pct_row, bounds="|", length=80) + "\n"

    # Summary totals
    if summary_stats:
        summary += lrpad("", filler="-", bounds="|", length=80) + "\n"
        total_row = f"{'TOTAL':<12}{'':<4}{summary_stats.get('total_acquisitions', 0):>9} "
        total_row += f"{summary_stats.get('total_wait_time_all_semaphores', 0.0):>11.3f}s{summary_stats.get('total_lease_time_all_semaphores', 0.0):>11.3f}s"
        total_row += f"{summary_stats.get('overall_average_wait', 0.0):>10.3f}s{summary_stats.get('overall_average_lease', 0.0):>10.3f}s"
        summary += lrpad(total_row, bounds="|", length=80) + "\n"

    summary += lrpad("", bounds="|", length=80) + "\n"
    summary += lrpad("", bounds="+", filler="=", length=80)

    logger.info(summary)


def get_total_wait_time(name: SemaphoreType) -> float:
    """Get total wait time for a semaphore type across all processes."""
    if name not in get_args(SemaphoreType):
        raise ValueError(f"`{name}` is not a valid semaphore type.")

    tracker = TimingTracker(name=name)
    total_wait_time, _, _ = tracker.get_timing_data()
    return total_wait_time


def get_total_lease_time(name: SemaphoreType) -> float:
    """Get total lease time for a semaphore type across all processes."""
    if name not in get_args(SemaphoreType):
        raise ValueError(f"`{name}` is not a valid semaphore type.")

    tracker = TimingTracker(name=name)
    _, total_lease_time, _ = tracker.get_timing_data()
    return total_lease_time


def get_semaphore_stats(name: SemaphoreType) -> dict[str, float | int]:
    """Get comprehensive stats for a semaphore type."""
    if name not in get_args(SemaphoreType):
        raise ValueError(f"`{name}` is not a valid semaphore type.")

    tracker = TimingTracker(name=name)
    total_wait_time, total_lease_time, lease_count = tracker.get_timing_data()

    return {
        "total_wait_time": total_wait_time,
        "average_wait_time": total_wait_time / lease_count if lease_count > 0 else 0.0,
        "total_lease_time": total_lease_time,
        "lease_count": lease_count,
        "average_lease_time": total_lease_time / lease_count if lease_count > 0 else 0.0,
    }


def reset_timing_data(name: SemaphoreType) -> None:
    """Reset timing data for a semaphore type."""
    if name not in get_args(SemaphoreType):
        raise ValueError(f"`{name}` is not a valid semaphore type.")

    tracker = TimingTracker(name=name)
    tracker.reset_timing_data()


def get_all_timing_data(semaphores_spec: dict[str, int]) -> dict[str, dict[str, float | int]]:
    """Get timing data for all semaphore types with percentage breakdowns."""
    result = {}
    total_wait_time = 0.0
    total_lease_time = 0.0

    for sema_type, sema_count in semaphores_spec.items():
        try:
            stats = get_semaphore_stats(sema_type)
            result[sema_type] = stats
            total_wait_time += stats["total_wait_time"]
            total_lease_time += stats["total_lease_time"]
            result[sema_type]["semaphore_count"] = sema_count
        except Exception as e:
            logger.warning(f"Failed to get timing data for {sema_type}: {e}")
            result[sema_type] = {
                "total_wait_time": 0.0,
                "average_wait_time": 0.0,
                "total_lease_time": 0.0,
                "lease_count": 0,
                "average_lease_time": 0.0,
                "semaphore_count": sema_count,
            }

    for sema_type in semaphores_spec.keys():
        if sema_type in result:
            wait_time = result[sema_type]["total_wait_time"]
            lease_time = result[sema_type]["total_lease_time"]
            wait_percentage = (wait_time / total_wait_time * 100) if total_wait_time > 0 else 0.0
            lease_percentage = (
                (lease_time / total_lease_time * 100) if total_lease_time > 0 else 0.0
            )
            result[sema_type]["wait_time_percentage"] = wait_percentage
            result[sema_type]["lease_time_percentage"] = lease_percentage

    total_lease_acquisitions = sum(
        stats["lease_count"]
        for stats in result.values()
        if isinstance(stats, dict) and "lease_count" in stats
    )

    result["_summary"] = {
        "total_wait_time_all_semaphores": total_wait_time,
        "total_lease_time_all_semaphores": total_lease_time,
        "total_acquisitions": total_lease_acquisitions,
        "overall_average_wait": total_wait_time / total_lease_acquisitions
        if total_lease_acquisitions > 0
        else 0.0,
        "overall_average_lease": total_lease_time / total_lease_acquisitions
        if total_lease_acquisitions > 0
        else 0.0,
    }

    return result


def reset_all_timing_data() -> None:
    """Reset timing data for all semaphore types."""
    for sema_type in get_args(SemaphoreType):
        try:
            reset_timing_data(sema_type)
        except Exception as e:
            logger.warning(f"Failed to reset timing data for {sema_type}: {e}")
