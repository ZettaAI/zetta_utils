"""
Shared memory tracker for worker pool activity and idle timeout management.
"""
from __future__ import annotations

import struct
import time
from multiprocessing import shared_memory

import attrs

from zetta_utils import log

logger = log.get_logger("mazepa")

# Format: last_activity_time (double), active_worker_count (int)
ACTIVITY_FORMAT = "di"
ACTIVITY_SIZE = struct.calcsize(ACTIVITY_FORMAT)


@attrs.frozen
class PoolActivityTracker:
    """
    Manages pool-wide activity tracking using shared memory.

    Tracks:
    - last_activity_time: When any worker last received or completed a task
    - active_worker_count: Number of workers currently processing tasks
    """

    pool_name: str

    def _get_shared_memory_name(self) -> str:
        return f"zetta_pool_activity_{self.pool_name}"

    def _get_shared_memory(self) -> shared_memory.SharedMemory:
        """Get existing shared memory."""
        name = self._get_shared_memory_name()
        return shared_memory.SharedMemory(name=name)

    def create_shared_memory(self) -> shared_memory.SharedMemory:
        """Create new shared memory block (called by pool manager)."""
        name = self._get_shared_memory_name()
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=ACTIVITY_SIZE)
            # Initialize: last_activity_time = now, active_worker_count = 0
            shm.buf[:ACTIVITY_SIZE] = struct.pack(ACTIVITY_FORMAT, time.time(), 0)
            logger.info(f"Created pool activity tracker: {name}")
            return shm
        except FileExistsError as exc:
            raise FileExistsError(
                f"Pool activity tracker `{self.pool_name}` already exists."
            ) from exc

    def update_activity_time(self) -> None:
        """Update last activity time to now (called when work happens)."""
        shm = None
        try:
            shm = self._get_shared_memory()
            current_data = struct.unpack(ACTIVITY_FORMAT, shm.buf[:ACTIVITY_SIZE])
            active_count = current_data[1]
            # Update activity time to now, keep worker count
            shm.buf[:ACTIVITY_SIZE] = struct.pack(ACTIVITY_FORMAT, time.time(), active_count)
        except FileNotFoundError as e:
            logger.warning(f"Pool activity tracker not found: {e}")
        finally:
            if shm:
                shm.close()

    def increment_active_workers(self) -> None:
        """Increment active worker count (called when starting task processing)."""
        shm = None
        try:
            shm = self._get_shared_memory()
            current_data = struct.unpack(ACTIVITY_FORMAT, shm.buf[:ACTIVITY_SIZE])
            last_activity = current_data[0]
            active_count = current_data[1] + 1
            shm.buf[:ACTIVITY_SIZE] = struct.pack(ACTIVITY_FORMAT, last_activity, active_count)
        except FileNotFoundError as e:
            logger.warning(f"Pool activity tracker not found: {e}")
        finally:
            if shm:
                shm.close()

    def decrement_active_workers(self) -> None:
        """Decrement active worker count (called when finishing task processing)."""
        shm = None
        try:
            shm = self._get_shared_memory()
            current_data = struct.unpack(ACTIVITY_FORMAT, shm.buf[:ACTIVITY_SIZE])
            last_activity = current_data[0]
            assert current_data[1] > 0
            active_count = current_data[1] - 1
            shm.buf[:ACTIVITY_SIZE] = struct.pack(ACTIVITY_FORMAT, last_activity, active_count)
        except FileNotFoundError as e:
            logger.warning(f"Pool activity tracker not found: {e}")
        finally:
            if shm:
                shm.close()

    def get_activity_data(self) -> tuple[float, int]:
        """
        Get activity data.

        Returns:
            (last_activity_time, active_worker_count)
        """
        shm = None
        try:
            shm = self._get_shared_memory()
            data = struct.unpack(ACTIVITY_FORMAT, shm.buf[:ACTIVITY_SIZE])
            return data[0], data[1]
        except FileNotFoundError as e:
            logger.warning(f"Pool activity tracker not found: {e}")
            # If tracker doesn't exist, return sensible defaults
            return time.time(), 0
        finally:
            if shm:
                shm.close()

    def check_idle_timeout(self, idle_timeout: float) -> bool:
        """
        Check if pool has been idle for longer than timeout.

        Returns True if:
        - No workers are active AND
        - Time since last activity > idle_timeout
        """
        last_activity, active_count = self.get_activity_data()
        time_since_activity = time.time() - last_activity
        is_idle = active_count == 0 and time_since_activity > idle_timeout

        if is_idle:
            logger.info(
                f"Pool idle timeout check: {time_since_activity:.1f}s since last activity, "
                f"{active_count} active workers"
            )

        return is_idle

    def unlink(self) -> None:
        """Cleanup shared memory."""
        shm_name = self._get_shared_memory_name()
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
            logger.info(f"Cleaned up pool activity tracker: {shm_name}")
        except FileNotFoundError:
            pass
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(f"Failed to cleanup pool activity tracker: {e}")
