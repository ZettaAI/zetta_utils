from __future__ import annotations

import contextlib
import os
from typing import List, Literal, get_args

import attrs
from posix_ipc import (  # pylint: disable=no-name-in-module
    O_CREX,
    ExistentialError,
    Semaphore,
)

from zetta_utils import log

logger = log.get_logger("zetta_utils")
SemaphoreType = Literal["read", "write", "cuda", "cpu"]

DEFAULT_SEMA_COUNT = 1


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
            yield
    finally:
        try:
            for name in semaphores_spec_:
                sema = Semaphore(name_to_posix_name(name, os.getpid()))
                sema.unlink()
            logger.info(f"Cleaned up semaphores created by process {os.getpid()}.")
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


def semaphore(name: SemaphoreType) -> Semaphore:
    """
    Fetches and returns either the semaphore associated with the current process,
    or the semaphore associated with the parent process, or a dummy semaphore, in that order.
    """
    if not name in get_args(SemaphoreType):
        raise ValueError(f"`{name}` is not a valid semaphore type.")
    try:
        return Semaphore(name_to_posix_name(name, os.getpid()))
    except ExistentialError:
        try:
            return Semaphore(name_to_posix_name(name, os.getppid()))
        except ExistentialError:
            return DummySemaphore()
