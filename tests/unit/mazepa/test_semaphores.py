# pylint: disable=bare-except, c-extension-no-member

import os
from typing import List

import posix_ipc
import psutil
import pytest

from zetta_utils.mazepa.semaphores import (
    DummySemaphore,
    SemaphoreType,
    configure_semaphores,
    name_to_posix_name,
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


def test_get_grandparent_semaphore():
    grandpa_pid = psutil.Process(os.getppid()).ppid()
    try:
        sema = posix_ipc.Semaphore(name_to_posix_name("read", grandpa_pid))
        sema.unlink()
    except:
        pass
    sema = posix_ipc.Semaphore(name_to_posix_name("read", grandpa_pid), flags=posix_ipc.O_CREX)
    assert sema.name == semaphore("read").name
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
