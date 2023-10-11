import pytest

import zetta_utils.common.multiprocessing
from zetta_utils.common.multiprocessing import (
    get_persistent_process_pool,
    setup_persistent_process_pool,
)


def times2(x):
    return 2 * x


@pytest.mark.parametrize("args, expected", [[[1, 2, 3], [2, 4, 6]]])
def test_run_func(args, expected):
    with setup_persistent_process_pool(3):
        pool = get_persistent_process_pool()
        assert pool is not None
        assert list(pool.map(times2, args)) == expected


def test_no_init_with_1_proc():
    with setup_persistent_process_pool(1):
        assert get_persistent_process_pool() is None


def test_double_init_exc():
    with pytest.raises(RuntimeError):
        with setup_persistent_process_pool(2):
            with setup_persistent_process_pool(2):
                pass


def test_unalloc_nonexistent_exc():
    with pytest.raises(RuntimeError):
        with setup_persistent_process_pool(2):
            pool = get_persistent_process_pool()
            assert pool is not None
            pool.shutdown()
            zetta_utils.common.multiprocessing.PERSISTENT_PROCESS_POOL = None
        # exception on exiting context
