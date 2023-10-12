import time

import pytest

from zetta_utils.message_queues.local.queue import LocalQueue


def success_fn():
    return "Success"


def test_make_and_delete_local_queue():
    with LocalQueue("test_queue"):
        pass


def test_get_tq_queue():
    with LocalQueue("test_queue"):
        LocalQueue("test_queue")._get_tq_queue()  # pylint:disable = protected-access


def test_push_pull():
    with LocalQueue("test_queue") as q:
        payloads = {None, 1, "asdfadsfdsa", success_fn}
        q.push(list(payloads))
        time.sleep(0.1)
        result = q.pull(max_num=len(payloads))
        assert len(result) == len(payloads)
        received_payloads = {r.payload for r in result}
        assert received_payloads == payloads


def test_delete():
    with LocalQueue("test_queue") as q:
        q.push([None])
        time.sleep(0.1)
        result = q.pull(max_num=10)
        assert len(result) == 1
        result[0].acknowledge_fn()
        time.sleep(1.1)
        result_empty = q.pull()
        assert len(result_empty) == 0


def test_extend_lease():
    with LocalQueue("test_queue") as q:
        q.push([None])
        time.sleep(0.1)
        result = q.pull()
        assert len(result) == 1
        result[0].extend_lease_fn(3)
        time.sleep(1)
        result_empty = q.pull()
        assert len(result_empty) == 0
        time.sleep(2.1)
        result_nonempty = q.pull()
        assert len(result_nonempty) == 1


def test_double_init_exc():
    with pytest.raises(RuntimeError):
        with LocalQueue("test_queue"):
            with LocalQueue("test_queue"):
                pass
