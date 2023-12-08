# pylint: disable=redefined-outer-name,exec-used
import time
from functools import partial

import boto3
import coolname
import pytest

from zetta_utils import mazepa  # pylint: disable=all
from zetta_utils.mazepa import constants
from zetta_utils.mazepa.tasks import _TaskableOperation
from zetta_utils.message_queues.sqs.queue import SQSQueue

from ..message_queues.sqs.test_queue import aws_credentials, sqs_endpoint

boto3.setup_default_session()


@pytest.fixture
def task_queue(sqs_endpoint):
    region_name = "us-east-1"
    queue_name = f"task-queue-{coolname.generate_slug(4)}"
    sqs = boto3.client("sqs", region_name=region_name, endpoint_url=sqs_endpoint)
    queue = sqs.create_queue(QueueName=queue_name)
    time.sleep(0.2)
    yield SQSQueue(
        name=queue_name, region_name=region_name, endpoint_url=sqs_endpoint, pull_lease_sec=1
    )
    sqs.delete_queue(QueueUrl=queue["QueueUrl"])
    time.sleep(0.2)


@pytest.fixture
def outcome_queue(sqs_endpoint):
    region_name = "us-east-1"
    queue_name = f"outcome-queue-{coolname.generate_slug(4)}"
    sqs = boto3.client("sqs", region_name=region_name, endpoint_url=sqs_endpoint)
    queue = sqs.create_queue(QueueName=queue_name)
    time.sleep(0.2)
    yield SQSQueue(
        name=queue_name, region_name=region_name, endpoint_url=sqs_endpoint, pull_lease_sec=1
    )
    sqs.delete_queue(QueueUrl=queue["QueueUrl"])
    time.sleep(0.2)


@pytest.fixture
def queues_with_worker(task_queue, outcome_queue):
    worker = partial(
        mazepa.run_worker,
        task_queue=task_queue,
        outcome_queue=outcome_queue,
        sleep_sec=0.2,
        max_runtime=5.0,
        debug=True,
    )
    yield task_queue, outcome_queue, worker


def return_false_fn(*args, **kwargs):
    return False


@pytest.fixture
def queues_with_cancelling_worker(task_queue, outcome_queue):
    worker = partial(
        mazepa.run_worker,
        task_queue=task_queue,
        outcome_queue=outcome_queue,
        sleep_sec=0.2,
        max_runtime=5.0,
        debug=True,
        task_filter_fn=return_false_fn,
    )
    yield task_queue, outcome_queue, worker


def success_fn():
    return "Success"


def runtime_error_fn():
    raise RuntimeError


def sleep_0p1_fn():
    time.sleep(0.1)


def sleep_1p5_fn():
    time.sleep(1.5)


def sleep_5_fn():
    time.sleep(5)


def test_task_upkeep(queues_with_worker) -> None:
    task_queue, outcome_queue, worker = queues_with_worker
    task = _TaskableOperation(sleep_1p5_fn, upkeep_interval_sec=0.1).make_task()
    task_queue.push([task])
    worker()
    rdy_tasks = task_queue.pull()
    assert len(rdy_tasks) == 0
    time.sleep(2.0)
    rdy_tasks = task_queue.pull()
    assert len(rdy_tasks) == 0
    outcomes = outcome_queue.pull()
    assert len(outcomes) == 1


def test_task_upkeep_finish(queues_with_worker) -> None:
    task_queue, outcome_queue, worker = queues_with_worker
    task = _TaskableOperation(sleep_0p1_fn).make_task()
    task.upkeep_settings.interval_sec = 0.5
    task_queue.push([task])
    worker()
    time.sleep(3.0)
    rdy_tasks = task_queue.pull()
    assert len(rdy_tasks) == 0
    outcomes = outcome_queue.pull()
    assert len(outcomes) == 1


def test_cancelling_worker(queues_with_cancelling_worker) -> None:
    task_queue, outcome_queue, worker = queues_with_cancelling_worker
    task = _TaskableOperation(sleep_5_fn).make_task()
    task_queue.push([task])
    worker()
    time.sleep(1.0)
    rdy_tasks = task_queue.pull()
    assert len(rdy_tasks) == 0
    outcomes = outcome_queue.pull()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0].payload.outcome.exception, mazepa.exceptions.MazepaCancel)


def test_worker_task_pull_error(queues_with_worker, mocker) -> None:
    task_queue, outcome_queue, worker = queues_with_worker
    task_queue.pull = mocker.MagicMock(side_effect=[RuntimeError("hola")])
    with pytest.raises(RuntimeError):
        worker()
    outcomes = outcome_queue.pull()
    assert len(outcomes) == 1
    assert outcomes[0].payload.task_id == constants.UNKNOWN_TASK_ID
    exc = outcomes[0].payload.outcome.exception
    assert isinstance(exc, RuntimeError)
    assert "hola" in str(exc)
