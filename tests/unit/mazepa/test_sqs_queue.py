# pylint: disable=redefined-outer-name,exec-used
import time
from multiprocessing import Process

import boto3
import coolname
import pytest
from moto import mock_sqs

import docker

# from zetta_utils import mazepa_layer_processing, common # pylint: disable=all
from zetta_utils import builder, common, mazepa
from zetta_utils.mazepa import SQSExecutionQueue
from zetta_utils.mazepa.tasks import Task, _TaskableOperation

boto3.setup_default_session()

builder.register("mazepa.SQSExecutionQueue")(SQSExecutionQueue)


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


def test_push_tasks_exc(mocker):
    mocker.patch("taskqueue.TaskQueue", lambda *args, **kwargs: mocker.MagicMock())
    sqseq = SQSExecutionQueue("q", outcome_queue_name=None)
    task = Task(success_fn)
    with pytest.raises(RuntimeError):
        sqseq.push_tasks([task])


def test_pull_task_outcomes_exc(mocker):
    mocker.patch("taskqueue.TaskQueue", lambda *args, **kwargs: mocker.MagicMock())
    sqseq = SQSExecutionQueue("q", outcome_queue_name=None)
    with pytest.raises(RuntimeError):
        sqseq.pull_task_outcomes()


@pytest.fixture(scope="session")
def aws_credentials():
    with common.set_env_ctx_mngr(
        AWS_ACCESS_KEY_ID="testing",
        AWS_SECRET_ACCESS_KEY="testing",
        AWS_SECURITY_TOKEN="testing",
        AWS_SESSION_TOKEN="testing",
        AWS_DEFAULT_REGION="us-east-1",
    ):
        yield


@pytest.fixture(scope="session")
def sqs_endpoint(aws_credentials):
    """Ensure that SQS service i s up and responsive."""
    with mock_sqs():
        client = docker.from_env()
        container = client.containers.run("graze/sqs-local", detach=True, ports={"9324": "9324"})

        timeout = 120
        stop_time = 1
        elapsed_time = 0
        while container.status != "running" and elapsed_time < timeout:
            time.sleep(stop_time)
            elapsed_time += stop_time
            container.reload()
        endpoint = "http://localhost:9324"
        yield endpoint
        container.kill()
        time.sleep(0.2)


@pytest.fixture
def work_queue(sqs_endpoint):
    region_name = "us-east-1"
    queue_name = f"work-queue-{coolname.generate_slug(4)}"
    sqs = boto3.client("sqs", region_name=region_name, endpoint_url=sqs_endpoint)
    queue = sqs.create_queue(QueueName=queue_name)
    time.sleep(0.2)
    yield (queue_name, region_name, sqs_endpoint)
    sqs.delete_queue(QueueUrl=queue["QueueUrl"])
    time.sleep(0.2)


@pytest.fixture
def outcome_queue(sqs_endpoint):
    region_name = "us-east-1"
    queue_name = f"outcome-queue-{coolname.generate_slug(4)}"
    sqs = boto3.client("sqs", region_name=region_name, endpoint_url=sqs_endpoint)
    queue = sqs.create_queue(QueueName=queue_name)
    time.sleep(0.2)
    yield (queue_name, region_name, sqs_endpoint)
    sqs.delete_queue(QueueUrl=queue["QueueUrl"])
    time.sleep(0.2)


@pytest.fixture
def queue_with_worker(work_queue, outcome_queue):
    work_queue_name, _, _ = work_queue
    outcome_queue_name, region_name, endpoint_url = outcome_queue

    # Cannot start another process by passing a queue object here,
    # As queue object contains a TaskQueue which contains boto objects,
    # which are not safe to pass to other processes. Using builder instead
    # queue = SQSExecutionQueue(
    #    name=work_queue_name,
    #    region_name=region_name,
    #    endpoint_url=endpoint_url,
    #    outcome_queue_name=outcome_queue_name,
    #    pull_lease_sec=2,
    # )
    # worker_p = Process(
    #    target=mazepa.run_worker,
    #    kwargs={"exec_queue": queue, "sleep_sec": 0.05, "max_runtime": 5.0},
    # )
    # from zetta_utils import mazepa_layer_processing  # pylint: disable=all

    worker_p = Process(
        target=builder.build,
        args=(
            {
                "@type": "mazepa.run_worker",
                "exec_queue": {
                    "@type": "mazepa.SQSExecutionQueue",
                    "name": work_queue_name,
                    "outcome_queue_name": outcome_queue_name,
                    "endpoint_url": endpoint_url,
                    "pull_lease_sec": 1,
                },
                "sleep_sec": 0.2,
                "debug": True,
                "max_runtime": 5.0,
            },
        ),
    )
    worker_p.start()
    yield work_queue_name, outcome_queue_name, region_name, endpoint_url
    worker_p.join()
    time.sleep(0.2)


@builder.register("return_false_fn")
def return_false_fn(*args, **kwargs):
    return False


@pytest.fixture
def queue_with_cancelling_worker(work_queue, outcome_queue):
    work_queue_name, _, _ = work_queue
    outcome_queue_name, region_name, endpoint_url = outcome_queue

    worker_p = Process(
        target=builder.build,
        args=(
            {
                "@type": "mazepa.run_worker",
                "exec_queue": {
                    "@type": "mazepa.SQSExecutionQueue",
                    "name": work_queue_name,
                    "outcome_queue_name": outcome_queue_name,
                    "endpoint_url": endpoint_url,
                },
                "sleep_sec": 0.2,
                "max_runtime": 5.0,
                "task_filter_fn": {"@type": "return_false_fn", "@mode": "partial"},
            },
        ),
    )
    worker_p.start()
    yield work_queue_name, outcome_queue_name, region_name, endpoint_url
    worker_p.join()
    time.sleep(0.2)


def test_task_upkeep(queue_with_worker) -> None:
    work_queue_name, outcome_queue_name, region_name, endpoint_url = queue_with_worker
    queue = SQSExecutionQueue(
        name=work_queue_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
        outcome_queue_name=outcome_queue_name,
    )
    task = _TaskableOperation(
        sleep_1p5_fn,
        upkeep_interval_sec=0.1,
    ).make_task()
    queue.push_tasks([task])
    time.sleep(1.0)
    rdy_tasks = queue.pull_tasks()
    assert len(rdy_tasks) == 0
    time.sleep(2.0)
    rdy_tasks = queue.pull_tasks()
    assert len(rdy_tasks) == 0


def test_execution(work_queue, outcome_queue):
    work_queue_name, _, _ = work_queue
    outcome_queue_name, region_name, endpoint_url = outcome_queue
    queue = SQSExecutionQueue(
        name=work_queue_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
        outcome_queue_name=outcome_queue_name,
        insertion_threads=0,
    )
    tasks = [
        _TaskableOperation(success_fn).make_task(),
        _TaskableOperation(success_fn).make_task(),
    ]
    failing_task = _TaskableOperation(runtime_error_fn).make_task()
    tasks.append(failing_task)
    queue.push_tasks(tasks)
    tasks[0]()
    tasks[2]()
    outcomes = queue.pull_task_outcomes()
    assert outcomes[tasks[0].id_].exception is None
    assert outcomes[tasks[0].id_].return_value == "Success"
    assert outcomes[tasks[2].id_].exception is not None
    assert outcomes[tasks[2].id_].return_value is None


def test_task_upkeep_finish(queue_with_worker) -> None:
    work_queue_name, outcome_queue_name, region_name, endpoint_url = queue_with_worker
    queue = SQSExecutionQueue(
        name=work_queue_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
        outcome_queue_name=outcome_queue_name,
        pull_lease_sec=2,
    )
    task = _TaskableOperation(sleep_0p1_fn).make_task()
    task.upkeep_settings.interval_sec = 0.5
    queue.push_tasks([task])
    time.sleep(3.0)
    rdy_tasks = queue.pull_tasks()
    assert len(rdy_tasks) == 0
    outcomes = queue.pull_task_outcomes()
    assert len(outcomes) == 1


def test_cancelling_worker(queue_with_cancelling_worker) -> None:
    work_queue_name, outcome_queue_name, region_name, endpoint_url = queue_with_cancelling_worker
    queue = SQSExecutionQueue(
        name=work_queue_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
        outcome_queue_name=outcome_queue_name,
        pull_lease_sec=2,
    )
    task: mazepa.Task = _TaskableOperation(sleep_5_fn).make_task()
    queue.push_tasks([task])
    time.sleep(1.0)
    rdy_tasks = queue.pull_tasks()
    assert len(rdy_tasks) == 0
    outcomes = queue.pull_task_outcomes()
    assert len(outcomes) == 1
    assert isinstance(outcomes[task.id_].exception, mazepa.exceptions.MazepaCancel)


def test_polling_not_done(work_queue, outcome_queue):
    work_queue_name, _, _ = work_queue
    outcome_queue_name, region_name, endpoint_url = outcome_queue
    queue = SQSExecutionQueue(
        name=work_queue_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
        outcome_queue_name=outcome_queue_name,
        pull_lease_sec=1,
    )
    queue.push_tasks([_TaskableOperation(success_fn).make_task()])
    pulled_tasks = queue.pull_tasks()
    time.sleep(1.5)
    pulled_tasks = queue.pull_tasks()
    assert len(pulled_tasks) == 1


def test_polling_done(work_queue, outcome_queue):
    work_queue_name, _, _ = work_queue
    outcome_queue_name, region_name, endpoint_url = outcome_queue
    queue = SQSExecutionQueue(
        name=work_queue_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
        outcome_queue_name=outcome_queue_name,
        pull_lease_sec=1,
    )
    queue.push_tasks([_TaskableOperation(success_fn).make_task()])
    pulled_tasks = queue.pull_tasks()
    pulled_tasks[0]()
    time.sleep(0.1)
    pulled_tasks = queue.pull_tasks()
    assert len(pulled_tasks) == 0
