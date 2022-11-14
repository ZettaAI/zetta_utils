# pylint: disable=redefined-outer-name,exec-used
import time

import boto3
import pytest

import docker
from zetta_utils.mazepa import SQSExecutionQueue, TaskStatus
from zetta_utils.mazepa.tasks import _Task, _TaskFactory


def test_push_tasks_exc(mocker):
    mocker.patch("taskqueue.TaskQueue", lambda *args, **kwargs: mocker.MagicMock())
    sqseq = SQSExecutionQueue("q", outcome_queue_name=None)
    task = _Task(lambda: "outcome")
    with pytest.raises(RuntimeError):
        sqseq.push_tasks([task])


def test_pull_task_outcomes_exc(mocker):
    mocker.patch("taskqueue.TaskQueue", lambda *args, **kwargs: mocker.MagicMock())
    sqseq = SQSExecutionQueue("q", outcome_queue_name=None)
    with pytest.raises(RuntimeError):
        sqseq.pull_task_outcomes()


@pytest.fixture(scope="session")
def sqs_endpoint():
    """Ensure that SQS service is up and responsive."""

    # `port_for` takes a container port and returns the corresponding host port
    # port = docker_services.port_for("sqs", 9324)
    # url = f"http://{docker_ip}:{port}"
    # docker_services.wait_until_responsive(
    #    timeout=90.0, pause=0.1, check=lambda: is_responsive(url)
    # )
    client = docker.from_env()
    container = client.containers.run("vsouza/sqs-local", detach=True, ports={"9324": "9324"})

    timeout = 120
    stop_time = 3
    elapsed_time = 0
    while container.status != "running" and elapsed_time < timeout:
        time.sleep(stop_time)
        elapsed_time += stop_time
        container.reload()
    endpoint = "http://localhost:9324"
    yield endpoint
    container.kill()


@pytest.fixture
def work_queue(sqs_endpoint):
    region_name = "us-east-1"
    queue_name = "work-queue"
    sqs = boto3.client("sqs", region_name=region_name, endpoint_url=sqs_endpoint)
    queue = sqs.create_queue(QueueName=queue_name)
    yield (queue_name, region_name, sqs_endpoint)
    sqs.delete_queue(QueueUrl=queue["QueueUrl"])


@pytest.fixture
def outcome_queue(sqs_endpoint):
    region_name = "us-east-1"
    queue_name = "outcome-queue"
    sqs = boto3.client("sqs", region_name=region_name, endpoint_url=sqs_endpoint)
    queue = sqs.create_queue(QueueName=queue_name)
    yield (queue_name, region_name, sqs_endpoint)
    sqs.delete_queue(QueueUrl=queue["QueueUrl"])


def test_execution(work_queue, outcome_queue):
    work_queue_name, _, _ = work_queue
    outcome_queue_name, region_name, endpoint_url = outcome_queue
    queue = SQSExecutionQueue(
        name=work_queue_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
        outcome_queue_name=outcome_queue_name,
    )
    tasks = [
        _TaskFactory(lambda: "Success").make_task(),
        _TaskFactory(lambda: "Success").make_task(),
        _TaskFactory(lambda: exec("raise(Exception())")).make_task(),
    ]
    queue.push_tasks(tasks)
    tasks[0]()
    tasks[2]()
    outcomes = queue.pull_task_outcomes()
    assert outcomes[tasks[0].id_].status == TaskStatus.SUCCEEDED
    assert outcomes[tasks[0].id_].return_value == "Success"
    assert outcomes[tasks[2].id_].status == TaskStatus.FAILED
    assert outcomes[tasks[2].id_].return_value is None


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
    queue.push_tasks([_TaskFactory(lambda: "Success").make_task()])
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
    queue.push_tasks([_TaskFactory(lambda: "Success").make_task()])
    pulled_tasks = queue.pull_tasks()
    pulled_tasks[0]()
    pulled_tasks = queue.pull_tasks()
    assert len(pulled_tasks) == 0
