# pylint: disable=redefined-outer-name,unused-argument
import time
from typing import Any

import boto3
import coolname
import pytest
from moto import mock_sqs

import docker
from zetta_utils import common
from zetta_utils.message_queues.sqs.queue import SQSQueue

boto3.setup_default_session()


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


@pytest.fixture(scope="module")
def sqs_endpoint(aws_credentials):
    """Ensure that SQS service is up and responsive."""
    with mock_sqs():
        client = docker.from_env()  # type: ignore
        container = client.containers.run(
            "softwaremill/elasticmq-native", detach=True, ports={"9324/tcp": 9324}, remove=True
        )

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
        time.sleep(3)


@pytest.fixture
def raw_queue(sqs_endpoint):
    region_name = "us-east-1"
    queue_name = f"work-queue-{coolname.generate_slug(4)}"
    sqs = boto3.client("sqs", region_name=region_name, endpoint_url=sqs_endpoint)
    queue = sqs.create_queue(QueueName=queue_name)
    time.sleep(0.2)
    yield (queue_name, region_name, sqs_endpoint)
    sqs.delete_queue(QueueUrl=queue["QueueUrl"])
    time.sleep(0.2)


def success_fn():
    return "Success"


def test_push_pull(raw_queue, mocker):
    raw_queue_name, region_name, endpoint_url = raw_queue
    # mocker.patch("taskqueue.TaskQueue", lambda *args, **kwargs: mocker.MagicMock())
    q = SQSQueue[Any](
        raw_queue_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
        insertion_threads=0,
    )
    payloads = {None, 1, "asdfadsfdsa", success_fn}
    q.push(list(payloads))
    time.sleep(0.1)
    result = q.pull(max_num=len(payloads))
    assert len(result) == len(payloads)
    received_payloads = {r.payload for r in result}
    assert received_payloads == payloads


def test_delete(raw_queue):
    raw_queue_name, region_name, endpoint_url = raw_queue
    q = SQSQueue[Any](
        raw_queue_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
        insertion_threads=0,
        pull_lease_sec=1,
    )
    q.push([None])
    time.sleep(0.1)
    result = q.pull(max_num=10)
    assert len(result) == 1
    result[0].acknowledge_fn()
    time.sleep(1.1)
    result_empty = q.pull()
    assert len(result_empty) == 0


def test_extend_lease(raw_queue):
    raw_queue_name, region_name, endpoint_url = raw_queue
    q = SQSQueue[Any](
        raw_queue_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
        insertion_threads=0,
        pull_lease_sec=1,
    )
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
