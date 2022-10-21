# pylint: disable=consider-using-set-comprehension
import uuid
import pytest
import boto3  # type: ignore
from moto import mock_sqs  # type: ignore

from zetta_utils import mazepa


@pytest.mark.parametrize("num_msg", [1, 8, 10, 15])
@mock_sqs
def test_send_receive_msgs(num_msg: int, mocker):
    region_name = "us-east-1"
    queue_name = "test-queue-x0"
    sqs = boto3.resource("sqs", region_name=region_name)
    sqs.create_queue(QueueName=queue_name)
    msgs = [str(uuid.uuid1()) for _ in range(num_msg)]
    max_msg_num = 10
    # mazepa.remote_execution_queues.sqs_utils.send_msg = wait_none()
    mocker.patch("tenacity.wait.wait_random.__call__", side_effect=lambda *args, **kwargs: 0)
    for msg in msgs:
        mazepa.remote_execution_queues.sqs_utils.send_msg(queue_name, region_name, msg)
    received_msgs = mazepa.remote_execution_queues.sqs_utils.receive_msgs(
        queue_name,
        region_name,
        max_msg_num=max_msg_num,
    )
    assert set([m.body for m in received_msgs]) == set(msgs[:max_msg_num])


@mock_sqs
def test_receive_msgs_early_stop_by_time(mocker):
    region_name = "us-east-1"
    queue_name = "test-queue-x0"
    sqs = boto3.resource("sqs", region_name=region_name)
    sqs.create_queue(QueueName=queue_name)
    num_msg = 100
    msgs = [str(uuid.uuid1()) for _ in range(num_msg)]
    mocker.patch("tenacity.wait.wait_random.__call__", side_effect=lambda *args, **kwargs: 0)

    for msg in msgs:
        mazepa.remote_execution_queues.sqs_utils.send_msg(queue_name, region_name, msg)

    received_msgs = mazepa.remote_execution_queues.sqs_utils.receive_msgs(
        queue_name,
        region_name,
        max_time_sec=0.00001,
    )
    assert len(received_msgs) < num_msg
