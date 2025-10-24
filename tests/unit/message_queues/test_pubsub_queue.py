"""Tests for PubSub queue implementation"""

# pylint: disable=unused-argument,redefined-outer-name,protected-access

import gzip
import json
import pickle
from unittest.mock import Mock, patch

import pytest

from zetta_utils.message_queues.pubsub.queue import PubSubPullQueue


@pytest.fixture
def mock_subscriber():
    """Create a mock Pub/Sub subscriber client"""
    with patch(
        "zetta_utils.message_queues.pubsub.queue.pubsub_v1.SubscriberClient"
    ) as mock_client_class:
        mock_client = Mock()
        mock_client.subscription_path = Mock(
            return_value="projects/test-project/subscriptions/test-sub"
        )
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def pubsub_queue(mock_subscriber):
    """Create a PubSubPullQueue instance"""
    queue = PubSubPullQueue(
        name="test_queue",
        project_id="test-project",
        subscription_name="test-sub",
        pull_lease_sec=60,
    )
    return queue


def create_mock_message(data: bytes, attributes: dict = None):
    """Helper to create a mock Pub/Sub message"""
    message = Mock()
    message.data = data
    message.attributes = attributes or {}

    received_message = Mock()
    received_message.message = message
    received_message.ack_id = "test-ack-id"

    return received_message


def test_pubsub_queue_initialization(pubsub_queue, mock_subscriber):
    """Test PubSubPullQueue initialization"""
    assert pubsub_queue.name == "test_queue"
    assert pubsub_queue.project_id == "test-project"
    assert pubsub_queue.subscription_name == "test-sub"
    assert pubsub_queue.pull_lease_sec == 60
    assert pubsub_queue.return_immediately is True
    assert pubsub_queue._subscriber == mock_subscriber
    assert (
        pubsub_queue._subscription_path
        == "projects/test-project/subscriptions/test-sub"
    )


def test_pull_json_messages(pubsub_queue, mock_subscriber):
    """Test pulling JSON-formatted messages"""
    # Setup mock response
    payload = {"key": "value", "number": 123}
    message_data = json.dumps(payload).encode("utf-8")
    mock_message = create_mock_message(
        message_data, {"attr1": "value1"}
    )

    mock_response = Mock()
    mock_response.received_messages = [mock_message]
    mock_subscriber.pull.return_value = mock_response

    # Pull messages
    messages = pubsub_queue.pull(max_num=10)

    # Verify
    assert len(messages) == 1
    assert messages[0].payload["key"] == "value"
    assert messages[0].payload["number"] == 123
    assert messages[0].payload["_pubsub_attributes"]["attr1"] == "value1"
    assert callable(messages[0].acknowledge_fn)
    assert callable(messages[0].extend_lease_fn)

    # Verify pull was called with correct parameters
    mock_subscriber.pull.assert_called_once()
    call_kwargs = mock_subscriber.pull.call_args[1]
    assert call_kwargs["max_messages"] == 10
    assert call_kwargs["return_immediately"] is True


def test_pull_pickled_messages(pubsub_queue, mock_subscriber):
    """Test pulling pickle-formatted messages"""
    # Setup mock response with pickled data
    payload = {"operation_id": 123, "data": [1, 2, 3]}
    message_data = pickle.dumps(payload)
    mock_message = create_mock_message(
        message_data, {"table_id": "test_table"}
    )

    mock_response = Mock()
    mock_response.received_messages = [mock_message]
    mock_subscriber.pull.return_value = mock_response

    # Pull messages
    messages = pubsub_queue.pull(max_num=5)

    # Verify
    assert len(messages) == 1
    assert messages[0].payload["operation_id"] == 123
    assert messages[0].payload["data"] == [1, 2, 3]
    assert messages[0].payload["_pubsub_attributes"]["table_id"] == "test_table"


def test_pull_gzipped_messages(pubsub_queue, mock_subscriber):
    """Test pulling gzip-compressed messages"""
    # Setup mock response with gzipped data
    payload = {"compressed": "data"}
    json_data = json.dumps(payload).encode("utf-8")
    message_data = gzip.compress(json_data)
    mock_message = create_mock_message(message_data)

    mock_response = Mock()
    mock_response.received_messages = [mock_message]
    mock_subscriber.pull.return_value = mock_response

    # Pull messages
    messages = pubsub_queue.pull(max_num=10)

    # Verify
    assert len(messages) == 1
    assert messages[0].payload["compressed"] == "data"


def test_pull_no_messages(pubsub_queue, mock_subscriber):
    """Test pulling when no messages are available"""
    mock_response = Mock()
    mock_response.received_messages = []
    mock_subscriber.pull.return_value = mock_response

    messages = pubsub_queue.pull(max_num=10)

    assert len(messages) == 0


def test_pull_invalid_json(pubsub_queue, mock_subscriber):
    """Test handling of invalid JSON messages"""
    # Setup mock response with invalid JSON
    message_data = b"not valid json"
    mock_message = create_mock_message(message_data)

    mock_response = Mock()
    mock_response.received_messages = [mock_message]
    mock_subscriber.pull.return_value = mock_response

    # Pull messages - should skip invalid message
    messages = pubsub_queue.pull(max_num=10)

    assert len(messages) == 0


def test_pull_invalid_utf8(pubsub_queue, mock_subscriber):
    """Test handling of invalid UTF-8 data"""
    # Setup mock response with invalid UTF-8
    message_data = b"\xff\xfe invalid utf-8"
    mock_message = create_mock_message(message_data)

    mock_response = Mock()
    mock_response.received_messages = [mock_message]
    mock_subscriber.pull.return_value = mock_response

    # Pull messages - should skip invalid message
    messages = pubsub_queue.pull(max_num=10)

    assert len(messages) == 0


def test_acknowledge_message(pubsub_queue, mock_subscriber):
    """Test acknowledging a message"""
    payload = {"test": "data"}
    message_data = json.dumps(payload).encode("utf-8")
    mock_message = create_mock_message(message_data)

    mock_response = Mock()
    mock_response.received_messages = [mock_message]
    mock_subscriber.pull.return_value = mock_response

    messages = pubsub_queue.pull(max_num=10)

    # Acknowledge the message
    messages[0].acknowledge_fn()

    # Verify acknowledge was called
    mock_subscriber.acknowledge.assert_called_once_with(
        subscription=pubsub_queue._subscription_path,
        ack_ids=["test-ack-id"],
    )


def test_extend_lease(pubsub_queue, mock_subscriber):
    """Test extending message lease"""
    payload = {"test": "data"}
    message_data = json.dumps(payload).encode("utf-8")
    mock_message = create_mock_message(message_data)

    mock_response = Mock()
    mock_response.received_messages = [mock_message]
    mock_subscriber.pull.return_value = mock_response

    messages = pubsub_queue.pull(max_num=10)

    # Extend lease
    messages[0].extend_lease_fn()

    # Verify extend was called
    mock_subscriber.modify_ack_deadline.assert_called_once_with(
        subscription=pubsub_queue._subscription_path,
        ack_ids=["test-ack-id"],
        ack_deadline_seconds=60,
    )


def test_pull_multiple_messages(pubsub_queue, mock_subscriber):
    """Test pulling multiple messages"""
    # Setup mock response with multiple messages
    messages_data = [
        json.dumps({"id": 1}).encode("utf-8"),
        json.dumps({"id": 2}).encode("utf-8"),
        json.dumps({"id": 3}).encode("utf-8"),
    ]
    mock_messages = [create_mock_message(data) for data in messages_data]

    mock_response = Mock()
    mock_response.received_messages = mock_messages
    mock_subscriber.pull.return_value = mock_response

    # Pull messages
    messages = pubsub_queue.pull(max_num=10)

    # Verify
    assert len(messages) == 3
    assert messages[0].payload["id"] == 1
    assert messages[1].payload["id"] == 2
    assert messages[2].payload["id"] == 3


def test_custom_return_immediately(mock_subscriber):
    """Test queue with custom return_immediately setting"""
    queue = PubSubPullQueue(
        name="test_queue",
        project_id="test-project",
        subscription_name="test-sub",
        return_immediately=False,
    )

    mock_response = Mock()
    mock_response.received_messages = []
    mock_subscriber.pull.return_value = mock_response

    queue.pull(max_num=10)

    # Verify pull was called with return_immediately=False
    call_kwargs = mock_subscriber.pull.call_args[1]
    assert call_kwargs["return_immediately"] is False
