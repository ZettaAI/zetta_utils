# pylint: disable=redefined-outer-name
import pytest

from zetta_utils.message_queues.sqs import utils as sqs_utils


@pytest.fixture
def mock_sqs_client(mocker):
    client = mocker.MagicMock()
    mocker.patch.object(sqs_utils, "get_sqs_client", return_value=client)
    mocker.patch.object(sqs_utils, "get_queue_url", return_value="https://sqs.example/q")
    return client


def test_get_queue_depth_parses_visible_and_in_flight(mock_sqs_client):
    mock_sqs_client.get_queue_attributes.return_value = {
        "Attributes": {
            "ApproximateNumberOfMessages": "5",
            "ApproximateNumberOfMessagesNotVisible": "2",
        }
    }

    visible, in_flight = sqs_utils.get_queue_depth("q", "us-east-1")

    assert (visible, in_flight) == (5, 2)
    call_kwargs = mock_sqs_client.get_queue_attributes.call_args.kwargs
    assert call_kwargs["QueueUrl"] == "https://sqs.example/q"
    assert set(call_kwargs["AttributeNames"]) == {
        "ApproximateNumberOfMessages",
        "ApproximateNumberOfMessagesNotVisible",
    }
