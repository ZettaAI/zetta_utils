"""Tests for PubSub module __init__"""

from zetta_utils.message_queues.pubsub import PubSubPullQueue, __all__


def test_pubsub_imports():
    """Test that PubSubPullQueue is properly exported"""
    assert "PubSubPullQueue" in __all__
    assert PubSubPullQueue is not None
