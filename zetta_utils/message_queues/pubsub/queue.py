# mypy: disable-error-code=attr-defined
"""Google Cloud Pub/Sub message queue implementation."""

from __future__ import annotations

import gzip
import json
import pickle
from typing import Any, TypeVar

import attrs
from google.cloud import pubsub_v1
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.common.partial import ComparablePartial
from zetta_utils.message_queues.base import PullMessageQueue, ReceivedMessage

T = TypeVar("T")


def _acknowledge_message(
    ack_id: str,
    subscriber_client: pubsub_v1.SubscriberClient,
    subscription_path: str,
):
    """Acknowledge a message from Pub/Sub."""
    subscriber_client.acknowledge(subscription=subscription_path, ack_ids=[ack_id])


def _extend_message_lease(
    ack_id: str,
    subscriber_client: pubsub_v1.SubscriberClient,
    subscription_path: str,
    ack_deadline_seconds: int,
):
    """Extend the acknowledgment deadline for a message."""
    subscriber_client.modify_ack_deadline(
        subscription=subscription_path,
        ack_ids=[ack_id],
        ack_deadline_seconds=ack_deadline_seconds,
    )


@builder.register("PubSubPullQueue")
@typechecked
@attrs.mutable
class PubSubPullQueue(PullMessageQueue[T]):
    """
    Google Cloud Pub/Sub pull-based message queue.

    This queue pulls messages from a GCP Pub/Sub subscription.
    """

    name: str
    project_id: str
    subscription_name: str
    pull_lease_sec: int = 60
    return_immediately: bool = True
    _subscriber: Any = attrs.field(init=False, default=None)
    _subscription_path: str = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        """Initialize Pub/Sub subscriber client after attrs initialization."""
        self._subscriber = pubsub_v1.SubscriberClient()
        self._subscription_path = self._subscriber.subscription_path(
            self.project_id, self.subscription_name
        )

    def pull(self, max_num: int = 10) -> list[ReceivedMessage[T]]:
        """
        Pull messages from the Pub/Sub subscription.

        Args:
            max_num: Maximum number of messages to pull

        Returns:
            List of received messages with their payloads
        """
        results = []

        response = self._subscriber.pull(
            subscription=self._subscription_path,
            max_messages=max_num,
            return_immediately=self.return_immediately,
        )

        for received_message in response.received_messages:
            try:
                # Try to decode the message data
                raw_data = received_message.message.data
                print(f"[DEBUG] Raw message data (first 50 bytes): {raw_data[:50]}")
                print(
                    "[DEBUG] Message attributes: "
                    f"{received_message.message.attributes}"
                )

                # Check message format
                if raw_data[:2] == b'\x1f\x8b':  # gzip magic number
                    print("[DEBUG] Detected gzip compression")
                    data = gzip.decompress(raw_data).decode("utf-8")
                    payload = json.loads(data)
                elif raw_data[:2] in (b'\x80\x04', b'\x80\x03'):  # pickle
                    print("[DEBUG] Detected pickle format")
                    payload = pickle.loads(raw_data)
                    print(f"[DEBUG] Unpickled payload: {payload}")
                else:
                    data = raw_data.decode("utf-8")
                    payload = json.loads(data)

                # Add message attributes to payload for easy access
                if received_message.message.attributes:
                    payload["_pubsub_attributes"] = dict(received_message.message.attributes)

                acknowledge_fn = ComparablePartial(
                    _acknowledge_message,
                    ack_id=received_message.ack_id,
                    subscriber_client=self._subscriber,
                    subscription_path=self._subscription_path,
                )

                extend_lease_fn = ComparablePartial(
                    _extend_message_lease,
                    ack_id=received_message.ack_id,
                    subscriber_client=self._subscriber,
                    subscription_path=self._subscription_path,
                    ack_deadline_seconds=self.pull_lease_sec,
                )

                result = ReceivedMessage[T](
                    payload=payload,
                    acknowledge_fn=acknowledge_fn,
                    extend_lease_fn=extend_lease_fn,
                    approx_receive_count=0,
                )

                results.append(result)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Error decoding message: {e}")
                continue

        return results
