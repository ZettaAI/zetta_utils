from __future__ import annotations

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import attrs
import boto3
import cachetools
from botocore.config import Config
from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_random

from zetta_utils.log import get_logger

logger = get_logger("zetta_utils")

RECEIVE_PARALLELISM = 8
RECEIVE_BATCH_SIZE = 10
RECEIVE_WAIT_TIME_SEC = 20
DELETE_BATCH_SIZE = 10
SQS_MAX_POOL_CONNECTIONS = 16

_CLIENT_CACHE_LOCK = threading.Lock()


@attrs.frozen
class SQSReceivedMsg:
    body: Any
    queue_name: str
    region_name: str
    receipt_handle: str
    approx_receive_count: int
    endpoint_url: Optional[str] = None


@cachetools.cached(cache={}, lock=_CLIENT_CACHE_LOCK)
def get_sqs_client(region_name: str, endpoint_url: Optional[str] = None):
    sess = boto3.session.Session()
    config = Config(max_pool_connections=SQS_MAX_POOL_CONNECTIONS)
    return sess.client("sqs", region_name=region_name, endpoint_url=endpoint_url, config=config)


@cachetools.cached(cache={}, lock=_CLIENT_CACHE_LOCK)
def get_queue_url(queue_name: str, region_name, endpoint_url: Optional[str] = None) -> str:
    sqs_client = get_sqs_client(region_name, endpoint_url=endpoint_url)
    result = sqs_client.get_queue_url(QueueName=queue_name)["QueueUrl"]
    return result


def _receive_one_batch(
    queue_name: str,
    region_name: str,
    endpoint_url: Optional[str],
    queue_url: str,
    batch_size: int,
    visibility_timeout: int,
    wait_time_sec: int,
) -> list[SQSReceivedMsg]:
    sqs_client = get_sqs_client(region_name, endpoint_url=endpoint_url)
    resp = sqs_client.receive_message(
        QueueUrl=queue_url,
        AttributeNames=["All"],
        MaxNumberOfMessages=batch_size,
        VisibilityTimeout=visibility_timeout,
        WaitTimeSeconds=wait_time_sec,
    )
    if "Messages" not in resp or len(resp["Messages"]) == 0:
        return []
    return [
        SQSReceivedMsg(
            body=message["Body"],
            receipt_handle=message["ReceiptHandle"],
            queue_name=queue_name,
            region_name=region_name,
            approx_receive_count=int(message["Attributes"]["ApproximateReceiveCount"]),
            endpoint_url=endpoint_url,
        )
        for message in resp["Messages"]
    ]


@retry(stop=stop_after_attempt(5), wait=wait_random(min=0.5, max=2))
def receive_msgs(
    queue_name: str,
    region_name: str,
    endpoint_url: Optional[str] = None,
    max_msg_num: int = 100,
    max_time_sec: float = 2.0,
    msg_batch_size: int = RECEIVE_BATCH_SIZE,
    visibility_timeout: int = 60,
    parallelism: int = RECEIVE_PARALLELISM,
    wait_time_sec: int = RECEIVE_WAIT_TIME_SEC,
) -> list[SQSReceivedMsg]:
    """
    Receive messages from an SQS queue using a thread pool of parallel
    ``ReceiveMessage`` calls.

    :param queue_name: Name of the SQS queue.
    :param region_name: AWS region of the queue.
    :param endpoint_url: Optional override endpoint (e.g. for ElasticMQ).
    :param max_msg_num: Upper bound on total messages returned.
    :param max_time_sec: Upper bound on wall-clock time spent receiving.
    :param msg_batch_size: ``MaxNumberOfMessages`` per single request (<=10).
    :param visibility_timeout: Visibility timeout, in seconds, applied to
        every received message.
    :param parallelism: Number of concurrent ``ReceiveMessage`` calls.
    :param wait_time_sec: Long-poll wait time, in seconds, per request.
    :return: Received messages, possibly fewer than ``max_msg_num``.
    """
    logger.debug(
        f"RECEIVE: Attempting to receive messages from queue '{queue_name}' "
        f"with visibility_timeout={visibility_timeout}s, parallelism={parallelism}"
    )
    queue_url = get_queue_url(queue_name, region_name, endpoint_url=endpoint_url)
    # Pre-warm client cache so worker threads do not race on the lock.
    get_sqs_client(region_name, endpoint_url=endpoint_url)

    result: list[SQSReceivedMsg] = []
    start_ts = time.time()

    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        while True:
            remaining = max_msg_num - len(result)
            if remaining <= 0:
                break

            num_workers = min(
                parallelism,
                max(1, (remaining + msg_batch_size - 1) // msg_batch_size),
            )
            futures = [
                pool.submit(
                    _receive_one_batch,
                    queue_name,
                    region_name,
                    endpoint_url,
                    queue_url,
                    min(msg_batch_size, max(1, remaining)),
                    visibility_timeout,
                    wait_time_sec,
                )
                for _ in range(num_workers)
            ]

            any_received = False
            for fut in as_completed(futures):
                batch = fut.result()
                if batch:
                    any_received = True
                    result.extend(batch)
                    if len(result) >= max_msg_num:
                        break

            if not any_received:
                break
            if len(result) >= max_msg_num:
                break
            if time.time() - start_ts >= max_time_sec:
                break

    if len(result) > max_msg_num:
        result = result[:max_msg_num]

    logger.debug(f"RECEIVE: Got {len(result)} messages from queue '{queue_name}'")
    return result


@retry(stop=stop_after_attempt(5), wait=wait_random(min=0.5, max=2))
def delete_msg_by_receipt_handle(
    receipt_handle: str,
    queue_name: str,
    region_name: str,
    endpoint_url: Optional[str] = None,
):
    queue_url = get_queue_url(queue_name, region_name, endpoint_url=endpoint_url)
    logger.debug(
        f"DELETE: Deleting message from queue '{queue_name}' (URL: {queue_url}), "
        f"handle: {receipt_handle[:30]}..."
    )
    try:
        get_sqs_client(region_name, endpoint_url=endpoint_url).delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle,
        )
        logger.debug(f"DELETE: Successfully deleted message from queue '{queue_name}'")
    except Exception as e:  # pragma: no cover
        logger.error(
            f"DELETE: Failed to delete message from queue '{queue_name}': {type(e).__name__}: {e}"
        )
        raise


@retry(stop=stop_after_attempt(10), wait=wait_random(min=0.1, max=5))
def change_message_visibility(
    receipt_handle: str,
    visibility_timeout: int,
    queue_name: str,
    region_name: str,
    endpoint_url: Optional[str] = None,
):
    queue_url = get_queue_url(queue_name, region_name, endpoint_url=endpoint_url)
    logger.debug(
        f"VISIBILITY: Changing visibility to {visibility_timeout}s for queue '{queue_name}' "
        f"(URL: {queue_url}), handle: {receipt_handle[:30]}..."
    )
    try:
        get_sqs_client(region_name, endpoint_url=endpoint_url).change_message_visibility(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=visibility_timeout,
        )
        logger.debug(
            f"VISIBILITY: Successfully changed visibility to {visibility_timeout}s "
            f"for queue '{queue_name}'"
        )
    except Exception as e:  # pragma: no cover
        logger.error(
            f"VISIBILITY: Failed to change visibility for queue '{queue_name}' "
            f"to {visibility_timeout}s: {type(e).__name__}: {e}"
        )
        raise


@retry(stop=stop_after_attempt(5), wait=wait_random(min=0.5, max=2))
def delete_msg_batch(
    receipt_handles: list[str],
    queue_name: str,
    region_name: str,
    endpoint_url: Optional[str] = None,
    try_count: int = 5,
) -> None:
    """
    Delete up to ``DELETE_BATCH_SIZE`` messages from an SQS queue in a single
    ``DeleteMessageBatch`` API call. Retries any entries the service reports
    as failed up to ``try_count`` times.

    :param receipt_handles: Receipt handles to delete; length must be
        ``<= DELETE_BATCH_SIZE``.
    :param queue_name: Name of the SQS queue.
    :param region_name: AWS region of the queue.
    :param endpoint_url: Optional override endpoint.
    :param try_count: Number of attempts for the batch as a whole.
    """
    if len(receipt_handles) == 0:
        return
    assert try_count > 0
    assert (
        len(receipt_handles) <= DELETE_BATCH_SIZE
    ), f"SQS only supports batch size <= {DELETE_BATCH_SIZE}"
    entries_left = {str(k): v for k, v in enumerate(receipt_handles)}

    queue_url = get_queue_url(queue_name, region_name, endpoint_url=endpoint_url)
    sqs_client = get_sqs_client(region_name, endpoint_url=endpoint_url)

    ack: Any = None
    for _ in range(try_count):
        ack = sqs_client.delete_message_batch(
            QueueUrl=queue_url,
            Entries=[{"Id": k, "ReceiptHandle": v} for k, v in entries_left.items()],
        )
        if "Successful" in ack:
            for entry in ack["Successful"]:
                entries_left.pop(entry["Id"], None)

        if len(entries_left) == 0:
            return

    raise RuntimeError(f"Failed to delete messages: {ack}")  # pragma: no cover


def delete_received_msgs(msgs: list[SQSReceivedMsg]) -> None:
    """
    Delete a heterogeneous collection of received messages, batching by
    ``(queue_name, region_name, endpoint_url)`` and chunking each group into
    ``DELETE_BATCH_SIZE``-sized batches.
    """
    if len(msgs) == 0:
        return
    receipts_by_queue: dict[tuple[str, str, Optional[str]], list[str]] = defaultdict(list)
    for msg in msgs:
        receipts_by_queue[(msg.queue_name, msg.region_name, msg.endpoint_url)].append(
            msg.receipt_handle
        )

    for (queue_name, region_name, endpoint_url), receipts in receipts_by_queue.items():
        for i in range(0, len(receipts), DELETE_BATCH_SIZE):
            chunk = receipts[i : i + DELETE_BATCH_SIZE]
            delete_msg_batch(
                chunk,
                queue_name=queue_name,
                region_name=region_name,
                endpoint_url=endpoint_url,
            )
