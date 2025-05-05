from __future__ import annotations

import time
from typing import Any, Optional

import attrs
import boto3
import cachetools
from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_random

from zetta_utils.log import get_logger

logger = get_logger("zetta_utils")


@attrs.frozen
class SQSReceivedMsg:
    body: Any
    queue_name: str
    region_name: str
    receipt_handle: str
    approx_receive_count: int
    endpoint_url: Optional[str] = None


@cachetools.cached(cache={})
def get_sqs_client(region_name: str, endpoint_url: Optional[str] = None):
    sess = boto3.session.Session()
    return sess.client("sqs", region_name=region_name, endpoint_url=endpoint_url)


@cachetools.cached(cache={})
def get_queue_url(queue_name: str, region_name, endpoint_url: Optional[str] = None) -> str:
    sqs_client = get_sqs_client(region_name, endpoint_url=endpoint_url)
    result = sqs_client.get_queue_url(QueueName=queue_name)["QueueUrl"]
    return result


# @retry(stop=stop_after_attempt(5), wait=wait_random(min=0.5, max=2))
def receive_msgs(
    queue_name: str,
    region_name: str,
    endpoint_url: Optional[str] = None,
    max_msg_num: int = 100,
    max_time_sec: float = 2.0,
    msg_batch_size: int = 10,
    visibility_timeout: int = 60,
) -> list[SQSReceivedMsg]:
    result = []  # type: list[SQSReceivedMsg]
    start_ts = time.time()
    while True:
        sqs_client = get_sqs_client(region_name, endpoint_url=endpoint_url)
        resp = sqs_client.receive_message(
            QueueUrl=get_queue_url(queue_name, region_name, endpoint_url=endpoint_url),
            AttributeNames=["All"],
            MaxNumberOfMessages=min(msg_batch_size, max_msg_num),
            VisibilityTimeout=visibility_timeout,
            WaitTimeSeconds=1,
        )
        if "Messages" not in resp or len(resp["Messages"]) == 0:
            break

        message_batch = [
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
        result += message_batch

        if len(result) >= max_msg_num:
            break
        now_ts = time.time()
        if now_ts - start_ts >= max_time_sec:
            break

    return result


@retry(stop=stop_after_attempt(5), wait=wait_random(min=0.5, max=2))
def delete_msg_by_receipt_handle(
    receipt_handle: str,
    queue_name: str,
    region_name: str,
    endpoint_url: Optional[str] = None,
):
    logger.debug(
        f"Deleting message with handle '{receipt_handle}' from queue '{queue_name}'"
        f"in region '{region_name}'"
    )
    get_sqs_client(region_name, endpoint_url=endpoint_url).delete_message(
        QueueUrl=get_queue_url(queue_name, region_name, endpoint_url=endpoint_url),
        ReceiptHandle=receipt_handle,
    )


@retry(stop=stop_after_attempt(10), wait=wait_random(min=0.1, max=5))
def change_message_visibility(
    receipt_handle: str,
    visibility_timeout: int,
    queue_name: str,
    region_name: str,
    endpoint_url: Optional[str] = None,
):
    logger.debug(
        f"Changing visibility of the message with handle '{receipt_handle}' "
        f"from queue '{queue_name}' in region '{region_name}' to {visibility_timeout}."
    )
    get_sqs_client(region_name, endpoint_url=endpoint_url).change_message_visibility(
        QueueUrl=get_queue_url(queue_name, region_name, endpoint_url=endpoint_url),
        ReceiptHandle=receipt_handle,
        VisibilityTimeout=visibility_timeout,
    )


# To be revived if we need batch deletes:
"""
@retry(stop=stop_after_attempt(5), wait=wait_random(min=0.5, max=2))
def delete_msg_batch(
    receipt_handles: list[str],
    queue_name: str,
    region_name: str,
    endpoint_url: Optional[str] = None,
    try_count: int = 5,
) -> None:
    assert try_count > 0
    assert len(receipt_handles) <= 10, "SQS only supports batch size <= 10"
    entries_left = {str(k): v for k, v in enumerate(receipt_handles)}

    ack = None  # type: Any
    for _ in range(try_count):
        ack = get_sqs_client(region_name, endpoint_url=endpoint_url).delete_message_batch(
            QueueUrl=get_queue_url(queue_name, region_name, endpoint_url=endpoint_url),
            Entries=[{"Id": k, "ReceiptHandle": v} for k, v in entries_left.items()],
        )
        if "Successful" in ack:
            for k in ack["Successful"]:
                del entries_left[k["Id"]]

        if len(entries_left) == 0:
            return

    raise RuntimeError(f"Failed to delete messages: {ack}")  # pragma: no cover

def delete_received_msgs(msgs: list[SQSReceivedMsg]) -> None:
    receipts_by_queue = defaultdict(list)  # type: dict[tuple[str, str, Optional[str]], list[str]]
    for msg in msgs:
        receipts_by_queue[(msg.queue_name, msg.region_name, msg.endpoint_url)].append(
            msg.receipt_handle
        )

    # break into chunks of 10
    receipt_chunks_by_queue = {
        k: [v[i : i + 10] for i in range(0, len(v), 10)] for k, v in receipts_by_queue.items()
    }
    for k, v in receipt_chunks_by_queue.items():
        for chunk in v:
            delete_msg_batch(
                chunk,
                queue_name=k[0],
                region_name=k[1],
                endpoint_url=k[2],
            )


@retry(stop=stop_after_attempt(5), wait=wait_random(min=0.5, max=2))
def send_msg(
    queue_name: str,
    region_name: str,
    msg_body: str,
    endpoint_url: Optional[str] = None,
):
    sqs_client = get_sqs_client(region_name, endpoint_url=endpoint_url)
    msg_ack = sqs_client.send_message(
        QueueUrl=get_queue_url(queue_name, region_name, endpoint_url=endpoint_url),
        MessageBody=msg_body,
    )
    if (
        "ResponseMetadata" not in msg_ack or msg_ack["ResponseMetadata"]["HTTPStatusCode"] != 200
    ):  # pragma: no cover
        raise RuntimeError(
            f"Unable to send message {msg_body} to {queue_name, region_name}: {msg_ack}"
        )


"""
