from contextlib import contextmanager
from typing import Optional

import boto3

from zetta_utils import builder, log

from .resource_tracker import ExecutionResource, register_execution_resource

logger = log.get_logger("zetta_utils")


@builder.register("sqs_queue_ctx_mngr")
@contextmanager
def sqs_queue_ctx_mngr(execution_id: str, name: str):
    sqs = boto3.resource("sqs")
    queue = sqs.create_queue(QueueName=name, Attributes={"SqsManagedSseEnabled": "false"})
    register_execution_resource(ExecutionResource(execution_id, "sqs_queue", name))

    logger.info(f"Created SQS queue with URL={queue.url}")
    try:
        yield
    finally:
        logger.info(f"Deleting SQS queue '{name}'")
        logger.debug(f"Deleting SQS queue with URL={queue.url}")
        queue.delete()


def get_queues(prefix: Optional[str] = None):
    """
    Gets a list of SQS queues. When a prefix is specified, only queues with names
    that start with the prefix are returned.

    :param prefix: The prefix used to restrict the list of returned queues.
    :return: A list of Queue objects.
    """
    sqs = boto3.resource("sqs")
    if prefix:
        queue_iter = sqs.queues.filter(QueueNamePrefix=prefix)
    else:
        queue_iter = sqs.queues.all()

    queues = list(queue_iter)
    logger.info(f"Found {len(queues)} queues.")
    return queues
