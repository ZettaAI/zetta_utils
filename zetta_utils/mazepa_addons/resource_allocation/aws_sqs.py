import uuid
from contextlib import contextmanager

import attrs
import boto3

from zetta_utils import builder, log

from .tracker import ExecutionResource, get_execution_resource_db

logger = log.get_logger("zetta_utils")


@builder.register("sqs_queue_ctx_mngr")
@contextmanager
def sqs_queue_ctx_mngr(execution_id: str, name: str):
    sqs = boto3.resource("sqs")
    queue = sqs.create_queue(QueueName=name, Attributes={"SqsManagedSseEnabled": "false"})

    resource_uuid = str(uuid.uuid4())
    execution_db = get_execution_resource_db()
    execution_resource = ExecutionResource(execution_id, "aws_sqs", name)
    execution_db[resource_uuid] = attrs.asdict(execution_resource)  # type: ignore

    logger.info(f"Created SQS queue with URL={queue.url}")
    try:
        yield
    finally:
        logger.info(f"Deleting SQS queue '{name}'")
        logger.debug(f"Deleting SQS queue with URL={queue.url}")
        queue.delete()
