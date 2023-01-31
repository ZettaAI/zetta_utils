from contextlib import contextmanager

import boto3

from zetta_utils import builder, log

from .resource_tracker import ExecutionResource, register_execution_resource

logger = log.get_logger("zetta_utils")


@builder.register("sqs_queue_ctx_mngr")
@contextmanager
def sqs_queue_ctx_mngr(execution_id: str, name: str):
    sqs = boto3.resource("sqs")
    queue = sqs.create_queue(QueueName=name, Attributes={"SqsManagedSseEnabled": "false"})

    register_execution_resource(ExecutionResource(execution_id, "aws_sqs", name))

    logger.info(f"Created SQS queue with URL={queue.url}")
    try:
        yield
    finally:
        logger.info(f"Deleting SQS queue '{name}'")
        logger.debug(f"Deleting SQS queue with URL={queue.url}")
        queue.delete()
