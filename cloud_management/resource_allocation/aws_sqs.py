from contextlib import contextmanager

from zetta_utils import builder, log
from zetta_utils.message_queues.sqs import SQSQueue
from zetta_utils.message_queues.sqs import utils as sqs_utils
from zetta_utils.run import (
    Resource,
    ResourceTypes,
    deregister_resource,
    register_resource,
)

logger = log.get_logger("zetta_utils")


@builder.register("sqs_queue_ctx_mngr")
@contextmanager
def sqs_queue_ctx_mngr(run_id: str, queue: SQSQueue):
    sqs = sqs_utils.get_sqs_client(queue.region_name)
    _queue = sqs.create_queue(QueueName=queue.name, Attributes={"SqsManagedSseEnabled": "false"})

    _id = register_resource(
        Resource(run_id, ResourceTypes.SQS_QUEUE.value, queue.name, region=queue.region_name)
    )

    logger.info(f"Created SQS queue with URL={_queue['QueueUrl']}")
    try:
        yield
    finally:
        logger.info(f"Deleting SQS queue '{queue.name}'")
        logger.debug(f"Deleting SQS queue with URL={_queue['QueueUrl']}")
        sqs.delete_queue(QueueUrl=_queue["QueueUrl"])
        deregister_resource(_id)
