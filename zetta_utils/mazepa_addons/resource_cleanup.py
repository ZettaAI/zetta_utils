"""
Garbage collection for execution resources.
"""

import os
import time

from boto3.exceptions import Boto3Error
from kubernetes.client.exceptions import ApiException as K8sApiException

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils.log import get_logger

from .execution_tracker import EXECUTION_DB, read_execution_clusters
from .resource_allocation.aws_sqs import get_queues
from .resource_allocation.k8s import get_cluster_configuration

logger = get_logger("zetta_utils")


def _get_stale_execution_ids() -> list[str]:
    client = EXECUTION_DB.backend.client  # type: ignore
    query = client.query(kind="Column")

    lookback = int(os.environ["EXEC_HEARTBEAT_LOOKBACK"])
    time_diff = time.time() - lookback

    query = query.add_filter("heartbeat", "<", time_diff)
    query.keys_only()

    entities = list(query.fetch())
    return [entity.key.parent.id_or_name for entity in entities]


def _cleanup_execution_resources(execution_id: str):
    logger.info(f"Deleting resources from execution {execution_id}")
    clusters = read_execution_clusters(execution_id)
    for cluster in clusters:
        try:
            k8s_client.Configuration.set_default(get_cluster_configuration(cluster))
            logger.info(f"Deleting k8s namespace `{execution_id}`")
            k8s_core_v1_api = k8s_client.CoreV1Api()
            k8s_core_v1_api.delete_namespace(name=execution_id)
        except K8sApiException as exc:
            logger.info(f"Failed to delete k8s namespace `{execution_id}`: {exc}")

    prefix = f"zzz-{execution_id}"
    queues = get_queues(prefix=prefix)
    for queue in queues:
        try:
            logger.info(f"Deleting SQS queue '{queue.url}'")
            queue.delete()
        except Boto3Error as exc:
            logger.info(f"Failed to delete SQS queue: {exc}")


if __name__ == "__main__":
    execution_ids = _get_stale_execution_ids()
    for exec_id in execution_ids:
        _cleanup_execution_resources(exec_id)
