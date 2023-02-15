"""
Garbage collection for execution resources.
"""

import os
import time

from boto3.exceptions import Boto3Error
from kubernetes.client.exceptions import ApiException as K8sApiException

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils.log import get_logger

from .execution_tracker import EXECUTION_DB, ExecutionInfoKeys, read_execution_clusters
from .resource_allocation.aws_sqs import get_queues
from .resource_allocation.k8s import (
    get_cluster_data,
    get_worker_sa,
    rm_workload_identity_role,
)

logger = get_logger("zetta_utils")


def _delete_execution_entry(execution_id: str) -> None:  # pragma: no cover
    client = EXECUTION_DB.backend.client  # type: ignore
    columns = [key.value for key in list(ExecutionInfoKeys)]
    parent_key = client.key("Row", execution_id)
    for column in columns:
        col_key = client.key("Column", column, parent=parent_key)
        client.delete(col_key)


def _get_stale_execution_ids() -> list[str]:  # pragma: no cover
    client = EXECUTION_DB.backend.client  # type: ignore
    query = client.query(kind="Column")

    lookback = int(os.environ["EXECUTION_HEARTBEAT_LOOKBACK"])
    time_diff = time.time() - lookback

    query = query.add_filter("heartbeat", "<", time_diff)
    query.keys_only()

    entities = list(query.fetch())
    return [entity.key.parent.id_or_name for entity in entities]


def cleanup_execution_resources(execution_id: str):  # pragma: no cover
    logger.info(f"Deleting resources from execution {execution_id}")
    clusters = read_execution_clusters(execution_id)
    for cluster in clusters:
        try:
            configuration, workload_pool = get_cluster_data(cluster)
            k8s_client.Configuration.set_default(configuration)
            logger.info(f"Deleting k8s namespace `{execution_id}`")
            k8s_core_v1_api = k8s_client.CoreV1Api()

            worker_sa = get_worker_sa(k8s_core_v1_api)
            rm_workload_identity_role(execution_id, cluster, workload_pool, principal=worker_sa)

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

    _delete_execution_entry(execution_id)


if __name__ == "__main__":  # pragma: no cover
    execution_ids = _get_stale_execution_ids()
    for exec_id in execution_ids:
        cleanup_execution_resources(exec_id)
