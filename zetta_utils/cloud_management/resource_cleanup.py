"""
Garbage collection for execution resources.
"""

import logging
import os
import time
from typing import Any, Dict, List

import taskqueue
from boto3.exceptions import Boto3Error
from google.api_core.exceptions import GoogleAPICallError
from kubernetes.client.exceptions import ApiException as K8sApiException

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils.log import get_logger
from zetta_utils.message_queues.sqs import utils as sqs_utils

from .execution_tracker import EXECUTION_DB, ExecutionInfoKeys, read_execution_clusters
from .resource_allocation.k8s import get_cluster_data
from .resource_allocation.resource_tracker import (
    EXECUTION_RESOURCE_DB,
    ExecutionResource,
    ExecutionResourceKeys,
    ExecutionResourceTypes,
)

logger = get_logger("zetta_utils")


def _delete_db_entry(client: Any, entry_id: str, columns: List[str]):
    parent_key = client.key("Row", entry_id)
    for column in columns:
        col_key = client.key("Column", column, parent=parent_key)
        client.delete(col_key)


def _delete_execution_entry(execution_id: str):  # pragma: no cover
    client = EXECUTION_DB.backend.client  # type: ignore
    columns = [key.value for key in list(ExecutionInfoKeys)]
    _delete_db_entry(client, execution_id, columns)


def _delete_resource_entry(resource_id: str):  # pragma: no cover
    client = EXECUTION_RESOURCE_DB.backend.client  # type: ignore
    columns = [key.value for key in list(ExecutionResourceKeys)]
    _delete_db_entry(client, resource_id, columns)


def _get_stale_execution_ids() -> list[str]:  # pragma: no cover
    client = EXECUTION_DB.backend.client  # type: ignore

    lookback = int(os.environ["EXECUTION_HEARTBEAT_LOOKBACK"])
    time_diff = time.time() - lookback

    filters = [("heartbeat", "<", time_diff)]
    query = client.query(kind="Column", filters=filters)
    query.keys_only()

    entities = list(query.fetch())
    return [entity.key.parent.id_or_name for entity in entities]


def _read_execution_resources(execution_id: str) -> Dict[str, ExecutionResource]:
    client = EXECUTION_RESOURCE_DB.backend.client  # type: ignore

    query = client.query(kind="Column")
    query = query.add_filter("execution_id", "=", execution_id)
    query.keys_only()

    entities = list(query.fetch())
    resouce_ids = [entity.key.parent.id_or_name for entity in entities]

    col_keys = ("type", "name")
    resources = EXECUTION_RESOURCE_DB[(resouce_ids, col_keys)]
    resources = [ExecutionResource(execution_id=execution_id, **res) for res in resources]
    return dict(zip(resouce_ids, resources))


def _delete_k8s_resources(
    execution_id: str, resources: Dict[str, ExecutionResource]
) -> bool:  # pragma: no cover
    success = True
    logger.info(f"Deleting k8s resources from execution {execution_id}")
    clusters = read_execution_clusters(execution_id)
    for cluster in clusters:
        try:
            configuration, _ = get_cluster_data(cluster)
        except GoogleAPICallError as exc:
            # cluster does not exit, discard resource entries
            if exc.code == 404:
                for resource_id in resources.keys():
                    _delete_resource_entry(resource_id)
            continue

        k8s_client.Configuration.set_default(configuration)

        k8s_apps_v1_api = k8s_client.AppsV1Api()
        k8s_core_v1_api = k8s_client.CoreV1Api()
        for resource_id, resource in resources.items():
            try:
                if resource.type == ExecutionResourceTypes.K8S_DEPLOYMENT.value:
                    logger.info(f"Deleting k8s deployment `{resource.name}`")
                    k8s_apps_v1_api.delete_namespaced_deployment(
                        name=resource.name, namespace="default"
                    )
                elif resource.type == ExecutionResourceTypes.K8S_SECRET.value:
                    logger.info(f"Deleting k8s secret `{resource.name}`")
                    k8s_core_v1_api.delete_namespaced_secret(
                        name=resource.name, namespace="default"
                    )
            except K8sApiException as exc:
                if exc.status == 404:
                    success = True
                    logger.info(f"Resource does not exist: `{resource.name}`: {exc}")
                    _delete_resource_entry(resource_id)
                else:
                    success = False
                    logger.warning(f"Failed to delete k8s resource `{resource.name}`: {exc}")
                    raise K8sApiException() from exc
    return success


def _delete_sqs_queues(resources: Dict[str, ExecutionResource]) -> bool:  # pragma: no cover
    success = True
    for resource_id, resource in resources.items():
        if resource.type != ExecutionResourceTypes.SQS_QUEUE.value:
            continue
        region_name = resource.region
        if resource.region == "" or resource.region is None:
            region_name = taskqueue.secrets.AWS_DEFAULT_REGION
        logger.info(f"Region {region_name}")
        sqs_client = sqs_utils.get_sqs_client(region_name=region_name)
        try:
            logger.info(f"Deleting SQS queue `{resource.name}`")
            queue_url = sqs_client.get_queue_url(QueueName=resource.name)["QueueUrl"]
            sqs_client.delete_queue(QueueUrl=queue_url)
        except sqs_client.exceptions.QueueDoesNotExist as exc:
            logger.info(f"Queue does not exist: `{resource.name}`: {exc}")
            _delete_resource_entry(resource_id)
        except Boto3Error as exc:
            success = False
            logger.warning(f"Failed to delete queue `{resource.name}`: {exc}")
    return success


def cleanup_execution(execution_id: str):
    success = True
    resources = _read_execution_resources(execution_id)
    success &= _delete_k8s_resources(execution_id, resources)
    success &= _delete_sqs_queues(resources)

    if success is True:
        _delete_execution_entry(execution_id)
        logger.info(f"`{exec_id}` execution cleanup complete.")
    else:
        logger.info(f"`{exec_id}` execution cleanup incomplete.")


if __name__ == "__main__":  # pragma: no cover
    execution_ids = _get_stale_execution_ids()
    logger.setLevel(logging.INFO)
    for exec_id in execution_ids:
        logger.info(f"Cleaning up execution `{exec_id}`")
        cleanup_execution(exec_id)
