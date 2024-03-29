"""
Garbage collection for run resources.
"""

import json
import logging
import os
import time
from typing import Mapping

import taskqueue
from boto3.exceptions import Boto3Error
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.datastore import query
from kubernetes.client.exceptions import ApiException as K8sApiException

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils.cloud_management.resource_allocation.k8s import (
    ClusterInfo,
    get_cluster_data,
)
from zetta_utils.log import get_logger
from zetta_utils.message_queues.sqs import utils as sqs_utils
from zetta_utils.run import (
    RESOURCE_DB,
    RUN_DB,
    Resource,
    ResourceTypes,
    RunInfo,
    RunState,
    deregister_resource,
    update_run_info,
)

logger = get_logger("zetta_utils")


def _get_stale_run_ids() -> list[str]:  # pragma: no cover
    lookback = int(os.environ["EXECUTION_HEARTBEAT_LOOKBACK"])
    time_diff = time.time() - lookback

    resourcedb_client = RESOURCE_DB.backend.client  # type: ignore
    _query = resourcedb_client.query(kind="Column")
    _query.projection = ["run_id"]
    run_ids = list(set(f"run-{x['run_id']}" for x in _query.fetch()))
    heartbeats = [x["heartbeat"] for x in RUN_DB[(run_ids, ("heartbeat",))]]

    result = []
    for run_id, heartbeat in zip(run_ids, heartbeats):
        if heartbeat < time_diff:
            result.append(run_id)
    return result


def _read_clusters(run_id_key: str) -> list[ClusterInfo]:  # pragma: no cover
    col_keys = ("clusters",)
    try:
        clusters_str = RUN_DB[(run_id_key, col_keys)][col_keys[0]]
    except KeyError:
        return []
    clusters: list[Mapping] = json.loads(clusters_str)
    return [ClusterInfo(**cluster) for cluster in clusters]


def _read_run_resources(run_id: str) -> dict[str, Resource]:
    client = RESOURCE_DB.backend.client  # type: ignore

    _query = client.query(kind="Column")
    _query.add_filter(filter=query.PropertyFilter("run_id", "=", run_id))
    _query.keys_only()

    entities = list(_query.fetch())
    resource_ids = [entity.key.parent.id_or_name for entity in entities]

    col_keys = ("type", "name")
    resources = RESOURCE_DB[(resource_ids, col_keys)]
    resources = [Resource(run_id=run_id, **res) for res in resources]
    return dict(zip(resource_ids, resources))


def _delete_k8s_resources(run_id: str, resources: dict[str, Resource]) -> bool:  # pragma: no cover
    success = True
    logger.info(f"Deleting k8s resources from run {run_id}")
    clusters = _read_clusters(f"run-{run_id}")
    for cluster in clusters:
        try:
            configuration, _ = get_cluster_data(cluster)
        except GoogleAPICallError as exc:
            # cluster does not exist, discard resource entries
            logger.info(f"Could not connect to {cluster}: ERROR CODE {exc.code}")
            if exc.code == 404:
                for resource_id in resources.keys():
                    deregister_resource(resource_id)
            continue

        k8s_client.Configuration.set_default(configuration)

        k8s_apps_v1_api = k8s_client.AppsV1Api()
        k8s_core_v1_api = k8s_client.CoreV1Api()
        for resource_id, resource in resources.items():
            try:
                if resource.type == ResourceTypes.K8S_DEPLOYMENT.value:
                    logger.info(f"Deleting k8s deployment `{resource.name}`")
                    k8s_apps_v1_api.delete_namespaced_deployment(
                        name=resource.name, namespace="default"
                    )
                elif resource.type == ResourceTypes.K8S_SECRET.value:
                    logger.info(f"Deleting k8s secret `{resource.name}`")
                    k8s_core_v1_api.delete_namespaced_secret(
                        name=resource.name, namespace="default"
                    )
            except K8sApiException as exc:
                if exc.status == 404:
                    success = True
                    logger.info(f"Resource does not exist: `{resource.name}`: {exc}")
                    deregister_resource(resource_id)
                else:
                    success = False
                    logger.warning(f"Failed to delete k8s resource `{resource.name}`: {exc}")
                    raise K8sApiException() from exc
    return success


def _delete_sqs_queues(resources: dict[str, Resource]) -> bool:  # pragma: no cover
    success = True
    for resource_id, resource in resources.items():
        if resource.type != ResourceTypes.SQS_QUEUE.value:
            continue
        region_name = resource.region
        if resource.region == "" or resource.region is None:
            region_name = taskqueue.secrets.AWS_DEFAULT_REGION
        sqs_client = sqs_utils.get_sqs_client(region_name=region_name)
        try:
            logger.info(f"Deleting SQS queue `{resource.name}`")
            queue_url = sqs_client.get_queue_url(QueueName=resource.name)["QueueUrl"]
            sqs_client.delete_queue(QueueUrl=queue_url)
        except sqs_client.exceptions.QueueDoesNotExist as exc:
            logger.info(f"Queue does not exist: `{resource.name}`: {exc}")
            deregister_resource(resource_id)
        except Boto3Error as exc:
            success = False
            logger.warning(f"Failed to delete queue `{resource.name}`: {exc}")
    return success


def cleanup_run(run_id: str):
    success = True
    resources = _read_run_resources(run_id)
    success &= _delete_k8s_resources(run_id, resources)
    success &= _delete_sqs_queues(resources)

    if success is True:
        logger.info(f"`{run_id}` run cleanup complete.")
        update_run_info(run_id, {RunInfo.STATE.value: RunState.TIMEDOUT.value})
    else:
        logger.info(f"`{run_id}` run cleanup failed.")


if __name__ == "__main__":  # pragma: no cover
    logger.setLevel(logging.INFO)
    for _id in _get_stale_run_ids():
        _id = _id[4:]
        logger.info(f"Cleaning up run `{_id}`")
        cleanup_run(_id)
