"""
Garbage collection for run resources.
"""

import json
import logging
import os
import time
from collections import defaultdict
from typing import Mapping

import taskqueue
from boto3.exceptions import Boto3Error
from google.api_core.exceptions import GoogleAPICallError
from kubernetes.client.exceptions import ApiException as K8sApiException

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils.cloud_management.resource_allocation.k8s import (
    ClusterInfo,
    get_cluster_data,
)
from zetta_utils.log import get_logger
from zetta_utils.mazepa_addons.configurations.execute_on_gcp_with_sqs import (
    DEFAULT_GCP_CLUSTER,
)
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


def _get_current_resources_and_stale_run_ids() -> (
    tuple[dict[str, dict], list[str]]
):  # pragma: no cover
    run_resources: dict[str, dict] = defaultdict(dict)
    _resources = RESOURCE_DB.query()
    for _resource_id, _resource in _resources.items():
        run_resources[str(_resource["run_id"])][_resource_id] = _resource

    run_ids = list(run_resources.keys())
    stale_ids = []
    heartbeats = [x.get("heartbeat", 0) for x in RUN_DB[(run_ids, ("heartbeat",))]]
    lookback = int(os.environ["EXECUTION_HEARTBEAT_LOOKBACK"])
    time_diff = time.time() - lookback
    for run_id, heartbeat in zip(run_ids, heartbeats):
        if heartbeat < time_diff:
            stale_ids.append(run_id[4:])
    return run_resources, stale_ids


def _read_clusters(run_id_key: str) -> list[ClusterInfo]:  # pragma: no cover
    try:
        clusters_str = RUN_DB[run_id_key]["clusters"]
    except KeyError:
        return [DEFAULT_GCP_CLUSTER]
    clusters: list[Mapping] = json.loads(clusters_str)
    return [ClusterInfo(**cluster) for cluster in clusters]


def _delete_k8s_resources(run_id: str, resources: dict[str, Resource]) -> bool:  # pragma: no cover
    success = True
    logger.info(f"Deleting k8s resources from run {run_id}")
    clusters = _read_clusters(run_id)
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
        k8s_batch_v1_api = k8s_client.BatchV1Api()
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
                elif resource.type == ResourceTypes.K8S_CONFIGMAP.value:
                    logger.info(f"Deleting k8s configmap `{resource.name}`")
                    k8s_core_v1_api.delete_namespaced_config_map(
                        name=resource.name,
                        namespace="default",
                    )
                elif resource.type == ResourceTypes.K8S_JOB.value:
                    logger.info(f"Deleting k8s job `{resource.name}`")
                    k8s_batch_v1_api.delete_namespaced_job(
                        name=resource.name,
                        namespace="default",
                        propagation_policy="Foreground",
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


def cleanup_run(run_id: str, resources_raw: dict):
    success = True
    resource_ids = resources_raw.keys()
    _resources = [Resource(**resource) for resource in resources_raw.values()]
    resources = dict(zip(resource_ids, _resources))

    success &= _delete_k8s_resources(run_id, resources)
    success &= _delete_sqs_queues(resources)
    if success is True:
        logger.info(f"`{run_id}` run cleanup complete.")
        update_run_info(run_id, {RunInfo.STATE.value: RunState.TIMEDOUT.value})
    else:
        logger.info(f"`{run_id}` run cleanup failed.")


if __name__ == "__main__":  # pragma: no cover
    logger.setLevel(logging.INFO)
    _resources, stale_run_ids = _get_current_resources_and_stale_run_ids()
    for _id in stale_run_ids:
        logger.info(f"Cleaning up run `{_id}`")
        cleanup_run(_id, _resources[_id])
