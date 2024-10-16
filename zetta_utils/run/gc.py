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

from kubernetes import client as k8s_client
from zetta_utils.cloud_management.resource_allocation import k8s
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
from zetta_utils.run.gc_slack import post_message

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
            if run_id[:4] == "run-":
                stale_ids.append(run_id[4:])
            else:
                stale_ids.append(run_id)
    return run_resources, stale_ids


def _read_clusters(run_id_key: str) -> list[k8s.ClusterInfo]:  # pragma: no cover
    try:
        clusters_str = RUN_DB[run_id_key]["clusters"]
    except KeyError:
        return [DEFAULT_GCP_CLUSTER]
    clusters: list[Mapping] = json.loads(clusters_str)
    return [k8s.ClusterInfo(**cluster) for cluster in clusters]


def _delete_dynamic_resource(
    resource_id: str, resource: Resource, configuration: k8s_client.Configuration
) -> bool:
    success = True
    try:
        k8s.delete_dynamic_resource(
            resource.name,
            configuration,
            k8s.keda.KEDA_API_VERSION,
            resource.type,
            namespace="default",
        )
        deregister_resource(resource_id)
    except k8s_client.ApiException as exc:
        if exc.status == 404:
            success = True
            logger.info(f"Resource does not exist: `{resource.name}`: {exc}")
            deregister_resource(resource_id)
        else:
            success = False
            msg = f"Failed to delete k8s resource `{resource.name}`: {exc}"
            logger.warning(msg)
            post_message(msg)
    return success


def _delete_k8s_resource(resource_id: str, resource: Resource) -> bool:
    success = True
    k8s_apps_v1_api = k8s_client.AppsV1Api()
    k8s_core_v1_api = k8s_client.CoreV1Api()
    k8s_batch_v1_api = k8s_client.BatchV1Api()
    try:
        if resource.type == ResourceTypes.K8S_DEPLOYMENT.value:
            logger.info(f"Deleting k8s deployment `{resource.name}`")
            k8s_apps_v1_api.delete_namespaced_deployment(name=resource.name, namespace="default")
        elif resource.type == ResourceTypes.K8S_SECRET.value:
            logger.info(f"Deleting k8s secret `{resource.name}`")
            k8s_core_v1_api.delete_namespaced_secret(name=resource.name, namespace="default")
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
        deregister_resource(resource_id)
    except k8s_client.ApiException as exc:
        if exc.status == 404:
            success = True
            logger.info(f"Resource does not exist: `{resource.name}`: {exc}")
            deregister_resource(resource_id)
        else:
            success = False
            msg = f"Failed to delete k8s resource `{resource.name}`: {exc}"
            logger.warning(msg)
            post_message(msg)
    return success


def _delete_k8s_resources(run_id: str, resources: dict[str, Resource]) -> bool:  # pragma: no cover
    success = True
    logger.info(f"Deleting k8s resources from run {run_id}")
    clusters = _read_clusters(run_id)
    for cluster in clusters:
        try:
            configuration, _ = k8s.get_cluster_data(cluster)
        except GoogleAPICallError as exc:
            # cluster does not exist, discard resource entries
            logger.info(f"Could not connect to {cluster}: ERROR CODE {exc.code}")
            if exc.code == 404:
                for resource_id in resources.keys():
                    deregister_resource(resource_id)
            else:
                msg = f"Error connecting to cluster {run_id}:{cluster}:{exc.code}:{exc.message}."
                post_message(msg)
            continue

        k8s_client.Configuration.set_default(configuration)
        for resource_id, resource in resources.items():
            if resource.type in [
                ResourceTypes.K8S_SCALED_JOB.value,
                ResourceTypes.K8S_SCALED_OBJECT.value,
                ResourceTypes.K8S_TRIGGER_AUTH.value,
            ]:
                success &= _delete_dynamic_resource(resource_id, resource, configuration)
                continue
            success &= _delete_k8s_resource(resource_id, resource)
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
            deregister_resource(resource_id)
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
    if len(stale_run_ids) > 0:
        _ids = "\n".join(stale_run_ids)
        post_message(f"Cleaning up {len(stale_run_ids)} runs.\n```{_ids}```")
    else:
        post_message("Nothing to do.", priority=False)
    for _id in stale_run_ids:
        logger.info(f"Cleaning up run `{_id}`")
        cleanup_run(_id, _resources[_id])
