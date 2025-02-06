"""
Helpers to create keda.sh ScaledObjects.
    https://keda.sh/docs/latest/reference/scaledobject-spec/
"""

from contextlib import contextmanager

from kubernetes.client import ApiClient

from kubernetes import client as k8s_client
from zetta_utils import log
from zetta_utils.cloud_management.resource_allocation.k8s.secret import secrets_ctx_mngr
from zetta_utils.message_queues.sqs import utils as sqs_utils
from zetta_utils.message_queues.sqs.queue import SQSQueue
from zetta_utils.run import (
    Resource,
    ResourceTypes,
    deregister_resource,
    register_resource,
)

from .common import (
    ClusterInfo,
    create_dynamic_resource,
    delete_dynamic_resource,
    get_cluster_data,
)
from .deployment import deployment_ctx_mngr

KEDA_API_VERSION = "keda.sh/v1alpha1"
logger = log.get_logger("zetta_utils")


def _get_sqs_trigger(sqs_trigger_name: str, queue: SQSQueue) -> dict:
    sqs_client = sqs_utils.get_sqs_client(region_name=queue.region_name)
    sqs_trigger = {
        "type": "aws-sqs-queue",
        "authenticationRef": {"name": sqs_trigger_name},
        "metadata": {
            "queueURL": sqs_client.get_queue_url(QueueName=queue.name)["QueueUrl"],
            "queueLength": "1",
            "awsRegion": queue.region_name,
            "scaleOnInFlight": "false",
        },
    }
    return sqs_trigger


def _get_sqs_trigger_auth_manifest(run_id: str, name: str) -> dict:
    manifest = {
        "apiVersion": KEDA_API_VERSION,
        "kind": ResourceTypes.K8S_TRIGGER_AUTH.value,
        "metadata": {"name": name},
        "spec": {
            "secretTargetRef": [
                {
                    "parameter": "awsAccessKeyID",
                    "name": f"run-{run_id}-secret-combined",
                    "key": "AWS_ACCESS_KEY_ID",
                },
                {
                    "parameter": "awsSecretAccessKey",
                    "name": f"run-{run_id}-secret-combined",
                    "key": "AWS_SECRET_ACCESS_KEY",
                },
            ]
        },
    }
    return manifest


def _get_scaled_object_manifest(
    run_id: str,
    triggers: list[dict],
    target_name: str,
    target_kind: str | None = "Deployment",
    min_replicas: int = 0,
    max_replicas: int = 10,
    polling_interval: int = 30,
    cool_down_period: int = 300,
) -> dict:
    """
    Create manifest for Keda ScaledObject.
    """
    name = f"run-{run_id}-scaledobject"
    manifest = {
        "apiVersion": KEDA_API_VERSION,
        "kind": ResourceTypes.K8S_SCALED_OBJECT.value,
        "metadata": {"name": name, "annotations": {}},
        "spec": {
            "scaleTargetRef": {
                "kind": target_kind,
                "name": target_name,
            },
            "pollingInterval": polling_interval,
            "cooldownPeriod": cool_down_period,
            "initialCooldownPeriod": 0,
            "minReplicaCount": min_replicas,
            "maxReplicaCount": max_replicas,
            "triggers": triggers,
        },
    }
    return manifest


def _get_scaled_job_manifest(
    run_id: str,
    triggers: list[dict],
    job_spec: k8s_client.V1JobSpec,
    min_replicas: int = 0,
    max_replicas: int = 10,
    polling_interval: int = 30,
    scaling_strategy: str = "eager",
) -> dict:
    """
    Create manifest for Keda ScaledObject.
    """
    api = ApiClient()
    name = f"run-{run_id}-scaledjob"
    manifest = {
        "apiVersion": KEDA_API_VERSION,
        "kind": ResourceTypes.K8S_SCALED_JOB.value,
        "metadata": {"name": name, "annotations": {}},
        "spec": {
            "jobTargetRef": api.sanitize_for_serialization(job_spec),
            "pollingInterval": polling_interval,
            "minReplicaCount": min_replicas,
            "maxReplicaCount": max_replicas,
            "scalingStrategy": {"strategy": scaling_strategy},
            "successfulJobsHistoryLimit": 0,
            "failedJobsHistoryLimit": 10,
            "triggers": triggers,
        },
    }
    return manifest


@contextmanager
def sqs_trigger_ctx_mngr(
    run_id: str,
    cluster_info: ClusterInfo,
    sqs_trigger_name: str,
    namespace: str | None = "default",
):
    configuration, _ = get_cluster_data(cluster_info)
    manifest = _get_sqs_trigger_auth_manifest(run_id, sqs_trigger_name)
    resource = Resource(run_id, ResourceTypes.K8S_TRIGGER_AUTH.value, sqs_trigger_name)
    create_dynamic_resource(
        resource.name, configuration, KEDA_API_VERSION, resource.type, manifest, namespace
    )
    _id = register_resource(resource)

    try:
        yield
    finally:
        # new configuration to refresh expired tokens (long running executions)
        configuration, _ = get_cluster_data(cluster_info)
        delete_dynamic_resource(
            resource.name, configuration, KEDA_API_VERSION, resource.type, namespace
        )
        deregister_resource(_id)


@contextmanager
def scaled_deployment_ctx_mngr(
    run_id: str,
    cluster_info: ClusterInfo,
    deployment: k8s_client.V1Deployment,
    secrets: list[k8s_client.V1Secret],
    max_replicas: int,
    sqs_trigger_name: str,
    queue: SQSQueue,
    namespace: str | None = "default",
    cool_down_period: int = 300,
):
    configuration, _ = get_cluster_data(cluster_info)
    with deployment_ctx_mngr(run_id, cluster_info, deployment, secrets, namespace=namespace):
        manifest = _get_scaled_object_manifest(
            run_id,
            [_get_sqs_trigger(sqs_trigger_name, queue)],
            target_name=deployment.metadata.name,
            max_replicas=max_replicas,
            cool_down_period=cool_down_period,
        )
        so_name = manifest["metadata"]["name"]
        resource = Resource(run_id, ResourceTypes.K8S_SCALED_OBJECT.value, so_name)
        create_dynamic_resource(
            resource.name, configuration, KEDA_API_VERSION, resource.type, manifest, namespace
        )
        _id = register_resource(resource)

        try:
            yield
        finally:
            # new configuration to refresh expired tokens (long running executions)
            configuration, _ = get_cluster_data(cluster_info)
            delete_dynamic_resource(
                resource.name, configuration, KEDA_API_VERSION, resource.type, namespace
            )
            deregister_resource(_id)


@contextmanager
def scaled_job_ctx_mngr(
    run_id: str,
    group_name: str,
    cluster_info: ClusterInfo,
    job_spec: k8s_client.V1JobSpec,
    secrets: list[k8s_client.V1Secret],
    replicas: int,
    sqs_trigger_name: str,
    queue: SQSQueue,
    max_replicas: int = 0,
    namespace: str | None = "default",
):
    if max_replicas == 0:
        max_replicas = replicas
        replicas = 0
    configuration, _ = get_cluster_data(cluster_info)
    with secrets_ctx_mngr(run_id, secrets, cluster_info, namespace=namespace):
        manifest = _get_scaled_job_manifest(
            f"{run_id}-{group_name}",
            [_get_sqs_trigger(sqs_trigger_name, queue)],
            job_spec=job_spec,
            min_replicas=replicas,
            max_replicas=max_replicas,
        )
        so_name = manifest["metadata"]["name"]
        resource = Resource(run_id, ResourceTypes.K8S_SCALED_JOB.value, so_name)
        create_dynamic_resource(
            resource.name, configuration, KEDA_API_VERSION, resource.type, manifest, namespace
        )
        _id = register_resource(resource)

        try:
            yield
        finally:
            # new configuration to refresh expired tokens (long running executions)
            configuration, _ = get_cluster_data(cluster_info)
            delete_dynamic_resource(
                resource.name, configuration, KEDA_API_VERSION, resource.type, namespace
            )
            deregister_resource(_id)
