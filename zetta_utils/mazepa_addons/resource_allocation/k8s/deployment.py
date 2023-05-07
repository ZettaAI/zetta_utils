"""
Helpers for k8s deployments.
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import builder, log, mazepa

from ..resource_tracker import (
    ExecutionResource,
    ExecutionResourceTypes,
    register_execution_resource,
)
from .common import ClusterInfo, get_cluster_data, get_worker_command
from .secrets import CV_SECRETS_NAME, get_worker_env_vars, secrets_ctx_mngr

logger = log.get_logger("zetta_utils")


def _get_worker_deployment_spec(
    name: str,
    image: str,
    worker_command: str,
    replicas: int,
    resources: Dict[str, int | float | str],
    labels: Dict[str, str],
    env_secret_mapping: Dict[str, str],
) -> k8s_client.V1Deployment:
    volume_mounts = [
        k8s_client.V1VolumeMount(
            mount_path="/root/.cloudvolume/secrets", name=CV_SECRETS_NAME, read_only=True
        ),
        k8s_client.V1VolumeMount(mount_path="/dev/shm", name="dshm"),
        k8s_client.V1VolumeMount(mount_path="/tmp", name="tmp"),
    ]

    container = k8s_client.V1Container(
        command=["/bin/sh"],
        args=["-c", worker_command],
        env=get_worker_env_vars(env_secret_mapping),
        name="zutils-worker",
        image=image,
        image_pull_policy="IfNotPresent",
        resources=k8s_client.V1ResourceRequirements(
            requests=resources,
            limits=resources,
        ),
        termination_message_path="/dev/termination-log",
        termination_message_policy="File",
        volume_mounts=volume_mounts,
    )

    schedule_toleration = k8s_client.V1Toleration(
        key="worker-pool", operator="Equal", value="true", effect="NoSchedule"
    )

    secret = k8s_client.V1SecretVolumeSource(default_mode=420, secret_name=CV_SECRETS_NAME)
    volume0 = k8s_client.V1Volume(name=CV_SECRETS_NAME, secret=secret)
    volume1 = k8s_client.V1Volume(
        name="dshm", empty_dir=k8s_client.V1EmptyDirVolumeSource(medium="Memory")
    )
    volume2 = k8s_client.V1Volume(
        name="tmp", empty_dir=k8s_client.V1EmptyDirVolumeSource(medium="Memory")
    )

    pod_spec = k8s_client.V1PodSpec(
        containers=[container],
        dns_policy="Default",
        restart_policy="Always",
        scheduler_name="default-scheduler",
        security_context={},
        termination_grace_period_seconds=30,
        tolerations=[schedule_toleration],
        volumes=[volume0, volume1, volume2],
    )

    deployment_template = k8s_client.V1PodTemplateSpec(
        metadata=k8s_client.V1ObjectMeta(labels=labels, creation_timestamp=None),
        spec=pod_spec,
    )

    deployment_spec = k8s_client.V1DeploymentSpec(
        progress_deadline_seconds=600,
        replicas=replicas,
        revision_history_limit=10,
        selector=k8s_client.V1LabelSelector(match_labels=labels),
        strategy=k8s_client.V1DeploymentStrategy(
            type="RollingUpdate",
            rolling_update=k8s_client.V1RollingUpdateDeployment(
                max_surge="25%", max_unavailable="25%"
            ),
        ),
        template=deployment_template,
    )

    deployment = k8s_client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=k8s_client.V1ObjectMeta(name=name, labels=labels),
        spec=deployment_spec,
    )

    return deployment


def get_deployment(  # pylint: disable=too-many-locals
    execution_id: str,
    image: str,
    queue: Union[mazepa.ExecutionQueue, Dict[str, Any]],
    replicas: int,
    resources: Dict[str, int | float | str],
    env_secret_mapping: Dict[str, str],
    labels: Optional[Dict[str, str]] = None,
):
    if labels is None:
        labels_final = {"execution_id": execution_id}
    else:
        labels_final = labels

    if isinstance(queue, dict):
        queue_spec = queue
    else:
        if hasattr(queue, "__built_with_spec"):
            queue_spec = queue.__built_with_spec  # pylint: disable=protected-access
        else:
            raise ValueError("Only queue's built by `zetta_utils.builder` are allowed.")
    worker_command = get_worker_command(queue_spec)
    logger.debug(f"Making a deployment with worker command: '{worker_command}'")

    return _get_worker_deployment_spec(
        name=execution_id,
        image=image,
        replicas=replicas,
        worker_command=worker_command,
        resources=resources,
        labels=labels_final,
        env_secret_mapping=env_secret_mapping,
    )


@builder.register("k8s_deployment_ctx_mngr")
@contextmanager
def deployment_ctx_mngr(
    execution_id: str,
    cluster_info: ClusterInfo,
    deployment: k8s_client.V1Deployment,
    secrets: List[k8s_client.V1Secret],
):
    configuration, _ = get_cluster_data(cluster_info)
    k8s_client.Configuration.set_default(configuration)
    k8s_apps_v1_api = k8s_client.AppsV1Api()

    with secrets_ctx_mngr(execution_id, secrets, cluster_info):
        logger.info(f"Creating k8s deployment `{deployment.metadata.name}`")
        k8s_apps_v1_api.create_namespaced_deployment(body=deployment, namespace="default")
        register_execution_resource(
            ExecutionResource(
                execution_id,
                ExecutionResourceTypes.K8S_DEPLOYMENT.value,
                deployment.metadata.name,
            )
        )

        try:
            yield
        finally:
            # new configuration to refresh expired tokens (long running executions)
            configuration, _ = get_cluster_data(cluster_info)
            k8s_client.Configuration.set_default(configuration)

            # need to create a new client for the above to take effect
            k8s_apps_v1_api = k8s_client.AppsV1Api()
            logger.info(f"Deleting k8s deployment `{deployment.metadata.name}`")
            k8s_apps_v1_api.delete_namespaced_deployment(
                name=deployment.metadata.name, namespace="default"
            )
