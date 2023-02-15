"""
Tools to interact with kubernetes clusters.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import attrs

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import builder, log, mazepa

from ..gcloud import crm as gcloud_crm
from .eks import eks_cluster_data
from .gke import gke_cluster_data

logger = log.get_logger("zetta_utils")

CV_SECRETS_NAME = "cloudvolume-secrets"


@builder.register("mazepa.k8s.ClusterInfo")
@attrs.frozen
class ClusterInfo:
    name: str
    region: Optional[str] = None
    project: Optional[str] = None


@attrs.frozen
class ClusterAuth:
    cert_name: str
    endpoint: str
    token: str


def _get_worker_command(queue_spec: Dict[str, Any]):
    result = (
        """
    zetta -vv -l try run -s '{
        "@type": "mazepa.run_worker"
        exec_queue:
    """
        + json.dumps(queue_spec)
        + """
        max_pull_num: 1
        sleep_sec: 5
    }'
    """
    )
    return result


def _get_worker_env_vars(env_secret_mapping: Dict[str, str]) -> list:
    name_path_map = {
        "MY_NODE_NAME": "spec.nodeName",
        "MY_POD_NAME": "metadata.name",
        "MY_POD_NAMESPACE": "metadata.namespace",
        "MY_POD_IP": "status.podIP",
        "MY_POD_SERVICE_ACCOUNT": "spec.serviceAccountName",
    }
    envs = [
        k8s_client.V1EnvVar(
            name=name,
            value_from=k8s_client.V1EnvVarSource(
                field_ref=k8s_client.V1ObjectFieldSelector(field_path=path)
            ),
        )
        for name, path in name_path_map.items()
    ]

    for k, v in env_secret_mapping.items():
        env_var = k8s_client.V1EnvVar(
            name=k,
            value_from=k8s_client.V1EnvVarSource(
                secret_key_ref=k8s_client.V1SecretKeySelector(key="value", name=v, optional=False)
            ),
        )
        envs.append(env_var)
    return envs


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
        args=["-c", worker_command],
        command=["/bin/sh"],
        env=_get_worker_env_vars(env_secret_mapping),
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


def get_secrets_and_mapping(
    execution_id: str, share_envs: Iterable[str] = ()
) -> Tuple[List[k8s_client.V1Secret], Dict[str, str]]:
    env_secret_mapping: Dict[str, str] = {}
    secrets_kv: Dict[str, str] = {}

    for env_k in share_envs:
        if not env_k.isupper() or not env_k.replace("_", "").isalpha():
            raise ValueError(
                "Only able to share environment variables with "
                f"only upper letters and underscores. Got: `{env_k}`"
            )
        env_v = os.environ.get(env_k, None)
        if env_v is None:
            raise ValueError(
                f"Please set `{env_k}` environment variable in order to create a deployment."
            )
        secret_name = f"{execution_id}-{env_k}".lower().replace("_", "-")
        env_secret_mapping[env_k] = secret_name
        secrets_kv[secret_name] = env_v

    secrets = []
    for k, v in secrets_kv.items():
        secret = k8s_client.V1Secret(
            metadata=k8s_client.V1ObjectMeta(name=k),
            string_data={"value": v},
        )
        secrets.append(secret)
    return secrets, env_secret_mapping


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
    worker_command = _get_worker_command(queue_spec)
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


def _get_provider_cluster_data(info: ClusterInfo) -> Tuple[ClusterAuth, str]:
    if info.project is not None:
        assert info.region is not None, "GKE cluster needs both `project` and `region`."
        logger.info("Cluster provider: GKE/GCP.")

        cluster_data, cert, token = gke_cluster_data(info.name, info.region, info.project)
        endpoint = cluster_data.endpoint
        workload_pool = cluster_data.workload_identity_config.workload_pool
    else:
        logger.info("Cluster provider: EKS/AWS.")

        cluster_data, cert, token = eks_cluster_data(info.name)
        endpoint = cluster_data["endpoint"]
        workload_pool = cluster_data["workload_identity_config"]["workload_pool"]

    return ClusterAuth(cert, endpoint, token), workload_pool


def add_workload_identity_role(
    execution_id: str, info: ClusterInfo, workload_pool: str, principal: str
) -> None:
    if info.project is not None:
        assert info.region is not None, "GKE cluster needs both `project` and `region`."

        member = f"serviceAccount:{workload_pool}[{execution_id}/default]"
        gcloud_crm.add_role(
            info.project, gcloud_crm.Role.WORKLOAD_IDENTITY_USER, member, principal=principal
        )
    else:
        NotImplementedError("Other provider clusters not supported at this time.")


def rm_workload_identity_role(
    execution_id: str, info: ClusterInfo, workload_pool: str, principal: str
) -> None:
    if info.project is not None:
        assert info.region is not None, "GKE cluster needs both `project` and `region`."

        member = f"serviceAccount:{workload_pool}[{execution_id}/default]"
        gcloud_crm.remove_role(
            info.project, gcloud_crm.Role.WORKLOAD_IDENTITY_USER, member, principal=principal
        )
    else:
        NotImplementedError("Other provider clusters not supported at this time.")


def get_cluster_data(info: ClusterInfo) -> Tuple[k8s_client.Configuration, str]:
    cluster_auth, workload_pool = _get_provider_cluster_data(info)

    logger.debug(f"Cluster endpoint: {cluster_auth.endpoint}")
    configuration = k8s_client.Configuration()
    configuration.host = f"https://{cluster_auth.endpoint}"
    configuration.ssl_ca_cert = cluster_auth.cert_name
    configuration.api_key_prefix["authorization"] = "Bearer"
    configuration.api_key["authorization"] = cluster_auth.token
    return configuration, workload_pool


def get_worker_sa(k8s_api: k8s_client.CoreV1Api) -> str:
    """Extract pre-configured service account for workload identity access in default ns"""
    try:
        default_ksa = k8s_api.read_namespaced_service_account(name="default", namespace="default")
        return default_ksa.metadata.annotations["iam.gke.io/gcp-service-account"]
    except KeyError as err:
        raise KeyError(f"Workload Identity not configured: {err}") from err


def _copy_cv_secrets_from_default_ns(k8s_api: k8s_client.CoreV1Api, execution_id: str) -> None:
    """Copy pre-configured cloudvolume secrets from default ns to current ns."""
    namespaced_cv_secret = k8s_api.read_namespaced_secret(
        name=CV_SECRETS_NAME, namespace="default"
    )

    old_metadata = namespaced_cv_secret.metadata
    new_metadata = k8s_client.V1ObjectMeta(
        name=CV_SECRETS_NAME,
        namespace=execution_id,
        annotations=old_metadata.annotations,
        labels=old_metadata.labels,
    )

    namespaced_cv_secret.metadata = new_metadata
    k8s_api.create_namespaced_secret(namespace=execution_id, body=namespaced_cv_secret)


@builder.register("k8s_namespace_ctx_mngr")
@contextmanager
def namespace_ctx_mngr(
    execution_id: str,
    cluster_info: ClusterInfo,
    secrets: List[k8s_client.V1Secret],
    deployments: List[k8s_client.V1Deployment],
):
    configuration, workload_pool = get_cluster_data(cluster_info)
    k8s_client.Configuration.set_default(configuration)
    k8s_core_v1_api = k8s_client.CoreV1Api()
    k8s_apps_v1_api = k8s_client.AppsV1Api()

    namespace_config = k8s_client.V1Namespace(metadata=k8s_client.V1ObjectMeta(name=execution_id))

    logger.info(f"Creating k8s namespace `{execution_id}`")
    k8s_core_v1_api.create_namespace(namespace_config)

    worker_sa = get_worker_sa(k8s_core_v1_api)
    logger.debug(f"Adding ksa workload identity access via `{worker_sa}`")
    add_workload_identity_role(execution_id, cluster_info, workload_pool, principal=worker_sa)

    _copy_cv_secrets_from_default_ns(k8s_core_v1_api, execution_id)
    for secret in secrets:
        logger.info(f"Creating namespaced k8s secret `{secret.metadata.name}`")
        k8s_core_v1_api.create_namespaced_secret(namespace=execution_id, body=secret)

    for deployment in deployments:
        logger.info(f"Creating namespaced k8s deployment `{deployment.metadata.name}`")
        k8s_apps_v1_api.create_namespaced_deployment(body=deployment, namespace=execution_id)

    try:
        yield
    finally:
        # new configuration to refresh expired tokens (long running executions)
        configuration, workload_pool = get_cluster_data(cluster_info)

        logger.debug(f"Removing ksa workload identity access via `{worker_sa}`")
        rm_workload_identity_role(execution_id, cluster_info, workload_pool, principal=worker_sa)

        k8s_client.Configuration.set_default(configuration)
        # need to create a new client for the above to take effect?
        k8s_core_v1_api = k8s_client.CoreV1Api()
        logger.info(f"Deleting k8s namespace `{execution_id}`")
        k8s_core_v1_api.delete_namespace(name=execution_id)
