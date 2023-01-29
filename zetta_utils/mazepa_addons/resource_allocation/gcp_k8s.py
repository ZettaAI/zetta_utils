from __future__ import annotations

import json
import os
from contextlib import ExitStack, contextmanager
from typing import Any, Dict, Iterable, Optional, Union, no_type_check

import kubernetes as k8s
from zetta_utils import builder, log, mazepa

from .tracker import ExecutionResource, register_execution_resource

logger = log.get_logger("zetta_utils")


def get_worker_command(queue_spec: Dict[str, Any]):
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
        k8s.client.V1EnvVar(  # type: ignore
            name=name,
            value_from=k8s.client.V1EnvVarSource(  # type: ignore
                field_ref=k8s.client.V1ObjectFieldSelector(field_path=path)  # type: ignore
            ),
        )
        for name, path in name_path_map.items()
    ]

    for k, v in env_secret_mapping.items():
        env_var = k8s.client.V1EnvVar(  # type: ignore
            name=k,
            value_from=k8s.client.V1EnvVarSource(  # type: ignore
                secret_key_ref=k8s.client.V1SecretKeySelector(  # type: ignore
                    key="value", name=v, optional=False
                )
            ),
        )
        envs.append(env_var)
    return envs


@no_type_check
def get_worker_deployment_spec(
    name: str,
    image: str,
    worker_command: str,
    replicas: int,
    resources: Dict[str, int | float | str],
    labels: Dict[str, str],
    env_secret_mapping: Dict[str, str],
) -> k8s.client.V1Deployment:
    volume_mounts = [
        k8s.client.V1VolumeMount(
            mount_path="/root/.cloudvolume/secrets", name="cloudvolume-secrets", read_only=True
        ),
        k8s.client.V1VolumeMount(mount_path="/dev/shm", name="dshm"),
        k8s.client.V1VolumeMount(mount_path="/tmp", name="tmp"),
    ]

    container = k8s.client.V1Container(
        args=["-c", worker_command],
        command=["/bin/sh"],
        env=_get_worker_env_vars(env_secret_mapping),
        name="zutils-worker",
        image=image,
        image_pull_policy="IfNotPresent",
        resources=k8s.client.V1ResourceRequirements(
            requests=resources,
            limits=resources,
        ),
        termination_message_path="/dev/termination-log",
        termination_message_policy="File",
        volume_mounts=volume_mounts,
    )

    schedule_toleration = k8s.client.V1Toleration(
        key="worker-pool", operator="Equal", value="true", effect="NoSchedule"
    )

    secret = k8s.client.V1SecretVolumeSource(default_mode=420, secret_name="cloudvolume-secrets")
    volume0 = k8s.client.V1Volume(name="cloudvolume-secrets", secret=secret)
    volume1 = k8s.client.V1Volume(
        name="dshm", empty_dir=k8s.client.V1EmptyDirVolumeSource(medium="Memory")
    )
    volume2 = k8s.client.V1Volume(
        name="tmp", empty_dir=k8s.client.V1EmptyDirVolumeSource(medium="Memory")
    )

    pod_spec = k8s.client.V1PodSpec(
        containers=[container],
        dns_policy="Default",
        restart_policy="Always",
        scheduler_name="default-scheduler",
        security_context={},
        termination_grace_period_seconds=30,
        tolerations=[schedule_toleration],
        volumes=[volume0, volume1, volume2],
    )

    deployment_template = k8s.client.V1PodTemplateSpec(
        metadata=k8s.client.V1ObjectMeta(labels=labels, creation_timestamp=None),
        spec=pod_spec,
    )

    deployment_spec = k8s.client.V1DeploymentSpec(
        progress_deadline_seconds=600,
        replicas=replicas,
        revision_history_limit=10,
        selector=k8s.client.V1LabelSelector(match_labels=labels),
        strategy=k8s.client.V1DeploymentStrategy(
            type="RollingUpdate",
            rolling_update=k8s.client.V1RollingUpdateDeployment(
                max_surge="25%", max_unavailable="25%"
            ),
        ),
        template=deployment_template,
    )

    deployment = k8s.client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=k8s.client.V1ObjectMeta(name=name, labels=labels),
        spec=deployment_spec,
    )

    return deployment


@builder.register("worker_k8s_deployment_ctx_mngr")
@contextmanager
def worker_k8s_deployment_ctx_mngr(  # pylint: disable=too-many-locals
    execution_id: str,
    image: str,
    queue: Union[mazepa.ExecutionQueue, Dict[str, Any]],
    replicas: int,
    resources: Dict[str, int | float | str],
    labels: Optional[Dict[str, str]] = None,
    share_envs: Iterable[str] = (),
):
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
                f"Please set `{env_k}` environment variable " "in order to create a deployment."
            )
        secret_name = f"{execution_id}-{env_k}".lower().replace("_", "-")
        env_secret_mapping[env_k] = secret_name
        secrets_kv[secret_name] = env_v

    k8s.config.load_kube_config()  # type: ignore

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

    deployment_spec = get_worker_deployment_spec(
        name=execution_id,
        image=image,
        replicas=replicas,
        worker_command=worker_command,
        resources=resources,
        labels=labels_final,
        env_secret_mapping=env_secret_mapping,
    )

    # create secrets to be mounted into deployment first
    secret_ctx_mngrs = [
        k8s_secret_ctx_mngr(execution_id, k, {"value": v}) for k, v in secrets_kv.items()
    ]
    k8s_apps_v1_api = k8s.client.AppsV1Api()  # type: ignore

    logger.info(f"Creating deployment `{deployment_spec['metadata']['name']}`")
    k8s_apps_v1_api.create_namespaced_deployment(body=deployment_spec, namespace="default")

    register_execution_resource(ExecutionResource(execution_id, "k8s_deployment", execution_id))

    with ExitStack() as stack:
        for mngr in secret_ctx_mngrs:
            stack.enter_context(mngr)

        try:
            yield
        finally:
            logger.info(f"Deleting deployment `{deployment_spec['metadata']['name']}`")
            k8s_apps_v1_api.delete_namespaced_deployment(
                name=deployment_spec["metadata"]["name"],
                namespace="default",
            )


@builder.register("k8s_secret_ctx_mngr")
@contextmanager
def k8s_secret_ctx_mngr(
    execution_id: str,
    name: str,
    string_data: Dict[str, str],
):
    k8s.config.load_kube_config()  # type: ignore
    secret = k8s.client.V1Secret(  # type: ignore
        api_version="v1",
        kind="Secret",
        metadata=k8s.client.V1ObjectMeta(name=name),  # type: ignore
        string_data=string_data,
    )

    k8s_core_v1_api = k8s.client.CoreV1Api()  # type: ignore
    logger.info(f"Creating secret `{name}`")
    k8s_core_v1_api.create_namespaced_secret(namespace="default", body=secret)

    register_execution_resource(ExecutionResource(execution_id, "k8s_secret", name))

    try:
        yield
    finally:
        logger.info(f"Deleting secret `{name}`")
        k8s_core_v1_api.delete_namespaced_secret(
            name=name,
            namespace="default",
        )
