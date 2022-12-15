from __future__ import annotations

import json
import os
from contextlib import ExitStack, contextmanager
from typing import Any, Dict, Iterable, Optional, Union

import kubernetes as k8s
from zetta_utils import builder, log, mazepa

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


def get_worker_deployment_spec(
    name: str,
    image: str,
    worker_command: str,
    replicas: int,
    resources: Dict[str, int | float | str],
    labels: Dict[str, str],
    env_secret_mapping: Dict[str, str],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "labels": labels,
            "name": name,
            "namespace": "default",
        },
        "spec": {
            "progressDeadlineSeconds": 600,
            "replicas": replicas,
            "revisionHistoryLimit": 10,
            "selector": {
                "matchLabels": labels,
            },
            "strategy": {
                "rollingUpdate": {"maxSurge": "25%", "maxUnavailable": "25%"},
                "type": "RollingUpdate",
            },
            "template": {
                "metadata": {
                    "creationTimestamp": None,
                    "labels": labels,
                },
                "spec": {
                    "containers": [
                        {
                            "name": "zutils-worker",
                            "env": [
                                {
                                    "name": "MY_NODE_NAME",
                                    "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
                                },
                                {
                                    "name": "MY_POD_NAME",
                                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
                                },
                                {
                                    "name": "MY_POD_NAMESPACE",
                                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}},
                                },
                                {
                                    "name": "MY_POD_IP",
                                    "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
                                },
                                {
                                    "name": "MY_POD_SERVICE_ACCOUNT",
                                    "valueFrom": {
                                        "fieldRef": {"fieldPath": "spec.serviceAccountName"}
                                    },
                                },
                            ],
                            "args": ["-c", worker_command],
                            "command": ["/bin/sh"],
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "resources": {
                                "limits": resources,
                                "requests": resources,
                            },
                            "terminationMessagePath": "/dev/termination-log",
                            "terminationMessagePolicy": "File",
                            "volumeMounts": [
                                {
                                    "mountPath": "/root/.cloudvolume/secrets",
                                    "name": "cloudvolume-secrets",
                                    "readOnly": True,
                                },
                                {"mountPath": "/tmp", "name": "tmp"},
                                {"mountPath": "/dev/shm", "name": "dshm"},
                            ],
                        }
                    ],
                    "dnsPolicy": "Default",
                    "restartPolicy": "Always",
                    "schedulerName": "default-scheduler",
                    "securityContext": {},
                    "terminationGracePeriodSeconds": 30,
                    "volumes": [
                        {
                            "name": "cloudvolume-secrets",
                            "secret": {"defaultMode": 420, "secretName": "cloudvolume-secrets"},
                        },
                        {"emptyDir": {"medium": "Memory"}, "name": "tmp"},
                        {"emptyDir": {"medium": "Memory"}, "name": "dshm"},
                    ],
                },
            },
        },
    }
    for k, v in env_secret_mapping.items():
        result["spec"]["template"]["spec"]["containers"][0]["env"].append(
            {
                "name": k,
                "valueFrom": {
                    "secretKeyRef": {
                        "name": v,
                        "key": "value",
                        "optional": False,
                    }
                },
            }
        )
    return result


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
        if hasattr(queue, "__init_builder_spec"):
            queue_spec = queue.__init_builder_spec  # pylint: disable=protected-access
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
    k8s_apps_v1_api = k8s.client.AppsV1Api()  # type: ignore

    logger.info(f"Creating deployment `{deployment_spec['metadata']['name']}`")
    k8s_apps_v1_api.create_namespaced_deployment(body=deployment_spec, namespace="default")

    secret_ctx_mngrs = [k8s_secret_ctx_mngr(k, {"value": v}) for k, v in secrets_kv.items()]
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

    try:
        yield
    finally:
        logger.info(f"Deleting secret `{name}`")
        k8s_core_v1_api.delete_namespaced_secret(
            name=name,
            namespace="default",
        )
