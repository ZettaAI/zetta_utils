from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

import kubernetes as k8s
from zetta_utils import builder, log, mazepa

logger = log.get_logger("zetta_utils")


def get_worker_command(queue_spec: Dict[str, Any]):
    result = (
        """
    zetta -vv -l inference run -s '{
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
    zetta_user: str,
    zetta_project: str,
) -> Dict[str, Any]:
    result = {
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
                                    "name": "GRAFANA_CLOUD_ACCESS_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "grafana-cloud-api-sergiy",
                                            "key": "key",
                                            "optional": False,
                                        }
                                    },
                                },
                                {
                                    "name": "AWS_ACCESS_KEY_ID",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "aws-key-sergiy",
                                            "key": "access_key_id",
                                            "optional": False,
                                        }
                                    },
                                },
                                {
                                    "name": "AWS_SECRET_ACCESS_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "aws-key-sergiy",
                                            "key": "secret_access_key",
                                            "optional": False,
                                        }
                                    },
                                },
                                {
                                    "name": "ZETTA_USER",
                                    "value": zetta_user,
                                },
                                {
                                    "name": "ZETTA_PROJECT",
                                    "value": zetta_project,
                                },
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
    return result


@builder.register("worker_k8s_deployment_ctx_mngr")
@contextmanager
def worker_k8s_deployment_ctx_mngr(
    execution_id: str,
    image: str,
    queue: Union[mazepa.ExecutionQueue, Dict[str, Any]],
    replicas: int,
    resources: Dict[str, int | float | str],
    labels: Optional[Dict[str, str]] = None,
    zetta_user: str = os.environ.get("ZETTA_USER", "UNDEFINED"),
    zetta_project: str = os.environ.get("ZETTA_USER", "UNDEFINED"),
):
    k8s.config.load_kube_config()  # type: ignore
    labels_final = {"zetta_user": zetta_user}
    if labels is not None:
        labels_final = {**labels_final, **labels}
    # TODO: raise error on UNDEFINES

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
        zetta_user=zetta_user,
        zetta_project=zetta_project,
    )
    k8s_apps_v1_api = k8s.client.AppsV1Api()  # type: ignore

    logger.info(f"Creating deployment `{deployment_spec['metadata']['name']}`")
    k8s_apps_v1_api.create_namespaced_deployment(body=deployment_spec, namespace="default")
    try:
        yield
    finally:
        logger.info(f"Deleting deployment `{deployment_spec['metadata']['name']}`")
        k8s_apps_v1_api.delete_namespaced_deployment(
            name=deployment_spec["metadata"]["name"],
            namespace="default",
        )
