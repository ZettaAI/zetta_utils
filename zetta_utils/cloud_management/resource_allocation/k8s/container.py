"""Container utilities for Kubernetes pods."""

from __future__ import annotations

import os

from kubernetes import client as k8s_client
from kubernetes import config  # type: ignore


def get_main_container_status() -> int:
    """Check if main container is still running.

    Returns:
        -1 if running or status unknown, exit code otherwise.
    """
    if k8s_client is None or config is None:
        return -1

    try:
        pod_name = os.environ.get("POD_NAME")
        namespace = os.environ.get("POD_NAMESPACE", "default")
        if not pod_name:
            return -1

        config.load_incluster_config()
        v1 = k8s_client.CoreV1Api()
        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        for container_status in pod.status.container_statuses or []:
            if container_status.name == "main":
                state = container_status.state
                if state.terminated:
                    return int(state.terminated.exit_code)
        return -1
    except Exception:  # pylint: disable=broad-exception-caught
        return -1
