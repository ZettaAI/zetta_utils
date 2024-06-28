"""
Helpers for k8s configmap.
"""

from contextlib import contextmanager
from typing import Dict, Optional

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import log
from zetta_utils.run import (
    Resource,
    ResourceTypes,
    deregister_resource,
    register_resource,
)

from .common import ClusterInfo, get_cluster_data

logger = log.get_logger("zetta_utils")


def get_configmap(
    name: str,
    annotations: Optional[Dict] = None,
    labels: Optional[Dict] = None,
    data: Optional[Dict[str, str]] = None,
):
    name = f"run-{name}"
    meta = k8s_client.V1ObjectMeta(annotations=annotations, labels=labels, name=name)
    return k8s_client.V1ConfigMap(metadata=meta, data=data)


@contextmanager
def configmap_ctx_manager(
    run_id: str,
    cluster_info: ClusterInfo,
    configmap: k8s_client.V1ConfigMap,
    namespace: Optional[str] = "default",
):
    configuration, _ = get_cluster_data(cluster_info)
    k8s_client.Configuration.set_default(configuration)
    k8s_core_v1_api = k8s_client.CoreV1Api()

    logger.info(f"Creating k8s configmap `{configmap.metadata.name}`")
    k8s_core_v1_api.create_namespaced_config_map(body=configmap, namespace=namespace)
    _id = register_resource(
        Resource(
            run_id,
            ResourceTypes.K8S_CONFIGMAP.value,
            configmap.metadata.name,
        )
    )

    try:
        yield
    finally:
        # new configuration to refresh expired tokens (long running executions)
        configuration, _ = get_cluster_data(cluster_info)
        k8s_client.Configuration.set_default(configuration)

        # need to create a new client for the above to take effect
        k8s_core_v1_api = k8s_client.CoreV1Api()
        logger.info(f"Deleting k8s configmap `{configmap.metadata.name}`")
        k8s_core_v1_api.delete_namespaced_config_map(
            name=configmap.metadata.name,
            namespace=namespace,
        )
        deregister_resource(_id)
