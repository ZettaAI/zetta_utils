"""
Helpers for k8s service.
"""

from contextlib import contextmanager
from typing import Dict, List, Optional

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import log

from ..resource_tracker import (
    ExecutionResource,
    ExecutionResourceTypes,
    register_execution_resource,
)
from .common import ClusterInfo, get_cluster_data

logger = log.get_logger("zetta_utils")


def get_service(
    name: str,
    namespace: str,
    annotations: Optional[Dict] = None,
    labels: Optional[Dict] = None,
    selector: Optional[Dict] = None,
    service_type: Optional[str] = None,
    ports: Optional[List[k8s_client.V1ServicePort]] = None,
):
    meta = k8s_client.V1ObjectMeta(
        annotations=annotations, labels=labels, name=name, namespace=namespace
    )
    service_spec = k8s_client.V1ServiceSpec(ports=ports, selector=selector, type=service_type)
    return k8s_client.V1Service(metadata=meta, spec=service_spec)


@contextmanager
def service_ctx_manager(
    execution_id: str,
    cluster_info: ClusterInfo,
    service: k8s_client.V1Service,
):
    configuration, _ = get_cluster_data(cluster_info)
    k8s_client.Configuration.set_default(configuration)
    k8s_core_v1_api = k8s_client.CoreV1Api()

    logger.info(f"Creating k8s service `{service.metadata.name}`")
    k8s_core_v1_api.create_namespaced_service(body=service, namespace=service.metadata.namespace)
    register_execution_resource(
        ExecutionResource(
            execution_id,
            ExecutionResourceTypes.K8S_SERVICE.value,
            service.metadata.name,
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
        logger.info(f"Deleting k8s service `{service.metadata.name}`")
        k8s_core_v1_api.delete_namespaced_service(
            name=service.metadata.name, namespace=service.metadata.namespace
        )
