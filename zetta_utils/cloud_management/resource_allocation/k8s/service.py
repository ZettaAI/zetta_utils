"""
Helpers for k8s service.
"""

from contextlib import contextmanager
from typing import Dict, List, Mapping, Optional

from kubernetes.client.exceptions import ApiException

from kubernetes import client as k8s_client
from zetta_utils import log
from zetta_utils.run.resource import (
    Resource,
    ResourceTypes,
    deregister_resource,
    register_resource,
)

from .common import ClusterInfo, get_cluster_data

logger = log.get_logger("zetta_utils")


def get_headless_service(
    name: str,
    selector_labels: Dict[str, str],
):
    name = f"run-{name}"
    meta = k8s_client.V1ObjectMeta(name=name, labels=selector_labels)
    service_spec = k8s_client.V1ServiceSpec(
        cluster_ip="None",
        selector=selector_labels,
        publish_not_ready_addresses=True,
    )
    return k8s_client.V1Service(metadata=meta, spec=service_spec)


def get_service(
    name: str,
    annotations: Optional[Dict] = None,
    labels: Optional[Dict] = None,
    ports: Optional[List[k8s_client.V1ServicePort]] = None,
    selector: Optional[Dict[str, str]] = None,
    service_type: Optional[str] = None,
):
    name = f"run-{name}"
    meta = k8s_client.V1ObjectMeta(annotations=annotations, labels=labels, name=name)
    service_spec = k8s_client.V1ServiceSpec(ports=ports, selector=selector, type=service_type)
    return k8s_client.V1Service(metadata=meta, spec=service_spec)


def create_namespaced_service(
    *,
    namespace: str,
    body: Mapping,
    k8s_core_v1_api: k8s_client.CoreV1Api | None = None,
) -> k8s_client.V1Service:
    """Create a Service via ``CoreV1Api``.

    Mirrors :func:`create_namespaced_pod`'s call shape — keyword-only arguments
    and an optional explicit API client for testability.

    :param namespace: namespace to create the Service in.
    :param body: a dict matching ``V1Service``. The caller renders this from a
        Service template.
    :param k8s_core_v1_api: optional override; tests inject a mock.
    :return: the created ``V1Service``.
    """
    api = k8s_core_v1_api or k8s_client.CoreV1Api()
    return api.create_namespaced_service(namespace=namespace, body=body)


def delete_namespaced_service(
    *,
    name: str,
    namespace: str,
    k8s_core_v1_api: k8s_client.CoreV1Api | None = None,
) -> None:
    """Best-effort delete of a Service via ``CoreV1Api``.

    Swallows ``ApiException`` with status 404 or 410 (the Service is already
    gone), re-raising any other status.

    :param name: Service name.
    :param namespace: Service namespace.
    :param k8s_core_v1_api: optional override; tests inject a mock.
    """
    api = k8s_core_v1_api or k8s_client.CoreV1Api()
    try:
        api.delete_namespaced_service(name=name, namespace=namespace)
    except ApiException as e:
        if e.status in (404, 410):
            return
        raise


@contextmanager
def service_ctx_manager(
    run_id: str,
    cluster_info: ClusterInfo,
    service: k8s_client.V1Service,
    namespace: Optional[str] = "default",
):
    configuration, _ = get_cluster_data(cluster_info)
    k8s_client.Configuration.set_default(configuration)
    k8s_core_v1_api = k8s_client.CoreV1Api()

    logger.info(f"Creating k8s service `{service.metadata.name}`")
    k8s_core_v1_api.create_namespaced_service(body=service, namespace=namespace)

    _id = register_resource(
        Resource(
            run_id,
            ResourceTypes.K8S_SERVICE.value,
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
            name=service.metadata.name,
            namespace=namespace,
        )
        deregister_resource(_id)
