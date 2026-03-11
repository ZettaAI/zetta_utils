"""
Helpers for k8s statefulsets.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional

from kubernetes import client as k8s_client
from zetta_utils import log
from zetta_utils.run.resource import (
    Resource,
    ResourceTypes,
    deregister_resource,
    register_resource,
)

from .common import ClusterInfo, get_cluster_data
from .secret import secrets_ctx_mngr

logger = log.get_logger("zetta_utils")


def get_statefulset(
    name: str,
    pod_spec: k8s_client.V1PodSpec,
    replicas: int,
    service_name: str,
    labels: Optional[Dict[str, str]] = None,
    revision_history_limit: Optional[int] = 10,
) -> k8s_client.V1StatefulSet:
    name = f"run-{name}"
    labels = labels or {"app": name}
    pod_template = k8s_client.V1PodTemplateSpec(
        metadata=k8s_client.V1ObjectMeta(labels=labels),
        spec=pod_spec,
    )

    statefulset_spec = k8s_client.V1StatefulSetSpec(
        replicas=replicas,
        revision_history_limit=revision_history_limit,
        selector=k8s_client.V1LabelSelector(match_labels=labels),
        service_name=service_name,
        pod_management_policy="Parallel",
        template=pod_template,
    )

    statefulset = k8s_client.V1StatefulSet(
        metadata=k8s_client.V1ObjectMeta(name=name, labels=labels),
        spec=statefulset_spec,
    )

    return statefulset


@contextmanager
def statefulset_ctx_manager(
    run_id: str,
    cluster_info: ClusterInfo,
    statefulset: k8s_client.V1StatefulSet,
    secrets: List[k8s_client.V1Secret],
    namespace: str = "default",
):
    configuration, _ = get_cluster_data(cluster_info)
    k8s_client.Configuration.set_default(configuration)
    k8s_apps_v1_api = k8s_client.AppsV1Api()

    with secrets_ctx_mngr(run_id, secrets, cluster_info, namespace=namespace):
        logger.info(f"Creating k8s statefulset `{statefulset.metadata.name}`")
        k8s_apps_v1_api.create_namespaced_stateful_set(body=statefulset, namespace=namespace)
        _id = register_resource(
            Resource(
                run_id,
                ResourceTypes.K8S_STATEFULSET.value,
                statefulset.metadata.name,
            )
        )

        try:
            yield
        finally:
            configuration, _ = get_cluster_data(cluster_info)
            k8s_client.Configuration.set_default(configuration)

            k8s_apps_v1_api = k8s_client.AppsV1Api()
            logger.info(f"Deleting k8s statefulset `{statefulset.metadata.name}`")
            k8s_apps_v1_api.delete_namespaced_stateful_set(
                name=statefulset.metadata.name, namespace=namespace
            )
            deregister_resource(_id)
