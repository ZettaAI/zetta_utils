"""Preflight check at master startup: verify (1) the master can
authenticate to the cluster, and (2) the named resources from
``scripts/gcp/rbac.yml`` exist. Surfaces gaps as clear errors instead
of opaque 401-loops in daemon threads or 403s in worker pod logs."""

from __future__ import annotations

import kubernetes.client as k8s_client
from kubernetes.client.exceptions import ApiException

from .common import ClusterInfo, get_cluster_data


def _raise_auth_error(cluster_info: ClusterInfo, exc: ApiException) -> None:
    raise PermissionError(
        f"Cannot access cluster {cluster_info.name!r} "
        f"(project={cluster_info.project}): {exc.status} {exc.reason}. "
        f"If 401: run `gcloud auth application-default login`. "
        f"If 403: grant `roles/container.developer` on the project."
    ) from exc


def verify_cluster_access(cluster_info: ClusterInfo, namespace: str = "default") -> None:
    """Verify the master can use the cluster and rbac.yml has been applied."""
    configuration, _ = get_cluster_data(cluster_info)
    api_client = k8s_client.ApiClient(configuration=configuration)
    core_api = k8s_client.CoreV1Api(api_client=api_client)
    apps_api = k8s_client.AppsV1Api(api_client=api_client)
    rbac = k8s_client.RbacAuthorizationV1Api(api_client=api_client)

    try:
        core_api.list_namespaced_pod(namespace=namespace, limit=1)
        apps_api.list_namespaced_deployment(namespace=namespace, limit=1)
    except ApiException as exc:
        _raise_auth_error(cluster_info, exc)

    checks = [
        (rbac.read_namespaced_role, "pod-reader", {"namespace": namespace}, "Role"),
        (rbac.read_namespaced_role, "pod-metrics-reader", {"namespace": namespace}, "Role"),
        (
            rbac.read_namespaced_role_binding,
            "pod-reader-binding",
            {"namespace": namespace},
            "RoleBinding",
        ),
        (
            rbac.read_namespaced_role_binding,
            "read-pod-metrics-binding",
            {"namespace": namespace},
            "RoleBinding",
        ),
        (rbac.read_cluster_role, "node-reader", {}, "ClusterRole"),
        (rbac.read_cluster_role_binding, "node-reader-binding", {}, "ClusterRoleBinding"),
    ]

    missing: list[str] = []
    for read_fn, name, kwargs, kind in checks:
        try:
            read_fn(name=name, **kwargs)
        except ApiException as exc:
            if exc.status == 404:
                missing.append(f"{kind}/{name}")
                continue
            _raise_auth_error(cluster_info, exc)

    if missing:
        raise PermissionError(
            f"Cluster {cluster_info.name!r} is missing rbac.yml resources: "
            f"{missing}. Apply: "
            f"`kubectl apply -f scripts/gcp/rbac.yml --context={cluster_info.name}`"
        )
