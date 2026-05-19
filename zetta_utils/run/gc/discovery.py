"""Cluster routing for the run garbage collector via runtime discovery.

For each registered k8s resource, find which cluster currently hosts an
object with that ``(type, name)`` by listing once per
``(cluster, managed kind)`` and consulting the presence map.

Cost per invocation is bounded by ``len(run_clusters) * len(managed_kinds)``
list calls, independent of resource count per run.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from typing import Any, Callable

import attrs
import kubernetes.client as k8s_client
from google.api_core.exceptions import GoogleAPICallError

from zetta_utils.cloud_management.resource_allocation.k8s.common import ClusterInfo
from zetta_utils.run.gc.deleters import (
    K8S_DELETERS,
    get_cluster_clients,
    reset_cluster_clients,
)
from zetta_utils.run.gc.utils import retried
from zetta_utils.run.resource import Resource, ResourceTypes


@attrs.frozen
class ClusterFailure:
    """A cluster-wide failure that prevents discovery for that cluster.

    :param error_class: ``"cluster_404"`` (cluster gone) or
        ``"cluster_auth"`` (auth / unreachable / 5xx after retries).
    :param error: One-line human-readable detail for logging / Slack.
    """

    error_class: str
    error: str


_ListFn = Callable[[k8s_client.CoreV1Api, k8s_client.AppsV1Api, k8s_client.BatchV1Api], set[str]]


def _names_from_typed_list(response: Any) -> set[str]:
    return {item.metadata.name for item in response.items}


def _names_from_custom_list(response: Any) -> set[str]:
    items = response.get("items", []) if isinstance(response, dict) else []
    return {item["metadata"]["name"] for item in items}


def _list_keda(plural: str) -> _ListFn:
    """Build a list_fn that enumerates a KEDA CRD via ``CustomObjectsApi``.

    The custom client is built from the shared ``api_client`` so the
    per-cluster credentials cached in :func:`get_cluster_clients` are
    reused without a second auth round-trip.
    """

    def list_fn(core: k8s_client.CoreV1Api, _apps: Any, _batch: Any) -> set[str]:
        custom = k8s_client.CustomObjectsApi(api_client=core.api_client)
        response = custom.list_namespaced_custom_object(
            group="keda.sh", version="v1alpha1", namespace="default", plural=plural
        )
        return _names_from_custom_list(response)

    return list_fn


_LIST_FNS: dict[str, _ListFn] = {
    ResourceTypes.K8S_CONFIGMAP.value: lambda core, apps, batch: _names_from_typed_list(
        core.list_namespaced_config_map(namespace="default")
    ),
    ResourceTypes.K8S_DEPLOYMENT.value: lambda core, apps, batch: _names_from_typed_list(
        apps.list_namespaced_deployment(namespace="default")
    ),
    ResourceTypes.K8S_JOB.value: lambda core, apps, batch: _names_from_typed_list(
        batch.list_namespaced_job(namespace="default")
    ),
    ResourceTypes.K8S_SECRET.value: lambda core, apps, batch: _names_from_typed_list(
        core.list_namespaced_secret(namespace="default")
    ),
    ResourceTypes.K8S_SERVICE.value: lambda core, apps, batch: _names_from_typed_list(
        core.list_namespaced_service(namespace="default")
    ),
    ResourceTypes.K8S_STATEFULSET.value: lambda core, apps, batch: _names_from_typed_list(
        apps.list_namespaced_stateful_set(namespace="default")
    ),
    # --- transient legacy KEDA listings; remove once orphan RESOURCE_DB rows are drained ---
    "ScaledJob": _list_keda("scaledjobs"),
    "ScaledObject": _list_keda("scaledobjects"),
    "TriggerAuthentication": _list_keda("triggerauthentications"),
}


def discover_locations(
    run_clusters: list[ClusterInfo],
    resources: dict[str, Resource],
) -> tuple[dict[str, ClusterInfo], dict[ClusterInfo, ClusterFailure]]:
    """Route each k8s resource to the cluster that currently hosts it.

    :param run_clusters: Clusters the run registered (``RunInfo.CLUSTERS``).
    :param resources: All registered resources for the run, keyed by
        resource id. Non-k8s resources are ignored; SQS routing is by
        :attr:`Resource.region` in the orchestrator.
    :returns: ``(location, cluster_failures)`` where ``location`` maps
        resource id to the cluster hosting it (resources absent from every
        cluster are simply omitted; callers treat them as
        :class:`DeleteStatus.NOT_FOUND`), and ``cluster_failures`` maps
        each unreachable cluster to a :class:`ClusterFailure`.
    """
    by_type: dict[str, set[str]] = defaultdict(set)
    for resource in resources.values():
        if resource.type in K8S_DELETERS:
            by_type[resource.type].add(resource.name)

    location: dict[str, ClusterInfo] = {}
    cluster_failures: dict[ClusterInfo, ClusterFailure] = {}
    if not by_type:
        return location, cluster_failures

    presence: dict[tuple[ClusterInfo, str], set[str]] = {}

    for cluster in run_clusters:
        failure = _populate_presence_for_cluster(cluster, by_type, presence)
        if failure is not None:
            cluster_failures[cluster] = failure

    for resource_id, resource in resources.items():
        if resource.type not in K8S_DELETERS:
            continue
        for cluster in run_clusters:
            names = presence.get((cluster, resource.type))
            if names is not None and resource.name in names:
                location[resource_id] = cluster
                break

    return location, cluster_failures


def _populate_presence_for_cluster(
    cluster: ClusterInfo,
    by_type: dict[str, set[str]],
    presence: dict[tuple[ClusterInfo, str], set[str]],
) -> ClusterFailure | None:
    try:
        core, apps, batch = get_cluster_clients(cluster)
    except GoogleAPICallError as exc:
        reset_cluster_clients(cluster)
        return _classify_cluster_failure(exc)

    for rtype in by_type:
        list_fn = _LIST_FNS[rtype]
        try:
            names = retried(functools.partial(list_fn, core, apps, batch))
        except (k8s_client.ApiException, GoogleAPICallError) as exc:
            reset_cluster_clients(cluster)
            return _classify_cluster_failure(exc)
        presence[(cluster, rtype)] = names
    return None


def _classify_cluster_failure(exc: BaseException) -> ClusterFailure:
    if isinstance(exc, GoogleAPICallError) and getattr(exc, "code", None) == 404:
        return ClusterFailure("cluster_404", "cluster not found")
    detail = getattr(exc, "reason", None) or str(exc)
    return ClusterFailure("cluster_auth", str(detail))
