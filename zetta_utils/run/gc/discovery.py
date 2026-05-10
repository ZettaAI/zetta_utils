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
from zetta_utils.run.gc.utils import retry_transient_api
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


_ListFn = Callable[[k8s_client.CoreV1Api, k8s_client.AppsV1Api, k8s_client.BatchV1Api], Any]

_LIST_FNS: dict[str, _ListFn] = {
    ResourceTypes.K8S_CONFIGMAP.value: lambda core, apps, batch: core.list_namespaced_config_map(
        namespace="default"
    ),
    ResourceTypes.K8S_DEPLOYMENT.value: lambda core, apps, batch: apps.list_namespaced_deployment(
        namespace="default"
    ),
    ResourceTypes.K8S_JOB.value: lambda core, apps, batch: batch.list_namespaced_job(
        namespace="default"
    ),
    ResourceTypes.K8S_SECRET.value: lambda core, apps, batch: core.list_namespaced_secret(
        namespace="default"
    ),
    ResourceTypes.K8S_SERVICE.value: lambda core, apps, batch: core.list_namespaced_service(
        namespace="default"
    ),
    ResourceTypes.K8S_STATEFULSET.value: (
        lambda core, apps, batch: apps.list_namespaced_stateful_set(namespace="default")
    ),
}


@retry_transient_api
def _retried_list(call: Callable[[], Any]) -> Any:
    return call()


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
            items = _retried_list(functools.partial(list_fn, core, apps, batch))
        except (k8s_client.ApiException, GoogleAPICallError) as exc:
            reset_cluster_clients(cluster)
            return _classify_cluster_failure(exc)
        presence[(cluster, rtype)] = {item.metadata.name for item in items.items}
    return None


def _classify_cluster_failure(exc: BaseException) -> ClusterFailure:
    if isinstance(exc, GoogleAPICallError) and getattr(exc, "code", None) == 404:
        return ClusterFailure("cluster_404", "cluster not found")
    detail = getattr(exc, "reason", None) or str(exc)
    return ClusterFailure("cluster_auth", str(detail))
