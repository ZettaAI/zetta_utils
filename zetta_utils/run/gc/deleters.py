"""Per-resource-type delete dispatchers for the run garbage collector.

Each deleter takes a :class:`Resource` and a context bag holding the
already-built api clients, runs the provider's delete call (wrapped in the
project-wide transient-API retry policy from :mod:`utils`), and returns a
:class:`DeleteOutcome` describing what happened. Deleters do not log or
deregister; the orchestrator collects outcomes and drives both.

Per-cluster k8s api clients (:func:`get_cluster_clients`) are built from
``get_cluster_data`` (programmatic ADC) and each cluster gets its own
:class:`kubernetes.client.ApiClient`, so we never call
``Configuration.set_default`` and there is no process-global mutation.

Per-region SQS clients are cached the same way.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

import attrs
import kubernetes.client as k8s_client
import taskqueue
from boto3.exceptions import Boto3Error

from zetta_utils.cloud_management.resource_allocation.k8s.common import (
    ClusterInfo,
    get_cluster_data,
)
from zetta_utils.message_queues.sqs import utils as sqs_utils
from zetta_utils.run.gc.utils import retried
from zetta_utils.run.resource import Resource, ResourceTypes


class DeleteStatus(Enum):
    DELETED = "deleted"
    NOT_FOUND = "404"  # provider said 404; we don't assert prior existence
    FAILED = "failed"


@attrs.frozen
class DeleteOutcome:
    """Result of a single resource delete attempt.

    :param status: Terminal status; see :class:`DeleteStatus`.
    :param error: Free-form one-line error for ``FAILED``; empty otherwise.
    :param error_class: Short category for ``FAILED`` consumed by the
        orchestrator's per-run aggregator (``"k8s_auth"``, ``"k8s_5xx"``,
        ``"k8s_other"``, ``"sqs"``). Empty when not ``FAILED``.
    """

    status: DeleteStatus
    error: str = ""
    error_class: str = ""


@attrs.frozen
class K8sDeleteContext:
    """API clients for k8s resource deleters; bound to one cluster."""

    core_v1: k8s_client.CoreV1Api
    apps_v1: k8s_client.AppsV1Api
    batch_v1: k8s_client.BatchV1Api


@attrs.frozen
class SqsDeleteContext:
    """SQS client for queue deleters; bound to one region."""

    sqs_client: Any


K8sDeleter = Callable[[Resource, K8sDeleteContext], DeleteOutcome]
SqsDeleter = Callable[[Resource, SqsDeleteContext], DeleteOutcome]


# --- per-cluster k8s client cache ---

_CLUSTER_CLIENTS: dict[
    ClusterInfo, tuple[k8s_client.CoreV1Api, k8s_client.AppsV1Api, k8s_client.BatchV1Api]
] = {}


def get_cluster_clients(
    cluster_info: ClusterInfo,
) -> tuple[k8s_client.CoreV1Api, k8s_client.AppsV1Api, k8s_client.BatchV1Api]:
    """Return per-cluster cached api clients.

    Each tuple shares one :class:`kubernetes.client.ApiClient` configured via
    :func:`get_cluster_data` (programmatic ADC), so clusters never mutate
    process-global state. Reset on failure via :func:`reset_cluster_clients`.
    """
    if cluster_info not in _CLUSTER_CLIENTS:
        configuration, _ = get_cluster_data(cluster_info)
        api_client = k8s_client.ApiClient(configuration=configuration)
        _CLUSTER_CLIENTS[cluster_info] = (
            k8s_client.CoreV1Api(api_client=api_client),
            k8s_client.AppsV1Api(api_client=api_client),
            k8s_client.BatchV1Api(api_client=api_client),
        )
    return _CLUSTER_CLIENTS[cluster_info]


def reset_cluster_clients(cluster_info: ClusterInfo) -> None:
    _CLUSTER_CLIENTS.pop(cluster_info, None)


def build_k8s_context(cluster_info: ClusterInfo) -> K8sDeleteContext:
    core, apps, batch = get_cluster_clients(cluster_info)
    return K8sDeleteContext(core_v1=core, apps_v1=apps, batch_v1=batch)


# --- per-region SQS client cache ---

_SQS_CLIENTS: dict[str, Any] = {}


def get_sqs_client_for(region: str) -> Any:
    """Return a cached boto3 SQS client for ``region``.

    Empty / falsy region falls back to ``taskqueue.secrets.AWS_DEFAULT_REGION``.
    """
    effective_region = region or taskqueue.secrets.AWS_DEFAULT_REGION
    if effective_region not in _SQS_CLIENTS:
        _SQS_CLIENTS[effective_region] = sqs_utils.get_sqs_client(region_name=effective_region)
    return _SQS_CLIENTS[effective_region]


def build_sqs_context(region: str) -> SqsDeleteContext:
    return SqsDeleteContext(sqs_client=get_sqs_client_for(region))


# --- k8s deleters ---


def _classify_k8s_exception(exc: k8s_client.ApiException) -> DeleteOutcome:
    detail = str(exc.reason or exc)
    if exc.status == 404:
        return DeleteOutcome(DeleteStatus.NOT_FOUND)
    if exc.status in (401, 403):
        return DeleteOutcome(DeleteStatus.FAILED, error=detail, error_class="k8s_auth")
    if exc.status is not None and (exc.status >= 500 or exc.status == 429):
        return DeleteOutcome(DeleteStatus.FAILED, error=detail, error_class="k8s_5xx")
    return DeleteOutcome(DeleteStatus.FAILED, error=detail, error_class="k8s_other")


def _run_k8s_delete(call: Callable[[], None]) -> DeleteOutcome:
    try:
        retried(call)
    except k8s_client.ApiException as exc:
        return _classify_k8s_exception(exc)
    return DeleteOutcome(DeleteStatus.DELETED)


def _delete_configmap(resource: Resource, ctx: K8sDeleteContext) -> DeleteOutcome:
    return _run_k8s_delete(
        lambda: ctx.core_v1.delete_namespaced_config_map(name=resource.name, namespace="default")
    )


def _delete_deployment(resource: Resource, ctx: K8sDeleteContext) -> DeleteOutcome:
    return _run_k8s_delete(
        lambda: ctx.apps_v1.delete_namespaced_deployment(name=resource.name, namespace="default")
    )


def _delete_job(resource: Resource, ctx: K8sDeleteContext) -> DeleteOutcome:
    return _run_k8s_delete(
        lambda: ctx.batch_v1.delete_namespaced_job(
            name=resource.name, namespace="default", propagation_policy="Foreground"
        )
    )


def _delete_secret(resource: Resource, ctx: K8sDeleteContext) -> DeleteOutcome:
    return _run_k8s_delete(
        lambda: ctx.core_v1.delete_namespaced_secret(name=resource.name, namespace="default")
    )


def _delete_service(resource: Resource, ctx: K8sDeleteContext) -> DeleteOutcome:
    return _run_k8s_delete(
        lambda: ctx.core_v1.delete_namespaced_service(name=resource.name, namespace="default")
    )


def _delete_statefulset(resource: Resource, ctx: K8sDeleteContext) -> DeleteOutcome:
    return _run_k8s_delete(
        lambda: ctx.apps_v1.delete_namespaced_stateful_set(name=resource.name, namespace="default")
    )


def _delete_keda(plural: str) -> K8sDeleter:
    """Build a deleter that targets a KEDA CRD via ``CustomObjectsApi``.

    The custom client reuses the per-cluster ``ApiClient`` cached by
    :func:`get_cluster_clients`, so credentials are not re-fetched.
    """

    def deleter(resource: Resource, ctx: K8sDeleteContext) -> DeleteOutcome:
        custom = k8s_client.CustomObjectsApi(api_client=ctx.core_v1.api_client)
        return _run_k8s_delete(
            lambda: custom.delete_namespaced_custom_object(
                group="keda.sh",
                version="v1alpha1",
                namespace="default",
                plural=plural,
                name=resource.name,
            )
        )

    return deleter


K8S_DELETERS: dict[str, K8sDeleter] = {
    ResourceTypes.K8S_CONFIGMAP.value: _delete_configmap,
    ResourceTypes.K8S_DEPLOYMENT.value: _delete_deployment,
    ResourceTypes.K8S_JOB.value: _delete_job,
    ResourceTypes.K8S_SECRET.value: _delete_secret,
    ResourceTypes.K8S_SERVICE.value: _delete_service,
    ResourceTypes.K8S_STATEFULSET.value: _delete_statefulset,
    # --- transient legacy KEDA cleanup; remove once orphan RESOURCE_DB rows are drained ---
    "ScaledJob": _delete_keda("scaledjobs"),
    "ScaledObject": _delete_keda("scaledobjects"),
    "TriggerAuthentication": _delete_keda("triggerauthentications"),
}


# --- SQS deleter ---


def _delete_sqs_queue(resource: Resource, ctx: SqsDeleteContext) -> DeleteOutcome:
    sqs = ctx.sqs_client
    try:
        queue_url = retried(lambda: sqs.get_queue_url(QueueName=resource.name)["QueueUrl"])
        retried(lambda: sqs.delete_queue(QueueUrl=queue_url))
    except sqs.exceptions.QueueDoesNotExist:
        return DeleteOutcome(DeleteStatus.NOT_FOUND)
    except Boto3Error as exc:
        return DeleteOutcome(DeleteStatus.FAILED, error=str(exc), error_class="sqs")
    return DeleteOutcome(DeleteStatus.DELETED)


SQS_DELETERS: dict[str, SqsDeleter] = {
    ResourceTypes.SQS_QUEUE.value: _delete_sqs_queue,
}
