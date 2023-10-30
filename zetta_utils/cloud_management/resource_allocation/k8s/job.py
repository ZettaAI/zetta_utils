"""
Helpers for k8s job.
"""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional

from kubernetes.client.exceptions import ApiException

from kubernetes import client as k8s_client  # type: ignore
from kubernetes import watch  # type: ignore
from zetta_utils import log

from ..resource_tracker import (
    ExecutionResource,
    ExecutionResourceTypes,
    register_execution_resource,
)
from .common import ClusterInfo, get_cluster_data
from .secret import secrets_ctx_mngr

logger = log.get_logger("zetta_utils")


def _get_job_spec(
    pod_spec: k8s_client.V1PodSpec,
    meta: k8s_client.V1ObjectMeta,
    active_deadline_seconds: Optional[int] = None,
    backoff_limit: Optional[int] = 3,
    pod_failure_policy: Optional[k8s_client.V1PodFailurePolicy] = None,
    selector: Optional[k8s_client.V1LabelSelector] = None,
    suspend: Optional[bool] = False,
):
    pod_template = k8s_client.V1PodTemplateSpec(metadata=meta, spec=pod_spec)
    return k8s_client.V1JobSpec(
        active_deadline_seconds=active_deadline_seconds,
        backoff_limit=backoff_limit,
        pod_failure_policy=pod_failure_policy,
        selector=selector,
        suspend=suspend,
        template=pod_template,
    )


def _reset_batch_api(cluster_info: ClusterInfo):
    configuration, _ = get_cluster_data(cluster_info)
    k8s_client.Configuration.set_default(configuration)
    return k8s_client.BatchV1Api()


def get_job_template(
    name: str,
    pod_spec: k8s_client.V1PodSpec,
    active_deadline_seconds: Optional[int] = None,
    backoff_limit: Optional[int] = 3,
    pod_failure_policy: Optional[k8s_client.V1PodFailurePolicy] = None,
    labels: Optional[Dict[str, str]] = None,
    selector: Optional[k8s_client.V1LabelSelector] = None,
    suspend: Optional[bool] = False,
) -> k8s_client.V1JobTemplateSpec:
    meta = k8s_client.V1ObjectMeta(name=name, labels=labels)
    job_spec = _get_job_spec(
        pod_spec=pod_spec,
        meta=meta,
        active_deadline_seconds=active_deadline_seconds,
        backoff_limit=backoff_limit,
        pod_failure_policy=pod_failure_policy,
        selector=selector,
        suspend=suspend,
    )
    return k8s_client.V1JobTemplateSpec(metadata=meta, spec=job_spec)


def get_job(
    name: str,
    pod_spec: k8s_client.V1PodSpec,
    active_deadline_seconds: Optional[int] = None,
    backoff_limit: Optional[int] = 3,
    pod_failure_policy: Optional[k8s_client.V1PodFailurePolicy] = None,
    labels: Optional[Dict[str, str]] = None,
    selector: Optional[k8s_client.V1LabelSelector] = None,
    suspend: Optional[bool] = False,
) -> k8s_client.V1Job:
    meta = k8s_client.V1ObjectMeta(name=name, labels=labels)
    job_spec = _get_job_spec(
        pod_spec=pod_spec,
        meta=meta,
        active_deadline_seconds=active_deadline_seconds,
        backoff_limit=backoff_limit,
        pod_failure_policy=pod_failure_policy,
        selector=selector,
        suspend=suspend,
    )
    return k8s_client.V1Job(metadata=meta, spec=job_spec)


def _wait_for_job_start(
    job: k8s_client.V1Job,
    namespace: str,
    batch_v1_api: k8s_client.BatchV1Api,
) -> None:
    while True:
        job = batch_v1_api.read_namespaced_job_status(
            name=job.metadata.name,
            namespace=namespace,
        )
        logger.info(f"Waiting for `{job.metadata.name}` to start.")
        if job.status.ready == 1:
            break
        time.sleep(5)
    logger.info(f"`{job.metadata.name}` job started.")


def follow_job_logs(
    job: k8s_client.V1Job,
    cluster_info: ClusterInfo,
    namespace: str = "default",
):
    batch_v1_api = _reset_batch_api(cluster_info)
    _wait_for_job_start(job, namespace, batch_v1_api)

    core_api = k8s_client.CoreV1Api()
    podlist = core_api.list_namespaced_pod(
        namespace=namespace, label_selector=f"job-name={job.metadata.name}"
    )
    job_name = podlist.items[0].metadata.name
    log_stream = watch.Watch().stream(
        core_api.read_namespaced_pod_log,
        name=job_name,
        namespace=namespace,
    )
    for output in log_stream:
        logger.info(output)


def get_job_pod(
    job: k8s_client.V1Job,
    cluster_info: ClusterInfo,
    namespace: str = "default",
) -> k8s_client.V1Pod:
    batch_v1_api = _reset_batch_api(cluster_info)
    _wait_for_job_start(job, namespace, batch_v1_api)

    core_api = k8s_client.CoreV1Api()
    podlist = core_api.list_namespaced_pod(
        namespace=namespace, label_selector=f"job-name={job.metadata.name}"
    )
    return podlist.items[0]


def wait_for_job_completion(
    job: k8s_client.V1Job,
    cluster_info: ClusterInfo,
    namespace: str = "default",
):
    batch_v1_api = _reset_batch_api(cluster_info)
    _wait_for_job_start(job, namespace, batch_v1_api)

    job = batch_v1_api.read_namespaced_job_status(
        name=job.metadata.name,
        namespace=namespace,
    )
    not_done = job.status.succeeded == 0 or job.status.succeeded is None
    while not_done:
        logger.info(f"Waiting for `{job.metadata.name}` to complete.")
        time.sleep(5)
        try:
            job = batch_v1_api.read_namespaced_job_status(
                name=job.metadata.name,
                namespace=namespace,
            )
        except ApiException:
            batch_v1_api = _reset_batch_api(cluster_info)
            job = batch_v1_api.read_namespaced_job_status(
                name=job.metadata.name,
                namespace=namespace,
            )

        not_done = job.status.succeeded == 0 or job.status.succeeded is None
    logger.info(f"`{job.metadata.name}` job completed.")


@contextmanager
def job_ctx_manager(
    execution_id: str,
    cluster_info: ClusterInfo,
    job: k8s_client.V1Job,
    secrets: List[k8s_client.V1Secret],
    namespace: Optional[str] = "default",
):
    batch_v1_api = _reset_batch_api(cluster_info)
    with secrets_ctx_mngr(execution_id, secrets, cluster_info):
        logger.info(f"Creating k8s job `{job.metadata.name}`")
        batch_v1_api.create_namespaced_job(body=job, namespace=namespace)
        register_execution_resource(
            ExecutionResource(
                execution_id,
                ExecutionResourceTypes.K8S_JOB.value,
                job.metadata.name,
            )
        )

        try:
            yield
        finally:
            # new configuration to refresh expired tokens (long running executions)
            batch_v1_api = _reset_batch_api(cluster_info)
            logger.info(f"Deleting k8s job `{job.metadata.name}`")
            batch_v1_api.delete_namespaced_job(
                name=job.metadata.name,
                namespace=namespace,
                propagation_policy="Foreground",
            )
