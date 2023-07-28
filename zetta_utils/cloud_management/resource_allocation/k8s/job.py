"""
Helpers for k8s job.
"""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional

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
    selector: Optional[k8s_client.V1LabelSelector] = None,
    suspend: Optional[bool] = False,
):
    pod_template = k8s_client.V1PodTemplateSpec(metadata=meta, spec=pod_spec)
    return k8s_client.V1JobSpec(
        active_deadline_seconds=active_deadline_seconds,
        backoff_limit=backoff_limit,
        selector=selector,
        suspend=suspend,
        template=pod_template,
    )


def get_job_template(
    name: str,
    pod_spec: k8s_client.V1PodSpec,
    active_deadline_seconds: Optional[int] = None,
    backoff_limit: Optional[int] = 3,
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
        selector=selector,
        suspend=suspend,
    )
    return k8s_client.V1JobTemplateSpec(metadata=meta, spec=job_spec)


def get_job(
    name: str,
    pod_spec: k8s_client.V1PodSpec,
    active_deadline_seconds: Optional[int] = None,
    backoff_limit: Optional[int] = 3,
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
        selector=selector,
        suspend=suspend,
    )
    return k8s_client.V1Job(metadata=meta, spec=job_spec)


@contextmanager
def job_ctx_manager(
    execution_id: str,
    cluster_info: ClusterInfo,
    job: k8s_client.V1Job,
    secrets: List[k8s_client.V1Secret],
    namespace: Optional[str] = "default",
):
    configuration, _ = get_cluster_data(cluster_info)
    k8s_client.Configuration.set_default(configuration)

    batch_v1_api = k8s_client.BatchV1Api()
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
            while True:
                job = batch_v1_api.read_namespaced_job_status(
                    name=job.metadata.name,
                    namespace=namespace,
                )
                logger.info(f"`{job.metadata.name}` job status: {job.status.ready}")
                if job.status.ready == 1:
                    break
                time.sleep(5)

            core_api = k8s_client.CoreV1Api()
            pods = core_api.list_namespaced_pod(
                namespace=namespace, label_selector=f"job-name={job.metadata.name}"
            )
            job_pod = pods.items[0]
            log_stream = watch.Watch().stream(
                core_api.read_namespaced_pod_log,
                name=job_pod.metadata.name,
                namespace=namespace,
            )
            for output in log_stream:
                logger.info(output)
        finally:
            # new configuration to refresh expired tokens (long running executions)
            configuration, _ = get_cluster_data(cluster_info)
            k8s_client.Configuration.set_default(configuration)

            # need to create a new client for the above to take effect
            batch_v1_api = k8s_client.BatchV1Api()
            logger.info(f"Deleting k8s job `{job.metadata.name}`")
            batch_v1_api.delete_namespaced_job(
                name=job.metadata.name,
                namespace=namespace,
                propagation_policy="Foreground",
            )
