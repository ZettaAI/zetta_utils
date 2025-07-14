"""
Helpers for k8s job.
"""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional

from kubernetes.client.exceptions import ApiException

from kubernetes import client as k8s_client
from kubernetes import watch  # type: ignore
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


def get_job_spec(
    pod_spec: k8s_client.V1PodSpec,
    meta: Optional[k8s_client.V1ObjectMeta] = None,
    active_deadline_seconds: Optional[int] = None,
    backoff_limit: Optional[int] = 1,
    pod_failure_policy: Optional[k8s_client.V1PodFailurePolicy] = None,
    selector: Optional[k8s_client.V1LabelSelector] = None,
    suspend: Optional[bool] = False,
) -> k8s_client.V1JobSpec:
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
    name = f"run-{name}"
    meta = k8s_client.V1ObjectMeta(name=name, labels=labels)
    job_spec = get_job_spec(
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
    name = f"run-{name}"
    meta = k8s_client.V1ObjectMeta(name=name, labels=labels)
    job_spec = get_job_spec(
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
        time.sleep(15)
    logger.info(f"`{job.metadata.name}` job started.")


def follow_job_logs(
    job: k8s_client.V1Job,
    cluster_info: ClusterInfo,
    namespace: str = "default",
    tail_lines: Optional[int] = None,
    wait_until_start: Optional[bool] = True,
):
    batch_v1_api = _reset_batch_api(cluster_info)
    if wait_until_start:
        _wait_for_job_start(job, namespace, batch_v1_api)

    try:
        core_api = k8s_client.CoreV1Api()
        podlist = core_api.list_namespaced_pod(
            namespace=namespace, label_selector=f"job-name={job.metadata.name}"
        )
        job_name = podlist.items[0].metadata.name
        log_stream = watch.Watch().stream(
            core_api.read_namespaced_pod_log,
            name=job_name,
            namespace=namespace,
            tail_lines=tail_lines,
        )
        if tail_lines is None:
            for output in log_stream:
                logger.info(output)
        else:
            result = []
            for output in log_stream:
                result.append(output)
                if len(result) == tail_lines:
                    logger.info("\n".join(result))
                    result = []
            logger.info("\n".join(result))
    except ApiException:
        # resets credential config after timeout
        follow_job_logs(job, cluster_info, namespace, tail_lines, wait_until_start)


def get_job_pod(
    job: k8s_client.V1Job,
    cluster_info: ClusterInfo,
    namespace: str = "default",
    wait_until_start: Optional[bool] = True,
) -> k8s_client.V1Pod:
    batch_v1_api = _reset_batch_api(cluster_info)
    if wait_until_start:
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

    while True:
        logger.info(f"Waiting for `{job.metadata.name}` to complete.")
        time.sleep(15)
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

        if job.status.succeeded == 1:
            break

        if job.status.failed:
            if job.status.failed > job.spec.backoff_limit:
                logger.warning(f"`{job.metadata.name}` backoff_limit reached.")
                break
            pod = get_job_pod(job, cluster_info, wait_until_start=False)
            follow_job_logs(job, cluster_info, tail_lines=64, wait_until_start=False)
            logger.warning(
                f"Retrying job `{job.metadata.name}`: {job.status.failed}/{job.spec.backoff_limit}"
            )
            _wait_for_job_start(job, namespace, batch_v1_api)

    logger.info(f"`{job.metadata.name}` job completed.")
    pod = get_job_pod(job, cluster_info, wait_until_start=False)
    logger.info(f"job pod phase: {pod.status.phase}")
    follow_job_logs(job, cluster_info, tail_lines=64, wait_until_start=False)


@contextmanager
def job_ctx_manager(
    run_id: str,
    cluster_info: ClusterInfo,
    job: k8s_client.V1Job,
    secrets: List[k8s_client.V1Secret],
    namespace: Optional[str] = "default",
):
    batch_v1_api = _reset_batch_api(cluster_info)
    with secrets_ctx_mngr(run_id, secrets, cluster_info):
        logger.info(f"Creating k8s job `{job.metadata.name}`")
        batch_v1_api.create_namespaced_job(body=job, namespace=namespace)
        _id = register_resource(
            Resource(
                run_id,
                ResourceTypes.K8S_JOB.value,
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
            deregister_resource(_id)
