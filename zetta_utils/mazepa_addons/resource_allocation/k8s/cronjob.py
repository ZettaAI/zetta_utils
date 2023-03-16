"""
Kubernetes cronjob.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import attrs

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import builder, log

from .common import ClusterInfo, get_cluster_data

logger = log.get_logger("zetta_utils")


def _get_cronjob(
    name: str,
    namespace: str,
    image: str,
    command: List[str],
    command_args: List[str],
    envs: List[k8s_client.V1EnvVar],
    resources: Dict[str, int | float | str],
    spec_config: CronJobSpecConfig,
    labels: Optional[Dict[str, str]] = None,
) -> k8s_client.V1CronJob:

    container = k8s_client.V1Container(
        command=command,
        args=command_args,
        env=envs,
        name=name,
        image=image,
        image_pull_policy="IfNotPresent",
        resources=k8s_client.V1ResourceRequirements(
            requests=resources,
            limits=resources,
        ),
        termination_message_path="/dev/termination-log",
        termination_message_policy="File",
        volume_mounts=[],
    )

    schedule_toleration = k8s_client.V1Toleration(
        key="worker-pool", operator="Equal", value="true", effect="NoSchedule"
    )

    pod_spec = k8s_client.V1PodSpec(
        containers=[container],
        dns_policy="Default",
        restart_policy="OnFailure",
        scheduler_name="default-scheduler",
        security_context={},
        termination_grace_period_seconds=30,
        tolerations=[schedule_toleration],
    )

    common_meta = k8s_client.V1ObjectMeta(name=name, namespace=namespace, labels=labels)

    pod_template = k8s_client.V1PodTemplateSpec(metadata=common_meta, spec=pod_spec)
    job_spec = k8s_client.V1JobSpec(template=pod_template)

    job_template = k8s_client.V1JobTemplateSpec(metadata=common_meta, spec=job_spec)
    cronjob_spec = k8s_client.V1CronJobSpec(
        concurrency_policy=spec_config.concurrency_policy,
        failed_jobs_history_limit=spec_config.failed_jobs_history_limit,
        job_template=job_template,
        schedule=spec_config.schedule,
        starting_deadline_seconds=spec_config.starting_deadline_seconds,
        successful_jobs_history_limit=spec_config.successful_jobs_history_limit,
        suspend=spec_config.suspend,
        time_zone=spec_config.time_zone,
    )

    return k8s_client.V1CronJob(metadata=common_meta, spec=cronjob_spec)


@builder.register("mazepa.k8s.CronJobSpec")
@attrs.frozen
class CronJobSpecConfig:
    concurrency_policy: Optional[str] = "Forbid"
    failed_jobs_history_limit: Optional[int] = 1
    schedule: Optional[str] = "@hourly"
    starting_deadline_seconds: Optional[int] = None
    successful_jobs_history_limit: Optional[int] = 1
    suspend: Optional[bool] = False
    time_zone: Optional[str] = None


@builder.register("mazepa.k8s.configure_cronjob")
def configure_cronjob(
    cluster: ClusterInfo,
    name: str,
    namespace: str,
    image: str,
    command: List[str],
    command_args: List[str],
    env_vars: Dict[str, str],
    resources: Dict[str, int | float | str],
    spec_config: CronJobSpecConfig = CronJobSpecConfig(),
    labels: Optional[Dict[str, str]] = None,
    patch: Optional[bool] = False,
):
    """
    Create a cronjob or patch/update an existing one.
    """
    configuration, _ = get_cluster_data(cluster)
    k8s_client.Configuration.set_default(configuration)
    batch_v1_api = k8s_client.BatchV1Api()

    envs = []
    for key, val in env_vars.items():
        envs.append(k8s_client.V1EnvVar(name=key, value=val))

    cronjob = _get_cronjob(
        name=name,
        namespace=namespace,
        image=image,
        command=command,
        command_args=command_args,
        envs=envs,
        resources=resources,
        spec_config=spec_config,
        labels=labels,
    )
    if patch:
        batch_v1_api.patch_namespaced_cron_job(name=name, namespace=namespace, body=cronjob)
    else:
        batch_v1_api.create_namespaced_cron_job(namespace=namespace, body=cronjob)
