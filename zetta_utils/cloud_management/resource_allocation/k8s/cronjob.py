"""
Kubernetes cronjob.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import attrs

from kubernetes import client as k8s_client  # type: ignore
from zetta_utils import builder, log

from .common import ClusterInfo, get_cluster_data
from .job import get_job_template
from .pod import get_pod_spec

logger = log.get_logger("zetta_utils")


def _get_cronjob(
    name: str,
    image: str,
    command: List[str],
    command_args: List[str],
    envs: List[k8s_client.V1EnvVar],
    resource_limits: Dict[str, int | float | str],
    spec_config: CronJobSpecConfig,
    labels: Optional[Dict[str, str]] = None,
    tolerations: Optional[k8s_client.V1Toleration] = None,
    resource_requests: Optional[Dict[str, int | float | str]] = None,
) -> k8s_client.V1CronJob:
    tolerations = tolerations or []
    pod_spec = get_pod_spec(
        name=name,
        image=image,
        command=command,
        command_args=command_args,
        envs=envs,
        resources=resource_limits,
        restart_policy="OnFailure",
        tolerations=tolerations,
        resource_requests=resource_requests,
    )

    job_template = get_job_template(
        name=name,
        pod_spec=pod_spec,
        labels=labels,
    )

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

    return k8s_client.V1CronJob(metadata=job_template.metadata, spec=cronjob_spec)


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
    preset_env_vars: Optional[List[str]] = None,
    spec_config: CronJobSpecConfig = CronJobSpecConfig(),
    labels: Optional[Dict[str, str]] = None,
    patch: Optional[bool] = False,
    resource_requests: Optional[Dict[str, int | float | str]] = None,
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

    if preset_env_vars is None:
        preset_env_vars = []

    for env_var in preset_env_vars:
        env = k8s_client.V1EnvVar(name=env_var, value=os.environ[env_var])
        envs.append(env)

    cronjob = _get_cronjob(
        name=name,
        image=image,
        command=command,
        command_args=command_args,
        envs=envs,
        resource_limits=resources,
        spec_config=spec_config,
        labels=labels,
        resource_requests=resource_requests,
    )
    if patch:
        batch_v1_api.patch_namespaced_cron_job(name=name, namespace=namespace, body=cronjob)
    else:
        batch_v1_api.create_namespaced_cron_job(namespace=namespace, body=cronjob)
