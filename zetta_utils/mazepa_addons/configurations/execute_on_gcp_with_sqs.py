# pylint: disable=too-many-locals
from __future__ import annotations

import copy
import os
from contextlib import AbstractContextManager, ExitStack
from typing import Any, Final, Iterable, Literal, Optional, TypedDict, Union

import attrs
from typeguard import typechecked
from typing_extensions import NotRequired

from zetta_utils import builder, log, mazepa, run
from zetta_utils.cloud_management.resource_allocation import aws_sqs, gcloud, k8s
from zetta_utils.mazepa import SemaphoreType, execute
from zetta_utils.mazepa.task_outcome import OutcomeReport
from zetta_utils.mazepa.task_router import TaskRouter
from zetta_utils.mazepa.tasks import Task
from zetta_utils.message_queues import sqs  # pylint: disable=unused-import
from zetta_utils.message_queues.base import PullMessageQueue, PushMessageQueue

from .execute_locally import execute_locally

logger = log.get_logger("zetta_utils")

REQUIRED_ENV_VARS: Final = [
    # "GRAFANA_CLOUD_ACCESS_KEY",
    "ZETTA_USER",
    "ZETTA_PROJECT",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
]


DEFAULT_GCP_CLUSTER_NAME: Final = "zutils-x3"
DEFAULT_GCP_CLUSTER_REGION: Final = "us-east1"
DEFAULT_GCP_CLUSTER_PROJECT: Final = "zetta-research"

DEFAULT_GCP_CLUSTER: Final = k8s.ClusterInfo(
    name=DEFAULT_GCP_CLUSTER_NAME,
    region=DEFAULT_GCP_CLUSTER_REGION,
    project=DEFAULT_GCP_CLUSTER_PROJECT,
)


def _ensure_required_env_vars():
    missing_vars = set()
    for e in REQUIRED_ENV_VARS:
        if e not in os.environ:
            missing_vars.add(e)

    if len(missing_vars) != 0:
        raise RuntimeError(
            f"Missing the following required environment variables: {missing_vars}. "
            f"It is recommended to put these variables into your `~/.bashrc`/`~/.zshrc`"
        )


@attrs.frozen
class WorkerGroup:
    replicas: int
    resource_limits: dict[str, int | float | str]
    max_replicas: int = 0
    queue_tags: list[str] | None = None
    num_procs: int = 1
    sqs_based_scaling: bool = True
    resource_requests: dict[str, int | float | str] | None = None
    semaphores_spec: dict[SemaphoreType, int] | None = None
    provisioning_model: Literal["standard", "spot"] = "spot"
    idle_worker_timeout: int = 300
    labels: dict[str, str] | None = None
    gpu_accelerator_type: str | None = None


class WorkerGroupDict(TypedDict, total=False):
    replicas: int
    resource_limits: dict[str, int | float | str]
    max_replicas: NotRequired[int]
    queue_tags: NotRequired[list[str]]
    num_procs: NotRequired[int]
    sqs_based_scaling: NotRequired[bool]
    resource_requests: NotRequired[dict[str, int | float | str]]
    semaphores_spec: NotRequired[dict[SemaphoreType, int]]
    provisioning_model: NotRequired[Literal["standard", "spot"]]
    idle_worker_timeout: NotRequired[int]
    labels: NotRequired[dict[str, str]]
    gpu_accelerator_type: NotRequired[str]


def _get_group_taskqueue_and_contexts(
    execution_id: str,
    image: str,
    group: WorkerGroup,
    group_name: str,
    cluster: k8s.ClusterInfo,
    sqs_trigger_name: str,
    outcome_queue_spec: dict[str, Any],
    env_secret_mapping: dict[str, str],
    adc_available: bool = False,
) -> tuple[PushMessageQueue[Task], list[AbstractContextManager]]:
    ctx_managers: list[AbstractContextManager] = []

    work_queue_name = f"run-{execution_id}"
    if group.queue_tags is not None:
        work_queue_name += f"_{'_'.join(group.queue_tags)}"
    work_queue_name += "_work"
    task_queue_spec = {"@type": "SQSQueue", "name": work_queue_name}
    task_queue = builder.build(task_queue_spec)
    ctx_managers.append(aws_sqs.sqs_queue_ctx_mngr(execution_id, task_queue))

    if group.sqs_based_scaling:
        worker_command = k8s.get_mazepa_worker_command(
            task_queue_spec,
            outcome_queue_spec,
            group.num_procs,
            group.semaphores_spec,
            idle_timeout=group.idle_worker_timeout,
        )
        pod_spec = k8s.get_mazepa_pod_spec(
            image=image,
            command=worker_command,
            resources=group.resource_limits,
            env_secret_mapping=env_secret_mapping,
            provisioning_model=group.provisioning_model,
            resource_requests=group.resource_requests,
            restart_policy="Never",
            gpu_accelerator_type=group.gpu_accelerator_type,
            adc_available=adc_available,
        )
        job_spec = k8s.get_job_spec(pod_spec=pod_spec)
        scaled_job_ctx_mngr = k8s.scaled_job_ctx_mngr(
            execution_id,
            group_name=group_name,
            cluster_info=cluster,
            job_spec=job_spec,
            secrets=[],
            sqs_trigger_name=sqs_trigger_name,
            replicas=group.replicas,
            max_replicas=group.max_replicas,
            queue=task_queue,
        )
        ctx_managers.append(scaled_job_ctx_mngr)
    else:
        deployment = k8s.get_mazepa_worker_deployment(
            f"{execution_id}-{group_name}",
            image=image,
            task_queue_spec=task_queue_spec,
            outcome_queue_spec=outcome_queue_spec,
            replicas=group.replicas,
            resources=group.resource_limits,
            env_secret_mapping=env_secret_mapping,
            labels=group.labels,
            resource_requests=group.resource_requests,
            num_procs=group.num_procs,
            semaphores_spec=group.semaphores_spec,
            provisioning_model=group.provisioning_model,
            gpu_accelerator_type=group.gpu_accelerator_type,
            adc_available=adc_available,
        )
        deployment_ctx_mngr = k8s.deployment_ctx_mngr(
            execution_id,
            cluster_info=cluster,
            deployment=deployment,
            secrets=[],
        )
        ctx_managers.append(deployment_ctx_mngr)
    return task_queue, ctx_managers


def get_gcp_with_sqs_config(
    execution_id: str,
    image: str,
    groups: dict[str, WorkerGroupDict],
    cluster: k8s.ClusterInfo,
    ctx_managers: list[AbstractContextManager],
) -> tuple[PushMessageQueue[Task], PullMessageQueue[OutcomeReport], list[AbstractContextManager]]:
    task_queues = []
    secrets, env_secret_mapping, adc_available = k8s.get_secrets_and_mapping(
        execution_id, REQUIRED_ENV_VARS
    )

    outcome_queue_name = f"run-{execution_id}-outcome"
    outcome_queue_spec = {"@type": "SQSQueue", "name": outcome_queue_name, "pull_wait_sec": 2.5}
    outcome_queue = builder.build(outcome_queue_spec)
    ctx_managers.append(aws_sqs.sqs_queue_ctx_mngr(execution_id, outcome_queue))
    ctx_managers.append(k8s.secrets_ctx_mngr(execution_id, secrets=secrets, cluster_info=cluster))

    sqs_trigger_name = f"run-{execution_id}-keda-trigger-auth-aws"
    ctx_managers.append(k8s.sqs_trigger_ctx_mngr(execution_id, cluster, sqs_trigger_name))

    for group_name, group_dict in groups.items():
        group = WorkerGroup(**group_dict)
        group_name = group_name.replace("_", "-")
        task_queue, group_ctx_managers = _get_group_taskqueue_and_contexts(
            execution_id=execution_id,
            image=image,
            group=group,
            group_name=group_name,
            cluster=cluster,
            sqs_trigger_name=sqs_trigger_name,
            outcome_queue_spec=outcome_queue_spec,
            env_secret_mapping=env_secret_mapping,
            adc_available=adc_available,
        )
        task_queues.append(task_queue)
        ctx_managers.extend(group_ctx_managers)

    return TaskRouter(task_queues), outcome_queue, ctx_managers


@typechecked
@builder.register("mazepa.execute_on_gcp_with_sqs", versions=">=0.0.3")
def execute_on_gcp_with_sqs(  # pylint: disable=too-many-locals
    target: Union[mazepa.Flow, mazepa.ExecutionState],
    worker_image: str,
    worker_groups: dict[str, WorkerGroupDict],
    worker_cluster_name: Optional[str] = None,
    worker_cluster_region: Optional[str] = None,
    worker_cluster_project: Optional[str] = None,
    max_batch_len: int = 10000,
    batch_gap_sleep_sec: float = 0.5,
    num_procs: int = 1,
    semaphores_spec: dict[SemaphoreType, int] | None = None,
    extra_ctx_managers: Iterable[AbstractContextManager] = (),
    show_progress: bool = True,
    do_dryrun_estimation: bool = True,
    local_test: bool = False,
    local_test_queues_dir: str | None = None,
    debug: bool = False,
    checkpoint: Optional[str] = None,
    checkpoint_interval_sec: float = 300.0,
    raise_on_failed_checkpoint: bool = True,
):
    if debug and not local_test:
        raise ValueError("`debug` can only be set to `True` when `local_test` is also `True`.")

    if local_test:
        execute_locally(
            target=target,
            execution_id=run.RUN_ID,
            max_batch_len=max_batch_len,
            batch_gap_sleep_sec=batch_gap_sleep_sec,
            show_progress=show_progress,
            do_dryrun_estimation=do_dryrun_estimation,
            queues_dir=local_test_queues_dir,
            checkpoint=checkpoint,
            checkpoint_interval_sec=checkpoint_interval_sec,
            raise_on_failed_checkpoint=raise_on_failed_checkpoint,
            num_procs=num_procs,
            semaphores_spec=semaphores_spec,
            debug=debug,
        )
    else:
        assert gcloud.check_image_exists(worker_image), worker_image
        _ensure_required_env_vars()
        ctx_managers = copy.copy(list(extra_ctx_managers))

        if local_test_queues_dir:
            logger.warning(
                "`local_test_queues_dir` was given, but `local_test` is False. "
                "The argument will be unused, and remote workers will use the "
                "default locations for their local task and outcome queues."
            )

        if worker_cluster_name is None:
            logger.info(f"Cluster info not provided, using default: {DEFAULT_GCP_CLUSTER}")
            worker_cluster = DEFAULT_GCP_CLUSTER
            if worker_cluster_region is not None or worker_cluster_project is not None:
                raise ValueError(
                    "Both `worker_cluster_region` and `worker_cluster_project` must be `None` "
                    "when `worker_cluster_name` is `None`"
                )
        else:
            if worker_cluster_region is None or worker_cluster_project is None:
                raise ValueError(
                    "Both `worker_cluster_region` and `worker_cluster_project` must be provided "
                    "when `worker_cluster_name` is specified."
                )
            worker_cluster = k8s.ClusterInfo(
                name=worker_cluster_name,
                region=worker_cluster_region,
                project=worker_cluster_project,
            )
        assert (
            run.RUN_ID
        ), f"Invalid RUN_ID [{run.RUN_ID}], might not have been initialized properly."
        run.register_clusters([worker_cluster])
        task_queue, outcome_queue, ctx_managers = get_gcp_with_sqs_config(
            execution_id=run.RUN_ID,
            image=worker_image,
            groups=worker_groups,
            cluster=worker_cluster,
            ctx_managers=ctx_managers,
        )

        with ExitStack() as stack:
            for mngr in ctx_managers:
                stack.enter_context(mngr)

            execute(
                target=target,
                task_queue=task_queue,
                outcome_queue=outcome_queue,
                execution_id=run.RUN_ID,
                max_batch_len=max_batch_len,
                batch_gap_sleep_sec=batch_gap_sleep_sec,
                show_progress=show_progress,
                do_dryrun_estimation=do_dryrun_estimation,
                checkpoint=checkpoint,
                checkpoint_interval_sec=checkpoint_interval_sec,
                raise_on_failed_checkpoint=raise_on_failed_checkpoint,
            )
