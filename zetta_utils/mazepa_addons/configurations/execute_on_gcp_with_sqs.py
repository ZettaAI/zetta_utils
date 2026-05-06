# pylint: disable=too-many-locals
from __future__ import annotations

import copy
import os
import threading
from contextlib import AbstractContextManager, ExitStack
from typing import Any, Final, Iterable, Optional, TypedDict, Union

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

# Env vars that will be propagated to worker pods only if they are set on the master.
OPTIONAL_ENV_VARS: Final = [
    "CLICKHOUSE_PASSWORD",
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
    num_procs: int = 1
    sqs_based_scaling: bool = True
    resource_requests: dict[str, int | float | str] | None = None
    semaphores_spec: dict[SemaphoreType, int] | None = None
    provisioning_model: k8s.ProvisioningModel = "spot"
    idle_worker_timeout: int = 300
    labels: dict[str, str] | None = None
    gpu_accelerator_type: str | None = None
    required_zones: list[str] | None = None  # k8s will schedule workers in these zones
    preferred_zones: list[str] | None = None  # k8s will try to schedule workers in these zones
    # SIGTERM-to-SIGKILL grace window when a worker pod is being terminated by the
    # autoscaler scale-down; sized to the worst-case task duration so the worker
    # has time to drain its current task before getting killed.
    termination_grace_seconds: int = 300
    # When True, the autoscaler watches TriggeredScaleUp events for this group's
    # pods and nudges the chosen node pool directly when GKE's cluster
    # autoscaler hits its post-RESOURCE_POOL_EXHAUSTED backoff. Set False to
    # leave node-pool sizing entirely to the cluster autoscaler.
    nudge_node_pools: bool = True


class WorkerGroupDict(TypedDict, total=False):
    replicas: int
    resource_limits: dict[str, int | float | str]
    num_procs: NotRequired[int]
    sqs_based_scaling: NotRequired[bool]
    resource_requests: NotRequired[dict[str, int | float | str]]
    semaphores_spec: NotRequired[dict[SemaphoreType, int]]
    provisioning_model: NotRequired[k8s.ProvisioningModel]
    idle_worker_timeout: NotRequired[int]
    labels: NotRequired[dict[str, str]]
    gpu_accelerator_type: NotRequired[str]
    required_zones: NotRequired[list[str]]
    preferred_zones: NotRequired[list[str]]
    termination_grace_seconds: NotRequired[int]
    nudge_node_pools: NotRequired[bool]


def _get_group_taskqueue_and_contexts(
    execution_id: str,
    image: str,
    group: WorkerGroup,
    group_name: str,
    cluster: k8s.ClusterInfo,
    outcome_queue_spec: dict[str, Any],
    env_secret_mapping: dict[str, str],
    suppress_worker_logs: bool,
    resource_monitor_interval: float | None,
    adc_available: bool = False,
    cave_secret_available: bool = False,
) -> tuple[PushMessageQueue[Task], list[AbstractContextManager]]:
    ctx_managers: list[AbstractContextManager] = []

    work_queue_name = f"run-{execution_id}_{group_name}_work"
    task_queue_spec = {"@type": "SQSQueue", "name": work_queue_name}
    task_queue = builder.build(task_queue_spec)
    ctx_managers.append(aws_sqs.sqs_queue_ctx_mngr(execution_id, task_queue))
    env_secret_mapping["RUN_ID"] = execution_id

    initial_replicas = 1 if group.sqs_based_scaling else group.replicas
    deployment_labels = {"run_id": execution_id, "worker_group": group_name}
    if group.labels:
        deployment_labels.update(group.labels)
    deployment = k8s.get_mazepa_worker_deployment(
        f"{execution_id}-{group_name}",
        image=image,
        task_queue_spec=task_queue_spec,
        outcome_queue_spec=outcome_queue_spec,
        replicas=initial_replicas,
        resources=group.resource_limits,
        env_secret_mapping=env_secret_mapping,
        labels=deployment_labels,
        resource_requests=group.resource_requests,
        num_procs=group.num_procs,
        semaphores_spec=group.semaphores_spec,
        provisioning_model=group.provisioning_model,
        gpu_accelerator_type=group.gpu_accelerator_type,
        adc_available=adc_available,
        cave_secret_available=cave_secret_available,
        suppress_worker_logs=suppress_worker_logs,
        resource_monitor_interval=resource_monitor_interval,
        required_zones=group.required_zones,
        preferred_zones=group.preferred_zones,
        worker_type=group_name,
        termination_grace_seconds=group.termination_grace_seconds,
    )
    if group.sqs_based_scaling:
        ctx_managers.append(
            k8s.autoscaling_deployment_ctx_mngr(
                execution_id,
                cluster_info=cluster,
                deployment=deployment,
                secrets=[],
                queue_name=task_queue.name,
                region_name=task_queue.region_name,
                max_replicas=group.replicas,
                nudge_node_pools=group.nudge_node_pools,
            )
        )
    else:
        ctx_managers.append(
            k8s.deployment_ctx_mngr(
                execution_id,
                cluster_info=cluster,
                deployment=deployment,
                secrets=[],
            )
        )
    return task_queue, ctx_managers


def get_gcp_with_sqs_config(
    execution_id: str,
    image: str,
    groups: dict[str, WorkerGroupDict],
    cluster: k8s.ClusterInfo,
    ctx_managers: list[AbstractContextManager],
    suppress_worker_logs: bool,
    resource_monitor_interval: float | None,
) -> tuple[PushMessageQueue[Task], PullMessageQueue[OutcomeReport], list[AbstractContextManager]]:
    task_queues = []
    share_envs = list(REQUIRED_ENV_VARS) + [e for e in OPTIONAL_ENV_VARS if e in os.environ]
    (
        secrets,
        env_secret_mapping,
        adc_available,
        cave_secret_available,
    ) = k8s.get_secrets_and_mapping(execution_id, share_envs)

    outcome_queue_name = f"run-{execution_id}-outcome"
    outcome_queue_spec = {"@type": "SQSQueue", "name": outcome_queue_name, "pull_wait_sec": 2.5}
    outcome_queue = builder.build(outcome_queue_spec)
    ctx_managers.append(aws_sqs.sqs_queue_ctx_mngr(execution_id, outcome_queue))
    ctx_managers.append(k8s.secrets_ctx_mngr(execution_id, secrets=secrets, cluster_info=cluster))

    for group_name, group_dict in groups.items():
        group = WorkerGroup(**group_dict)
        group_name = group_name.replace("_", "-")
        task_queue, group_ctx_managers = _get_group_taskqueue_and_contexts(
            execution_id=execution_id,
            image=image,
            group=group,
            group_name=group_name,
            cluster=cluster,
            outcome_queue_spec=outcome_queue_spec,
            env_secret_mapping=env_secret_mapping,
            adc_available=adc_available,
            cave_secret_available=cave_secret_available,
            suppress_worker_logs=suppress_worker_logs,
            resource_monitor_interval=resource_monitor_interval,
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
    write_progress_summary: bool = False,
    require_interrupt_confirm: bool = True,
    suppress_worker_logs: bool = True,
    resource_monitor_interval: float | None = None,
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
            write_progress_summary=write_progress_summary,
            require_interrupt_confirm=require_interrupt_confirm,
            suppress_worker_logs=suppress_worker_logs,
            resource_monitor_interval=resource_monitor_interval,
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
            suppress_worker_logs=suppress_worker_logs,
            resource_monitor_interval=resource_monitor_interval,
        )

        if semaphores_spec is not None:
            sem_widths = {str(k): v for k, v in semaphores_spec.items()}
            run.update_run_info(
                run.RUN_ID,
                {run.RunInfo.SEMAPHORE_WIDTHS.value: sem_widths},
            )

        thread = threading.Thread(
            target=k8s.pod.watch_for_pod_disruptions,
            args=(run.RUN_ID, worker_cluster),
            daemon=True,
        )
        thread.start()

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
                write_progress_summary=write_progress_summary,
                require_interrupt_confirm=require_interrupt_confirm,
            )
