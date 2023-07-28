from __future__ import annotations

import copy
import os
from contextlib import AbstractContextManager, ExitStack
from typing import Dict, Final, Iterable, Optional, Union

from zetta_utils import builder, log, mazepa
from zetta_utils.cloud import execution_tracker, resource_allocation

logger = log.get_logger("zetta_utils")

REQUIRED_ENV_VARS: Final = [
    # "GRAFANA_CLOUD_ACCESS_KEY",
    "ZETTA_USER",
    "ZETTA_PROJECT",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "WANDB_API_KEY",
]


DEFAULT_GCP_CLUSTER_NAME: Final = "zutils-x3"
DEFAULT_GCP_CLUSTER_REGION: Final = "us-east1"
DEFAULT_GCP_CLUSTER_PROJECT: Final = "zetta-research"

DEFAULT_GCP_CLUSTER: Final = resource_allocation.k8s.ClusterInfo(
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


def get_gcp_with_sqs_config(
    execution_id: str,
    worker_image: str,
    worker_cluster: resource_allocation.k8s.ClusterInfo,
    worker_replicas: int,
    worker_resources: Dict[str, int | float | str],
    worker_labels: Optional[Dict[str, str]],
    ctx_managers: list[AbstractContextManager],
) -> tuple[mazepa.ExecutionQueue, list[AbstractContextManager]]:

    work_queue_name = f"zzz-{execution_id}-work"
    ctx_managers.append(
        resource_allocation.aws_sqs.sqs_queue_ctx_mngr(execution_id, work_queue_name)
    )
    outcome_queue_name = f"zzz-{execution_id}-outcome"
    ctx_managers.append(
        resource_allocation.aws_sqs.sqs_queue_ctx_mngr(execution_id, outcome_queue_name)
    )
    exec_queue_spec = {
        "@type": "mazepa.SQSExecutionQueue",
        "name": work_queue_name,
        "outcome_queue_name": outcome_queue_name,
    }
    exec_queue = builder.build(exec_queue_spec)

    secrets, env_secret_mapping = resource_allocation.k8s.get_secrets_and_mapping(
        execution_id, REQUIRED_ENV_VARS
    )

    deployment = resource_allocation.k8s.get_zutils_worker_deployment(
        execution_id=execution_id,
        image=worker_image,
        queue=exec_queue_spec,
        replicas=worker_replicas,
        resources=worker_resources,
        env_secret_mapping=env_secret_mapping,
        labels=worker_labels,
    )

    ctx_managers.append(
        resource_allocation.k8s.deployment_ctx_mngr(
            execution_id=execution_id,
            cluster_info=worker_cluster,
            deployment=deployment,
            secrets=secrets,
        )
    )
    return exec_queue, ctx_managers


@builder.register("mazepa.execute_on_gcp_with_sqs")
def execute_on_gcp_with_sqs(  # pylint: disable=too-many-locals
    target: Union[mazepa.Flow, mazepa.ExecutionState],
    worker_image: str,
    worker_replicas: int,
    worker_labels: Optional[Dict[str, str]] = None,
    worker_cluster_name: Optional[str] = None,
    worker_cluster_region: Optional[str] = None,
    worker_cluster_project: Optional[str] = None,
    worker_resources: Dict[str, int | float | str] | None = None,
    max_batch_len: int = 10000,
    batch_gap_sleep_sec: float = 4.0,
    extra_ctx_managers: Iterable[AbstractContextManager] = (),
    show_progress: bool = True,
    do_dryrun_estimation: bool = True,
    local_test: bool = False,
    checkpoint: Optional[str] = None,
    checkpoint_interval_sec: float = 300.0,
    raise_on_failed_checkpoint: bool = True,
):
    _ensure_required_env_vars()
    execution_id = mazepa.id_generation.get_unique_id(
        prefix="exec", slug_len=4, add_uuid=False, max_len=50
    )

    execution_tracker.record_execution_run(execution_id)

    ctx_managers = copy.copy(list(extra_ctx_managers))
    if local_test:
        execution_tracker.register_execution(execution_id, [])
        exec_queue: mazepa.ExecutionQueue = mazepa.LocalExecutionQueue()
    else:
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
            worker_cluster = resource_allocation.k8s.ClusterInfo(
                name=worker_cluster_name,
                region=worker_cluster_region,
                project=worker_cluster_project,
            )
        execution_tracker.register_execution(execution_id, [worker_cluster])
        exec_queue, ctx_managers = get_gcp_with_sqs_config(
            execution_id=execution_id,
            worker_image=worker_image,
            worker_cluster=worker_cluster,
            worker_labels=worker_labels,
            worker_replicas=worker_replicas,
            worker_resources=worker_resources if worker_resources else {},
            ctx_managers=ctx_managers,
        )

    with ExitStack() as stack:
        stack.enter_context(execution_tracker.heartbeat_tracking_ctx_mngr(execution_id))
        for mngr in ctx_managers:
            stack.enter_context(mngr)

        mazepa.execute(
            target=target,
            exec_queue=exec_queue,
            execution_id=execution_id,
            max_batch_len=max_batch_len,
            batch_gap_sleep_sec=batch_gap_sleep_sec,
            show_progress=show_progress,
            do_dryrun_estimation=do_dryrun_estimation,
            checkpoint=checkpoint,
            checkpoint_interval_sec=checkpoint_interval_sec,
            raise_on_failed_checkpoint=raise_on_failed_checkpoint,
        )
