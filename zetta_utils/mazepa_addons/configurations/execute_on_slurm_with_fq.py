from __future__ import annotations

import copy
import datetime
import os
import subprocess
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Any, Final, Iterable, Optional, Union

import attrs
from simple_slurm import Slurm

from zetta_utils import builder, log, mazepa, run
from zetta_utils.cloud_management.resource_allocation.k8s.common import (
    get_mazepa_worker_command,
)
from zetta_utils.mazepa import SemaphoreType, execute
from zetta_utils.mazepa.task_outcome import OutcomeReport
from zetta_utils.mazepa.tasks import Task
from zetta_utils.message_queues.base import PullMessageQueue, PushMessageQueue

from .execute_locally import execute_locally

logger = log.get_logger("zetta_utils")

REQUIRED_ENV_VARS: Final = [
    # "GRAFANA_CLOUD_ACCESS_KEY",
    "ZETTA_USER",
    "ZETTA_PROJECT",
]


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


def is_job_running(job_id: int) -> bool:
    result = subprocess.run(
        f"squeue | grep {job_id}", shell=True, capture_output=True, text=True, check=True
    )
    return len(result.stdout) == 0


@contextmanager
def slurm_job_ctx_manager(
    slurm_obj: Slurm,
):
    try:
        job_id = slurm_obj.sbatch()
        logger.info(f"Started SLURM job {job_id}")
        yield
    except:
        logger.info(f"Cancelling SLURM job {job_id}")
        subprocess.run(["scancel", f"{job_id}"], capture_output=True, text=True, check=True)
        raise


def check_cpus_per_task(_, __, value):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"cpus_per_task must be a positive int, got {value}")


def check_mem_per_cpu(_, __, value):
    if not isinstance(value, str) or not value:
        raise ValueError("mem_per_cpu must be a non-empty string")


def check_gres(_, __, value):
    if value is not None and not isinstance(value, str):
        raise ValueError("gres must be a string or None")


@attrs.define
class SlurmWorkerResources:
    cpus_per_task: int = attrs.field(validator=check_cpus_per_task)
    mem_per_cpu: str = attrs.field(validator=check_mem_per_cpu)
    gres: str | None = attrs.field(validator=check_gres, default=None)

    def to_dict(self) -> dict:
        return dict(attrs.asdict(self, filter=lambda attr, value: value is not None).items())

    @classmethod
    def from_dict(cls, data: dict) -> SlurmWorkerResources:
        """Creates a TaskResources instance from a dictionary."""
        return cls(**data)


def get_slurm_with_fq_config(
    execution_id: str,
    worker_replicas: int,
    init_command: str,
    slurm_worker_resources: SlurmWorkerResources,
    ctx_managers: list[AbstractContextManager],
    num_procs: int = 1,
    semaphores_spec: dict[SemaphoreType, int] | None = None,
) -> tuple[PushMessageQueue[Task], PullMessageQueue[OutcomeReport], list[AbstractContextManager]]:
    work_queue_name = f"zzz-{execution_id}-work"
    outcome_queue_name = f"zzz-{execution_id}-outcome"

    task_queue_spec = {
        "@type": "FileQueue",
        "name": work_queue_name,
    }
    outcome_queue_spec = {
        "@type": "FileQueue",
        "name": outcome_queue_name,
        "pull_wait_sec": 2.5,
    }

    task_queue = builder.build(task_queue_spec)
    outcome_queue = builder.build(outcome_queue_spec)
    slurm_obj = Slurm(
        output=f"slurm_{execution_id}.out",
        time=datetime.timedelta(days=0, hours=0, minutes=10, seconds=4),
        ntasks=worker_replicas,
        partition="highpri",
        job_name=f"slurm-worker-{execution_id}",
        **slurm_worker_resources.to_dict(),
    )

    worker_command = get_mazepa_worker_command(
        task_queue_spec, outcome_queue_spec, num_procs, semaphores_spec
    )
    slurm_obj.add_cmd(init_command)
    slurm_obj.add_cmd(worker_command)

    ctx_managers.append(slurm_job_ctx_manager(slurm_obj=slurm_obj))
    return task_queue, outcome_queue, ctx_managers


@builder.register("mazepa.execute_on_slurm_with_fq")
def execute_on_slurm_with_fq(  # pylint: disable=too-many-locals
    target: Union[mazepa.Flow, mazepa.ExecutionState],
    worker_replicas: int,
    worker_resources: dict[str, Any],
    init_command: str = "",
    max_batch_len: int = 10000,
    batch_gap_sleep_sec: float = 0.5,
    extra_ctx_managers: Iterable[AbstractContextManager] = (),
    show_progress: bool = True,
    do_dryrun_estimation: bool = True,
    local_test: bool = False,
    debug: bool = False,
    checkpoint: Optional[str] = None,
    checkpoint_interval_sec: float = 300.0,
    raise_on_failed_checkpoint: bool = True,
    num_procs: int = 1,
    semaphores_spec: dict[SemaphoreType, int] | None = None,
):
    slurm_worker_resources = SlurmWorkerResources.from_dict(worker_resources)
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
            checkpoint=checkpoint,
            checkpoint_interval_sec=checkpoint_interval_sec,
            raise_on_failed_checkpoint=raise_on_failed_checkpoint,
            num_procs=num_procs,
            semaphores_spec=semaphores_spec,
            debug=debug,
        )
    else:
        _ensure_required_env_vars()
        ctx_managers = copy.copy(list(extra_ctx_managers))

        assert (
            run.RUN_ID
        ), f"Invalid RUN_ID [{run.RUN_ID}], might not have been initialized properly."

        # TODO: we may want to enable GC for SLURM, although it's nontrivial
        # run.register_clusters([worker_cluster])

        task_queue, outcome_queue, ctx_managers = get_slurm_with_fq_config(
            execution_id=run.RUN_ID,
            worker_replicas=worker_replicas,
            ctx_managers=ctx_managers,
            init_command=init_command,
            num_procs=num_procs,
            semaphores_spec=semaphores_spec,
            slurm_worker_resources=slurm_worker_resources,
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
