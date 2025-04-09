from __future__ import annotations

import copy
import os
import pathlib
import re
import shutil
import subprocess
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Any, Final, Iterable, Literal, Optional, Union

import attrs
from simple_slurm import Slurm

from zetta_utils import builder, log, mazepa, run
from zetta_utils.cloud_management import resource_allocation
from zetta_utils.cloud_management.resource_allocation.k8s import (
    get_mazepa_worker_command,
)
from zetta_utils.common.ctx_managers import set_env_ctx_mngr
from zetta_utils.mazepa import SemaphoreType, execute
from zetta_utils.mazepa.task_outcome import OutcomeReport
from zetta_utils.mazepa.tasks import Task
from zetta_utils.message_queues import sqs  # pylint: disable=unused-import
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
    job_id = None
    try:
        with set_env_ctx_mngr(CURRENT_BUILD_SPEC="", ZETTA_RUN_SPEC_PATH=""):
            job_id = slurm_obj.sbatch()

        logger.info(f"Started SLURM job {job_id}")
        yield
    except AssertionError as e:
        raise ProcessLookupError(
            "SLURM job failed to start. Check stderr for additional info from `sbatch` command"
        ) from e
    finally:
        if job_id is not None:
            logger.info(f"Cancelling SLURM job {job_id}")
            with set_env_ctx_mngr(CURRENT_BUILD_SPEC="", ZETTA_RUN_SPEC_PATH=""):
                subprocess.run(
                    ["scancel", f"{job_id}"], capture_output=True, text=True, check=True
                )


def check_cpus_per_task(_, __, value):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"cpus_per_task must be a positive int, got {value}")


def check_mem_per_cpu(_, __, value):
    if not isinstance(value, str) or not value:
        raise ValueError("mem_per_cpu must be a non-empty string")


def check_gres(_, __, value):
    if not (
        value is None
        or isinstance(value, str)
        or (isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value))
    ):
        raise ValueError("gres must be a (list of) string or None")


def check_timedelta(_, __, value):
    timedelta_pattern = re.compile(r"^(\d+-\d+(:\d+(:\d+)?)?)|(\d+:\d+(:\d+)?)|(\d+)$")

    if not (isinstance(value, str) or timedelta_pattern.match(value)):
        raise ValueError(
            'time must be a string in the format "minutes", "minutes:seconds", '
            '"hours:minutes:seconds", "days-hours", "days-hours:minutes" or '
            '"days-hours:minutes:seconds"'
        )


def check_array(_, __, value):
    if not (
        value is None
        or isinstance(value, (str, range))  # "0-31", "1,3,5,7", "1-7:2" are valid strings
        or (isinstance(value, Iterable) and all(isinstance(v, int) for v in value))
    ):
        raise ValueError(
            'array must be a list or range of int, a str ("0-31", "1,3,5,7", "1-7:2"), or None'
        )


@contextmanager
def fq_ctx_manager(
    queue_name: str,
):
    try:
        yield
    finally:
        logger.info(f"Deleting FQ `{queue_name}`")
        shutil.rmtree(queue_name)


@attrs.define
class SlurmWorkerResources:
    cpus_per_task: int = attrs.field(validator=check_cpus_per_task)
    mem_per_cpu: str = attrs.field(validator=check_mem_per_cpu)
    gres: str | None = attrs.field(validator=check_gres, default=None)
    array: Any = attrs.field(validator=check_array, default=None)
    time: str = attrs.field(validator=check_timedelta, default="6-0:0:0")
    partition: str | None = None

    def to_dict(self) -> dict:
        return dict(attrs.asdict(self, filter=lambda attr, value: value is not None).items())

    @classmethod
    def from_dict(cls, data: dict) -> SlurmWorkerResources:
        """Creates a TaskResources instance from a dictionary."""
        return cls(**data)


def get_slurm_contex_managers(
    execution_id: str,
    worker_replicas: int,
    init_command: str,
    slurm_worker_resources: SlurmWorkerResources,
    ctx_managers: list[AbstractContextManager],
    debug: bool,
    num_procs: int,
    semaphores_spec: dict[SemaphoreType, int] | None,
    message_queue: Literal["sqs", "fq"],
) -> tuple[PushMessageQueue[Task], PullMessageQueue[OutcomeReport], list[AbstractContextManager]]:
    work_queue_name = f"run-{execution_id}-work"
    outcome_queue_name = f"run-{execution_id}-outcome"

    if message_queue == "fq":
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
        if not debug:
            ctx_managers.append(fq_ctx_manager(work_queue_name))
            ctx_managers.append(fq_ctx_manager(outcome_queue_name))
    else:
        assert message_queue == "sqs"
        task_queue_spec = {
            "@type": "SQSQueue",
            "name": work_queue_name,
        }
        outcome_queue_spec = {
            "@type": "SQSQueue",
            "name": outcome_queue_name,
            "pull_wait_sec": 2.5,
        }
        task_queue = builder.build(task_queue_spec)
        outcome_queue = builder.build(outcome_queue_spec)
        ctx_managers.append(
            resource_allocation.aws_sqs.sqs_queue_ctx_mngr(execution_id, task_queue)
        )

        ctx_managers.append(
            resource_allocation.aws_sqs.sqs_queue_ctx_mngr(execution_id, outcome_queue)
        )

    pathlib.Path(f"slurm_{execution_id}").mkdir()
    slurm_obj = Slurm(
        output=f"slurm_{execution_id}/{Slurm.JOB_ARRAY_ID}.out",
        ntasks=worker_replicas,
        job_name=f"slurm-worker-{execution_id}",
        **slurm_worker_resources.to_dict(),
    )

    worker_command = get_mazepa_worker_command(
        task_queue_spec, outcome_queue_spec, num_procs, semaphores_spec
    )
    slurm_obj.add_cmd(init_command)
    slurm_obj.add_cmd(f"srun {worker_command}")

    ctx_managers.append(slurm_job_ctx_manager(slurm_obj=slurm_obj))
    return task_queue, outcome_queue, ctx_managers


@builder.register("mazepa.execute_on_slurm")
def execute_on_slurm(  # pylint: disable=too-many-locals
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
    message_queue: Literal["sqs", "fq"] = "sqs",
):
    slurm_worker_resources = SlurmWorkerResources.from_dict(worker_resources)

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

        task_queue, outcome_queue, ctx_managers = get_slurm_contex_managers(
            execution_id=run.RUN_ID,
            worker_replicas=worker_replicas,
            ctx_managers=ctx_managers,
            init_command=init_command,
            num_procs=num_procs,
            semaphores_spec=semaphores_spec,
            slurm_worker_resources=slurm_worker_resources,
            debug=debug,
            message_queue=message_queue,
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
