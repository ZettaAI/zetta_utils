import click

from zetta_utils.log import get_logger

logger = get_logger("zetta_utils")


@click.group()
def run_update_cli():
    ...


@run_update_cli.command()
@click.argument("run_id", type=str)
@click.option("-w", "--max-workers", type=int, required=True, help="Max number of workers.")
@click.option("-g", "--worker-groups", type=str, multiple=True, help="Worker group names.")
def run_update(run_id: str, max_workers: int, worker_groups: list[str] | None = None):
    """
    Update max workers for a given `run_id` and/or group names.
    """
    from kubernetes.client import ApiException

    from zetta_utils.cloud_management.resource_allocation import k8s

    def _patch(job_name: str, patch_body: dict):
        job_name = job_name.replace("_", "-")
        try:
            k8s.patch_scaledjob(job_name, patch_body=patch_body)
        except ApiException as exc:
            if exc.status == 404:
                logger.info(f"Resource does not exist: `{job_name}`: {exc}")
            else:
                msg = f"Failed to update k8s resource `{job_name}`: {exc}"
                logger.warning(msg)

    patch_body = {"spec": {"maxReplicaCount": max_workers}}
    if worker_groups:
        for job_name in worker_groups:
            _patch(f"run-{run_id}-{job_name}-sj", patch_body)
    else:
        # for deprecated keda
        _patch(f"run-{run_id}-sj", patch_body)
