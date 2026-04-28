import click
from kubernetes.client import ApiException

from kubernetes import client as k8s_client
from zetta_utils.cloud_management.resource_allocation.k8s.autoscaler import (
    MAX_REPLICAS_ANNOTATION,
    MIN_REPLICAS_ANNOTATION,
    SCALE_DOWN_STABILIZATION_SEC_ANNOTATION,
)
from zetta_utils.log import get_logger

logger = get_logger("zetta_utils")


@click.group()
def run_update_cli():
    ...


@run_update_cli.command()
@click.argument("run_id", type=str)
@click.option(
    "-g",
    "--worker-groups",
    type=str,
    multiple=True,
    required=True,
    help="Worker group name(s) to update.",
)
@click.option(
    "-w",
    "--max-workers",
    type=int,
    default=None,
    help="New max-replicas cap for the autoscaler.",
)
@click.option(
    "--min-workers",
    type=int,
    default=None,
    help="New min-replicas floor for the autoscaler.",
)
@click.option(
    "--scale-down-stabilization-sec",
    type=int,
    default=None,
    help="New scale-down debounce window (seconds).",
)
def run_update(
    run_id: str,
    worker_groups: list[str],
    max_workers: int | None,
    min_workers: int | None,
    scale_down_stabilization_sec: int | None,
):
    """Update autoscaler scaling knobs on the Deployment for one or more groups in a run.

    Patches annotations on each group's Deployment; the autoscaler reads
    those annotations on its next tick. At least one of ``--max-workers``,
    ``--min-workers``, ``--scale-down-stabilization-sec`` must be set.
    """
    annotations: dict[str, str] = {}
    if max_workers is not None:
        annotations[MAX_REPLICAS_ANNOTATION] = str(max_workers)
    if min_workers is not None:
        annotations[MIN_REPLICAS_ANNOTATION] = str(min_workers)
    if scale_down_stabilization_sec is not None:
        annotations[SCALE_DOWN_STABILIZATION_SEC_ANNOTATION] = str(scale_down_stabilization_sec)
    if not annotations:
        raise click.UsageError(
            "Specify at least one of --max-workers, --min-workers, "
            "--scale-down-stabilization-sec"
        )

    apps_api = k8s_client.AppsV1Api()
    body = {"metadata": {"annotations": annotations}}
    for group_name in worker_groups:
        deployment_name = f"run-{run_id}-{group_name}".replace("_", "-")
        try:
            apps_api.patch_namespaced_deployment(
                name=deployment_name, namespace="default", body=body
            )
            logger.info(f"Updated {deployment_name}: {annotations}")
        except ApiException as exc:
            if exc.status == 404:
                logger.info(f"Deployment does not exist: {deployment_name}: {exc}")
            else:
                logger.warning(f"Failed to patch deployment {deployment_name}: {exc}")
