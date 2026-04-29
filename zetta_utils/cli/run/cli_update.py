import click
from kubernetes.client import ApiException

from kubernetes import client as k8s_client
from kubernetes import config  # type: ignore
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

    config.load_kube_config()
    apps_api = k8s_client.AppsV1Api()
    deployments = apps_api.list_namespaced_deployment(
        namespace="default", label_selector=f"run_id={run_id}"
    ).items
    if not deployments:
        raise click.UsageError(
            f"No deployments found for run_id={run_id!r}. "
            f"Check the run id and that the run is still active."
        )
    by_group = {d.metadata.labels.get("worker_group"): d for d in deployments}
    body = {"metadata": {"annotations": annotations}}
    for group_name in worker_groups:
        deployment = by_group.get(group_name)
        if deployment is None:
            click.echo(
                f"Worker group {group_name!r} not found in run {run_id!r}; "
                f"available: {sorted(g for g in by_group if g)}",
                err=True,
            )
            continue
        deployment_name = deployment.metadata.name
        try:
            apps_api.patch_namespaced_deployment(
                name=deployment_name, namespace="default", body=body
            )
            logger.info(f"Updated {deployment_name}: {annotations}")
        except ApiException as exc:
            logger.warning(f"Failed to patch deployment {deployment_name}: {exc}")
