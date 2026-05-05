"""CLI: manually pester GKE for more nodes for a worker group's pools.

Use case: the in-process autoscaler / nudger sizes resizes from the
cluster autoscaler's ``TriggeredScaleUp`` events. When CA's per-pool
backoff suppresses those events (5-30 min after a failed scale-up), the
user can drive resizes directly via this command instead of waiting for
backoff to expire.
"""
from __future__ import annotations

import signal
import threading

import click
from kubernetes.client.exceptions import ApiException

from kubernetes import client as k8s_client
from kubernetes import config  # type: ignore
from zetta_utils.cloud_management.resource_allocation.k8s import gke
from zetta_utils.cloud_management.resource_allocation.k8s.common import (
    parse_cluster_info,
)
from zetta_utils.log import get_logger

logger = get_logger("zetta_utils")


@click.group()
def pester_nodes_cli():
    ...


@pester_nodes_cli.command()
@click.argument("run_id", type=str)
@click.option(
    "-g",
    "--worker-group",
    type=str,
    required=True,
    help="Worker group name to pester pools for.",
)
@click.option(
    "-n",
    "--additional-nodes",
    type=int,
    required=True,
    help="Per-zone nodes to add per pool on top of its current size.",
)
@click.option("--cluster-name", type=str, default=None, help="Cluster name (default: env config).")
@click.option("--cluster-region", type=str, default=None, help="Cluster region.")
@click.option("--cluster-project", type=str, default=None, help="Cluster project.")
@click.option(
    "--interval-sec",
    type=int,
    default=60,
    help="Seconds between resize iterations (default 60).",
)
def pester_nodes(  # pylint: disable=too-many-arguments,too-many-locals
    run_id: str,
    worker_group: str,
    additional_nodes: int,
    cluster_name: str | None,
    cluster_region: str | None,
    cluster_project: str | None,
    interval_sec: int,
):
    """Pester GKE for additional nodes on the pool(s) hosting a worker group.

    Discovers the pool(s) by inspecting the currently-scheduled pods of
    ``run_id`` / ``worker_group`` and reading their nodes'
    ``cloud.google.com/gke-nodepool`` label. Computes a fixed per-pool
    target = current_size + ``--additional-nodes`` (per zone, capped at
    each pool's autoscaling.maxNodeCount), then loops every
    ``--interval-sec`` re-issuing ``SetNodePoolSize`` to that target until
    the deployment's pending replicas hit zero or the user sends SIGINT.
    Re-issuing is idempotent — when GCE's underlying MIG resize fails
    transiently (capacity exhausted), repeated calls give it fresh chances
    once capacity opens up.
    """
    cluster_info = parse_cluster_info(cluster_name, cluster_region, cluster_project)
    if cluster_info.project is None or cluster_info.region is None:
        raise click.UsageError(
            "GKE pool resize requires cluster_info with project and region set; "
            "supply --cluster-name / --cluster-region / --cluster-project."
        )
    project = cluster_info.project
    region = cluster_info.region
    cluster = cluster_info.name

    config.load_kube_config()
    core_api = k8s_client.CoreV1Api()
    apps_api = k8s_client.AppsV1Api()

    group_label = worker_group.replace("_", "-")
    label_selector = f"run_id={run_id},worker_group={group_label}"

    deployment_name = _resolve_deployment(apps_api, label_selector, run_id, worker_group)

    pool_names = _resolve_pools_from_pods(core_api, label_selector)
    if not pool_names:
        raise click.UsageError(
            f"No scheduled pods found for run_id={run_id!r} "
            f"worker_group={group_label!r}; cannot infer pools. "
            f"Wait until at least one pod is running, then re-run."
        )
    click.echo(f"Discovered pools: {sorted(pool_names)}")

    targets = _compute_targets(project, region, cluster, pool_names, additional_nodes)
    if not targets:
        raise click.UsageError("No pools left to pester (all at max or unresolved).")
    click.echo(
        "Per-pool targets (per-zone): "
        + ", ".join(f"{name}={tgt}" for name, tgt in sorted(targets.items()))
    )

    stop_event = threading.Event()

    def _on_sigint(_signum, _frame):
        click.echo("\nInterrupted; exiting.", err=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _on_sigint)

    iteration = 0
    while not stop_event.is_set():
        iteration += 1
        dep = apps_api.read_namespaced_deployment(name=deployment_name, namespace="default")
        spec = dep.spec.replicas or 0
        ready = dep.status.ready_replicas or 0
        pending = spec - ready
        if pending <= 0:
            click.echo(f"[iter {iteration}] all {spec} replicas ready; done.")
            return

        click.echo(f"[iter {iteration}] pending={pending} (spec={spec}, ready={ready})")
        for pool_name, target in sorted(targets.items()):
            try:
                gke.resize_node_pool(project, region, cluster, pool_name, target)
                click.echo(f"  pool {pool_name}: requested per-zone target={target}")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning(f"resize {pool_name}: {exc}")

        if stop_event.wait(interval_sec):
            break

    click.echo("Exited.")


def _resolve_deployment(
    apps_api: k8s_client.AppsV1Api, label_selector: str, run_id: str, worker_group: str
) -> str:
    deps = apps_api.list_namespaced_deployment(
        namespace="default", label_selector=label_selector
    ).items
    if not deps:
        raise click.UsageError(
            f"No deployment found for run_id={run_id!r} worker_group={worker_group!r}."
        )
    if len(deps) > 1:
        names = sorted(d.metadata.name for d in deps)
        raise click.UsageError(
            f"Multiple deployments matched {label_selector!r}: {names}. "
            f"Refine the run_id / worker_group."
        )
    return deps[0].metadata.name


def _resolve_pools_from_pods(core_api: k8s_client.CoreV1Api, label_selector: str) -> set[str]:
    pods = core_api.list_namespaced_pod(namespace="default", label_selector=label_selector).items
    node_names = {p.spec.node_name for p in pods if p.spec.node_name}
    pools: set[str] = set()
    for nn in node_names:
        try:
            node = core_api.read_node(name=nn)
        except ApiException as exc:
            logger.warning(f"read node {nn}: {exc}")
            continue
        pool = (node.metadata.labels or {}).get("cloud.google.com/gke-nodepool")
        if pool:
            pools.add(pool)
    return pools


def _compute_targets(
    project: str, region: str, cluster: str, pool_names: set[str], additional_nodes: int
) -> dict[str, int]:
    """Snapshot each pool's current size + additional_nodes, capped at maxNodeCount.

    Returns a ``{pool_name: target_per_zone}`` map. Pools already at max are
    omitted with a message.
    """
    pools = gke.list_node_pools(project, region, cluster)
    by_name = {p.name: p for p in pools}
    targets: dict[str, int] = {}
    for name in sorted(pool_names):
        pool = by_name.get(name)
        if pool is None:
            click.echo(f"  pool {name!r} not found in cluster; skipping")
            continue
        current = pool.initial_node_count
        max_size = pool.autoscaling.max_node_count if pool.autoscaling else None
        target = current + additional_nodes
        if max_size is not None:
            if current >= max_size:
                click.echo(f"  pool {name}: already at max ({max_size}); skipping")
                continue
            target = min(target, max_size)
        targets[name] = target
    return targets
