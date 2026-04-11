import logging
import time
from collections import defaultdict
from copy import copy
from typing import Final

from zetta_utils.cloud_management.resource_allocation import gcloud
from zetta_utils.layer.db_layer.backend import DBRowDataT
from zetta_utils.layer.db_layer.firestore import build_firestore_layer
from zetta_utils.layer.db_layer.layer import DBLayer
from zetta_utils.log import get_logger

from .db import NODE_DB, POD_STATS_DB, RUN_DB

PROJECT = "zetta-research"
DATABASE_NAME = "pricing-db"

logger = get_logger("zetta_utils")

PRICING_LAYERS: dict[str, DBLayer] = {}
PROVISIONING_MAP: Final[dict[str, str]] = {"PREEMPTIBLE": "preemptible", "STANDARD": "ondemand"}
EGRESS_COST_PER_GIB_MIN: Final[float] = 0.02
EGRESS_COST_PER_GIB_MAX: Final[float] = 0.18
BYTES_PER_GIB: Final[int] = 1024 ** 3
_last_gcs_stats: dict[str, tuple] = {}
# Cached compute cost for the active run, populated by compute_costs() and read
# by aggregate_pod_stats() for log enrichment without an extra Firestore read.
# Only one run_id is active per process invocation.
_last_compute_cost: float | None = None


def compute_costs(run_id: str):
    def _get_start_time(node_info: DBRowDataT) -> float:
        """
        If there's a run_id already registered as using this node,
        use the timestamp when current `run_id` was created.
        This likely means a hot node was used in a subsequent run.
        """
        for _run_id in node_info["run_id"]:  # type: ignore
            if _run_id == run_id:
                continue
            if float(str(node_info[str(_run_id)])) < float(str(node_info[run_id])):
                return RUN_DB[run_id]["timestamp"]
        return float(str(node_info["creation_time"]))

    # Tight timeout: this runs on a RepeatTimer thread every 60s; the Firestore
    # SDK's default 300s deadline would block the timer thread for 5 minutes on
    # a transient connectivity blip and delay both subsequent ticks and run
    # shutdown.
    nodes = NODE_DB.query({"-run_id": [run_id]}, timeout=30.0)
    total_cost = 0.0
    for _info in nodes.values():
        node_cost_hourly = 0.0
        node_info: DBRowDataT = _info
        machine_class, _, _ = str(node_info["machine_type"]).split("-")
        node_region = node_info["region"]
        provisioning_model = PROVISIONING_MAP[str(node_info["provisioning_model"])]

        cpu = (f"CPU-{provisioning_model}", float(str(node_info["cpu_count"])))
        ram = (f"RAM-{provisioning_model}", float(str(node_info["memory_gib"])))

        for lname, count in [cpu, ram]:
            layer = PRICING_LAYERS.get(
                lname, build_firestore_layer(lname, DATABASE_NAME, project=PROJECT)
            )
            skus = layer.query({"-regions": [node_region], "class": [machine_class]}, union=False)
            if len(skus) != 1:
                logger.warning(
                    f"Expected 1 SKU for {lname} in {node_region}/{machine_class}, "
                    f"got {len(skus)}; skipping"
                )
                continue
            sku: DBRowDataT = next(iter(skus.values()))
            node_cost_hourly += float(str(sku["price_per_unit_usd"])) * count

        gpu_layer_name = f"GPU-{provisioning_model}"
        gpu_layer = PRICING_LAYERS.get(
            gpu_layer_name, build_firestore_layer(gpu_layer_name, DATABASE_NAME, project=PROJECT)
        )

        for gpu_indentifier, count in node_info["gpus"].items():  # type: ignore
            skus = gpu_layer.query(
                {"-regions": [node_region], "gpu_indentifier": [gpu_indentifier]}, union=False
            )
            if len(skus) != 1:
                logger.warning(
                    f"Expected 1 SKU for {gpu_layer_name} in {node_region}/{gpu_indentifier}, "
                    f"got {len(skus)}; skipping"
                )
                continue
            sku = next(iter(skus.values()))
            node_cost_hourly += float(str(sku["price_per_unit_usd"])) * count

        hourly_multiple = (float(str(node_info[run_id])) - _get_start_time(node_info)) / 3600
        total_cost += hourly_multiple * node_cost_hourly

    global _last_compute_cost  # pylint: disable=global-statement
    _last_compute_cost = total_cost
    RUN_DB[run_id] = {"compute_cost": total_cost}


def _aggregate_gcs_from_pods(pod_docs: dict) -> dict | None:
    """Aggregate GCS stats from pre-fetched per-pod docs."""
    bucket_stats: dict = defaultdict(
        lambda: {
            "class_a_count": 0,
            "class_b_count": 0,
            "egress_bytes": 0,
            "operations": defaultdict(int),
            "region_match": None,
        }
    )
    pod_count = 0
    for pod_stats in pod_docs.values():
        if not isinstance(pod_stats, dict):
            continue
        buckets_data = pod_stats.get("buckets", {})
        if not buckets_data:
            continue
        for bucket, stats in buckets_data.items():
            if not isinstance(stats, dict):
                continue
            agg_bucket = bucket_stats[bucket]
            agg_bucket["class_a_count"] += stats.get("class_a_count", 0)
            agg_bucket["class_b_count"] += stats.get("class_b_count", 0)
            agg_bucket["egress_bytes"] += stats.get("egress_bytes", 0)
            for op, count in stats.get("operations", {}).items():
                agg_bucket["operations"][op] += count
            if agg_bucket["region_match"] is None:
                agg_bucket["region_match"] = stats.get("region_match")
        pod_count += 1

    if pod_count == 0:
        return None

    aggregated: dict = {
        "buckets": {
            bucket: {**data, "operations": dict(data["operations"])}
            for bucket, data in bucket_stats.items()
        },
        "pod_count": pod_count,
    }
    aggregated["total_class_a"] = sum(b["class_a_count"] for b in aggregated["buckets"].values())
    aggregated["total_class_b"] = sum(b["class_b_count"] for b in aggregated["buckets"].values())
    aggregated["total_egress_bytes"] = sum(
        b["egress_bytes"] for b in aggregated["buckets"].values() if b.get("region_match") is False
    )
    egress_gib = aggregated["total_egress_bytes"] / BYTES_PER_GIB
    aggregated["egress_cost_min"] = egress_gib * EGRESS_COST_PER_GIB_MIN
    aggregated["egress_cost_max"] = egress_gib * EGRESS_COST_PER_GIB_MAX
    return aggregated


def _aggregate_semaphore_from_pods(
    pod_docs: dict, semaphore_widths: dict[str, int] | None = None
) -> dict | None:
    """Aggregate semaphore timing across pods, with per-worker-type breakdown.

    Output schema:
        {
            "per_type": {
                <sema>: {total_wait_time, total_lease_time, total_acquisitions,
                         avg_wait, avg_lease, [utilization_pct]}
            },
            "bottleneck": <sema with highest wait fraction>,
            "per_worker_type": {<worker>: {per_type, bottleneck}, ...},
        }
    """

    def _empty_accum() -> dict:
        return defaultdict(
            lambda: {
                "total_wait_time": 0.0,
                "total_lease_time": 0.0,
                "lease_count": 0,
                "min_start_time": float("inf"),
            }
        )

    fleet_accum = _empty_accum()
    per_worker_accum: dict[str, dict] = defaultdict(_empty_accum)

    for pod_stats in pod_docs.values():
        if not isinstance(pod_stats, dict):
            continue
        sema = pod_stats.get("semaphore_stats")
        if not isinstance(sema, dict) or not sema:
            continue
        worker_type = pod_stats.get("worker_type", "unspecified")
        for sema_type, fields in sema.items():
            if not isinstance(fields, dict):
                continue
            for accum in (fleet_accum, per_worker_accum[worker_type]):
                entry = accum[sema_type]
                entry["total_wait_time"] += fields.get("total_wait_time", 0.0)
                entry["total_lease_time"] += fields.get("total_lease_time", 0.0)
                entry["lease_count"] += fields.get("lease_count", 0)
                start = fields.get("start_time")
                if isinstance(start, (int, float)) and start < entry["min_start_time"]:
                    entry["min_start_time"] = start

    if not fleet_accum:
        return None

    def _finalize(accum: dict) -> dict:
        per_type: dict = {}
        now = time.time()
        for sema_type, vals in accum.items():
            count = vals["lease_count"]
            entry = {
                "total_wait_time": vals["total_wait_time"],
                "total_lease_time": vals["total_lease_time"],
                "total_acquisitions": count,
                "avg_wait": vals["total_wait_time"] / count if count else 0.0,
                "avg_lease": vals["total_lease_time"] / count if count else 0.0,
            }
            if (
                semaphore_widths
                and sema_type in semaphore_widths
                and semaphore_widths[sema_type] > 0
                and vals["min_start_time"] != float("inf")
            ):
                runtime = max(now - vals["min_start_time"], 1.0)
                entry["utilization_pct"] = (
                    vals["total_lease_time"] / (semaphore_widths[sema_type] * runtime) * 100.0
                )
            per_type[sema_type] = entry

        bottleneck: str | None = None
        if per_type:
            bottleneck = max(
                per_type.keys(),
                key=lambda t: (
                    per_type[t]["total_wait_time"]
                    / (per_type[t]["total_wait_time"] + per_type[t]["total_lease_time"])
                    if (per_type[t]["total_wait_time"] + per_type[t]["total_lease_time"]) > 0
                    else 0.0
                ),
            )
        return {"per_type": per_type, "bottleneck": bottleneck}

    result = _finalize(fleet_accum)
    result["per_worker_type"] = {wt: _finalize(accum) for wt, accum in per_worker_accum.items()}
    return result


def _aggregate_resource_from_pods(pod_docs: dict) -> dict | None:
    """Per-pod compact summary + fleet rollup + per-worker-type rollups."""
    per_pod: dict[str, dict] = {}
    per_worker_entries: dict[str, list[dict]] = defaultdict(list)

    for doc_key, pod_stats in pod_docs.items():
        if not isinstance(pod_stats, dict):
            continue
        res = pod_stats.get("resource_stats")
        if not isinstance(res, dict) or not res:
            continue
        cpu = res.get("cpu", {}) if isinstance(res.get("cpu"), dict) else {}
        mem = res.get("memory", {}) if isinstance(res.get("memory"), dict) else {}
        disk = res.get("disk_io", {}) if isinstance(res.get("disk_io"), dict) else {}
        net = res.get("network", {}) if isinstance(res.get("network"), dict) else {}

        compact = {
            "cpu_avg_percent": cpu.get("avg_percent", 0.0),
            "cpu_max_percent": cpu.get("max_percent", 0.0),
            "mem_max_percent": mem.get("max_percent", 0.0),
            "mem_max_used_gb": mem.get("max_used_gb", 0.0),
            "disk_read_gb": disk.get("total_read_gb", 0.0),
            "disk_write_gb": disk.get("total_write_gb", 0.0),
            "net_sent_gb": net.get("total_bytes_sent_gb", 0.0),
            "net_recv_gb": net.get("total_bytes_recv_gb", 0.0),
        }
        pod_name = doc_key.split("__", 1)[1] if "__" in doc_key else doc_key
        per_pod[pod_name] = compact
        worker_type = pod_stats.get("worker_type", "unspecified")
        per_worker_entries[worker_type].append(compact)

    if not per_pod:
        return None

    def _fleet_rollup(entries: list[dict]) -> dict:
        if not entries:
            return {}
        return {
            "avg_cpu_percent": sum(e["cpu_avg_percent"] for e in entries) / len(entries),
            "max_memory_percent": max(e["mem_max_percent"] for e in entries),
            "total_net_egress_gb": sum(e["net_sent_gb"] for e in entries),
        }

    return {
        "per_pod": per_pod,
        "fleet": _fleet_rollup(list(per_pod.values())),
        "per_worker_type": {
            wt: _fleet_rollup(entries) for wt, entries in per_worker_entries.items()
        },
    }


def aggregate_pod_stats(
    run_id: str, semaphore_widths: dict[str, int] | None = None
) -> dict | None:
    """Aggregate gcs/semaphore/resource stats from pod-stats collection.

    Single Firestore read of POD_STATS_DB, single write to RUN_DB[run_id].
    Compute cost for the log line is read from the in-process cache populated
    by compute_costs() to avoid an extra Firestore read.
    """
    pod_docs = POD_STATS_DB.query(column_filter={"run_id": [run_id]})
    if not pod_docs:
        return None

    gcs = _aggregate_gcs_from_pods(pod_docs)
    sema = _aggregate_semaphore_from_pods(pod_docs, semaphore_widths)
    resource = _aggregate_resource_from_pods(pod_docs)

    update: dict = {}
    if gcs is not None:
        update["gcs_stats"] = gcs
        # Promote egress (in GiB) to a root-level field so it's queryable
        # without descending into the nested gcs_stats map.
        update["total_egress_gib"] = gcs["total_egress_bytes"] / BYTES_PER_GIB
    if sema is not None:
        update["semaphore_stats"] = sema
    if resource is not None:
        update["resource_stats"] = resource
    if not update:
        return None

    RUN_DB[run_id] = update

    if gcs is not None:
        bottleneck = sema.get("bottleneck") if sema else None
        fleet = resource.get("fleet", {}) if resource else {}
        _current = (
            gcs["total_class_a"],
            gcs["total_class_b"],
            gcs["total_egress_bytes"],
            _last_compute_cost,
            bottleneck,
        )
        if any(_current) and _current != _last_gcs_stats.get(run_id):
            _last_gcs_stats[run_id] = _current
            cost_str = (
                f"estimated compute=${_last_compute_cost:.2f}, "
                if _last_compute_cost is not None
                else ""
            )
            extras = []
            if bottleneck:
                extras.append(f"bottleneck={bottleneck}")
            if "avg_cpu_percent" in fleet:
                extras.append(f"cpu={fleet['avg_cpu_percent']:.0f}%")
            if "max_memory_percent" in fleet:
                extras.append(f"mem_max={fleet['max_memory_percent']:.0f}%")
            extras_str = " | " + " ".join(extras) if extras else ""
            logger.info(
                f"{cost_str}"
                f"gcs stats: A={gcs['total_class_a']} "
                f"B={gcs['total_class_b']} egress={update['total_egress_gib']:.2f} GiB "
                f"(${gcs['egress_cost_min']:.2f}-${gcs['egress_cost_max']:.2f}) "
                f"from {gcs['pod_count']} pods, {len(gcs['buckets'])} buckets"
                f"{extras_str}"
            )

    return update


def update_compute_pricing_db(groups: dict):
    for group, items in groups.items():
        logger.info(f"Updating group {group}: {len(items)} entries.")
        collection = build_firestore_layer(group, DATABASE_NAME, project=PROJECT)
        row_keys = []
        row_values = []
        columns = set()
        for item in items:
            _item = copy(item)
            row_keys.append(_item.pop("sku_id"))
            _item["regions"] = list(_item["regions"])
            row_values.append(_item)
            columns.update(item.keys())
        idx_user = (row_keys, tuple(columns))
        collection[idx_user] = row_values


if __name__ == "__main__":  # pragma: no cover
    logger.setLevel(logging.INFO)
    sku_groups = gcloud.get_compute_sku_groups()
    update_compute_pricing_db(sku_groups)
