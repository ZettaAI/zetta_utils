import logging
from collections import defaultdict
from copy import copy
from typing import Final

from zetta_utils.cloud_management.resource_allocation import gcloud
from zetta_utils.layer.db_layer.backend import DBRowDataT
from zetta_utils.layer.db_layer.firestore import build_firestore_layer
from zetta_utils.layer.db_layer.layer import DBLayer
from zetta_utils.log import get_logger

from .db import NODE_DB, RUN_DB

PROJECT = "zetta-research"
DATABASE_NAME = "pricing-db"

logger = get_logger("zetta_utils")

PRICING_LAYERS: dict[str, DBLayer] = {}
PROVISIONING_MAP: Final[dict[str, str]] = {"PREEMPTIBLE": "preemptible", "STANDARD": "ondemand"}
EGRESS_COST_PER_GIB_MIN: Final[float] = 0.02
EGRESS_COST_PER_GIB_MAX: Final[float] = 0.18
BYTES_PER_GIB: Final[int] = 1024**3


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

    nodes = NODE_DB.query({"-run_id": [run_id]})
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
            assert len(skus) == 1, (lname, node_region, machine_class, skus)
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
            assert len(skus) == 1, (gpu_layer_name, node_region, gpu_indentifier, skus)
            sku = next(iter(skus.values()))
            node_cost_hourly += float(str(sku["price_per_unit_usd"])) * count

        hourly_multiple = (float(str(node_info[run_id])) - _get_start_time(node_info)) / 3600
        total_cost += hourly_multiple * node_cost_hourly

    RUN_DB[run_id] = {"compute_cost": total_cost}


def aggregate_gcs_stats(run_id: str) -> dict | None:
    """
    Aggregate GCS stats from all pods for a run.

    Aggregates per-bucket stats from gcs_stats_proxy (per-pod stats),
    merges operation counts per bucket, writes result to gcs_stats field.
    Only counts egress bytes from buckets where region_match is False (cross-region).
    """
    full_doc = RUN_DB[run_id]
    stats_map = full_doc.get("gcs_stats_proxy")
    if not isinstance(stats_map, dict):
        return None

    # Per-bucket aggregation
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

    for pod_stats in stats_map.values():
        if not isinstance(pod_stats, dict):
            continue
        buckets_data = pod_stats.get("buckets", {})
        for bucket, stats in buckets_data.items():
            if not isinstance(stats, dict):
                continue
            agg_bucket = bucket_stats[bucket]
            agg_bucket["class_a_count"] += stats.get("class_a_count", 0)
            agg_bucket["class_b_count"] += stats.get("class_b_count", 0)
            agg_bucket["egress_bytes"] += stats.get("egress_bytes", 0)
            for op, count in stats.get("operations", {}).items():
                agg_bucket["operations"][op] += count
            # Preserve region_match from first pod that reports it
            if agg_bucket["region_match"] is None:
                agg_bucket["region_match"] = stats.get("region_match")
        pod_count += 1

    if pod_count == 0:
        return None

    # Convert to regular dicts
    aggregated: dict = {
        "buckets": {
            bucket: {**data, "operations": dict(data["operations"])}
            for bucket, data in bucket_stats.items()
        },
        "pod_count": pod_count,
    }

    # Add totals for convenience
    aggregated["total_class_a"] = sum(b["class_a_count"] for b in aggregated["buckets"].values())
    aggregated["total_class_b"] = sum(b["class_b_count"] for b in aggregated["buckets"].values())
    # Only count egress from buckets where region_match is False (cross-region = billable)
    aggregated["total_egress_bytes"] = sum(
        b["egress_bytes"] for b in aggregated["buckets"].values() if b.get("region_match") is False
    )

    # Calculate egress cost range (GCP egress pricing varies by destination)
    egress_gib = aggregated["total_egress_bytes"] / BYTES_PER_GIB
    aggregated["egress_cost_min"] = egress_gib * EGRESS_COST_PER_GIB_MIN
    aggregated["egress_cost_max"] = egress_gib * EGRESS_COST_PER_GIB_MAX

    RUN_DB[run_id] = {"gcs_stats": aggregated}
    if (
        aggregated["total_class_a"]
        or aggregated["total_class_b"]
        or aggregated["total_egress_bytes"]
    ):
        logger.info(
            f"GCS stats: A={aggregated['total_class_a']} "
            f"B={aggregated['total_class_b']} egress={aggregated['total_egress_bytes']} bytes "
            f"(${aggregated['egress_cost_min']:.2f}-${aggregated['egress_cost_max']:.2f}) "
            f"from {pod_count} pods, {len(aggregated['buckets'])} buckets"
        )
    return aggregated


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
