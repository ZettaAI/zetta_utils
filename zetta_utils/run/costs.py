import logging
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
