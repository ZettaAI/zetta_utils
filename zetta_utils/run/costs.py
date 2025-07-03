import logging
from copy import copy

from zetta_utils.cloud_management.resource_allocation import gcloud
from zetta_utils.layer.db_layer.firestore import build_firestore_layer
from zetta_utils.log import get_logger

PROJECT = "zetta-research"
DATABASE_NAME = "pricing-db"

logger = get_logger("zetta_utils")


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
