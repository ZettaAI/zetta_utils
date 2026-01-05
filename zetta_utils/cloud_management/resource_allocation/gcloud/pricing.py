# pylint: disable=no-name-in-module
"""
Helpers to interact with GCloud pricing APIs.
"""

import re
from collections import defaultdict

from google.cloud import billing_v1


def _group_skus() -> dict:
    client = billing_v1.CloudCatalogClient()
    parent = "services/6F81-5844-456A"
    skus = client.list_skus(parent=parent)
    groups = defaultdict(list)
    for sku in skus:
        desc = sku.description.lower()
        usage_type = sku.category.usage_type.lower()
        rgroup = sku.category.resource_group
        try:
            name = sku.name
            sku_id = sku.sku_id
            description = sku.description
            service_regions = sku.service_regions
            pricing = sku.pricing_info[0].pricing_expression

            price = pricing.tiered_rates[0].unit_price
            nanos = price.nanos
            units = price.units
            price_micros = nanos / 1e9 + units
        except Exception:  # pylint: disable=broad-exception-caught
            price_micros = 0

        words = re.findall(r"\b\w+\b", description)
        possible_classes = [
            word for word in words if re.search(r"\d", word) and re.search(r"[A-Z]", word)
        ]

        _sku = {
            "name": name,
            "sku_id": sku_id,
            "description": description,
            "regions": service_regions,
            "usageUnit": pricing.usage_unit,
            "price_per_unit_usd": price_micros,
        }

        if len(possible_classes):
            _sku["class"] = possible_classes[0].lower()

        if rgroup == "N1Standard":
            if "core" in desc:
                rgroup = "CPU"
            elif "ram" in desc:
                rgroup = "RAM"

        if "custom" in desc:
            groups[f"{rgroup}-{usage_type}-custom"].append(_sku)
        elif "dws" in desc:
            groups[f"{rgroup}-{usage_type}-dws"].append(_sku)
        elif "committment" in desc:
            groups[f"{rgroup}-{usage_type}-committment"].append(_sku)
        elif "sole tenancy" in desc:
            groups[f"{rgroup}-{usage_type}-sole-tenancy"].append(_sku)
        elif "reserved" in desc:
            groups[f"{rgroup}-{usage_type}-reserved"].append(_sku)
        else:
            groups[f"{rgroup}-{usage_type}"].append(_sku)
    return groups


def _filter_relevant_groups(groups: dict) -> dict:
    others = set()
    filtered_groups = {}
    for group, items in groups.items():
        f = 0
        for k in ["cpu", "gpu", "ram", "n1standard", "tpu"]:
            if k in group.lower():
                f = 1
                filtered_groups[group] = items
        if not f:
            others.add(group)

    filtered_groups.pop("DaceITLLCd/b/aSenseTrafficPulse-ondemand", None)
    return filtered_groups


def _add_gpu_identifier(groups: dict) -> dict:
    gpu_type_identifiers = [
        "nvidia-h100-mega-80gb",
        "nvidia-h100-80gb",
        "nvidia-h200-141gb",
        "nvidia-a100-80gb",
        "nvidia-tesla-a100",
        "nvidia-gb200",
        "nvidia-b200",
        "nvidia-l4-vws",
        "nvidia-l4",
        "nvidia-tesla-t4-vws",
        "nvidia-tesla-t4",
        "nvidia-tesla-v100",
        "nvidia-tesla-p100-vws",
        "nvidia-tesla-p100",
        "nvidia-tesla-p4-vws",
        "nvidia-tesla-p4",
    ]

    for sku in groups["GPU-ondemand"] + groups["GPU-preemptible"]:
        desc = sku["description"].lower()
        assert "gpu" in desc, sku["description"]

        for identifier in gpu_type_identifiers:
            if "gpu_indentifier" in sku:
                raise ValueError()

            parts = identifier.split("-")
            matched = True
            for part in parts:
                if part == "nvidia":
                    continue
                matched = matched and part in desc
            if matched:
                sku["gpu_indentifier"] = identifier
                break
    return groups


def get_compute_sku_groups() -> dict:
    """
    Gets Compute Engine SKUs groups by types - CPU, GPU, RAM, TPU etc.
    And provisioning model - ondemand vs preemptible.
    """
    groups = _group_skus()
    groups = _filter_relevant_groups(groups)
    return _add_gpu_identifier(groups)
