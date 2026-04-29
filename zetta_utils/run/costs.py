import logging
import time
from collections import defaultdict
from copy import copy
from typing import Final

from zetta_utils.cloud_management.resource_allocation import gcloud
from zetta_utils.common.pprint import lrpad
from zetta_utils.layer.db_layer.backend import DBRowDataT
from zetta_utils.layer.db_layer.firestore import build_firestore_layer
from zetta_utils.layer.db_layer.layer import DBLayer
from zetta_utils.log import get_logger

from .db import NODE_DB, POD_STATS_DB, RUN_DB

PROJECT = "zetta-research"
DATABASE_NAME = "pricing-db"

logger = get_logger("zetta_utils")

PRICING_LAYERS: dict[str, DBLayer] = {}
# Cross-tick SKU price cache, keyed by (layer_name, region, key). Pricing is
# static for the lifetime of a run, so caching across compute_costs ticks
# eliminates redundant Firestore queries entirely on the second+ tick. None
# means "missing or ambiguous SKU"; cached to suppress re-warn spam.
_SKU_PRICE_CACHE: dict[tuple[str, str, str], float | None] = {}
PROVISIONING_MAP: Final[dict[str, str]] = {"PREEMPTIBLE": "preemptible", "STANDARD": "ondemand"}
EGRESS_COST_PER_GIB_MIN: Final[float] = 0.02
EGRESS_COST_PER_GIB_MAX: Final[float] = 0.18
BYTES_PER_GIB: Final[int] = 1024 ** 3
_last_gcs_stats: dict[str, tuple] = {}
# Cached compute cost for the active run, populated by compute_costs() and read
# by aggregate_pod_stats() for log enrichment without an extra Firestore read.
# Only one run_id is active per process invocation.
_last_compute_cost: float | None = None
_last_compute_cost_by_wt: dict[str, dict[str, float]] | None = None
_cached_semaphore_widths: dict[str, int] | None = None


def compute_costs(run_id: str):  # pylint: disable=too-many-locals
    run_timestamp = RUN_DB[run_id]["timestamp"]

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
                return run_timestamp
        return float(str(node_info["creation_time"]))

    # Tight timeout: this runs on a RepeatTimer thread every 60s; the Firestore
    # SDK's default 300s deadline would block the timer thread for 5 minutes on
    # a transient connectivity blip and delay both subsequent ticks and run
    # shutdown.
    nodes = NODE_DB.query({"-run_id": [run_id]}, timeout=30.0)

    def _get_sku_price(layer_name: str, region: str, key: str, key_field: str) -> float | None:
        cache_key = (layer_name, region, key)
        if cache_key in _SKU_PRICE_CACHE:
            return _SKU_PRICE_CACHE[cache_key]
        layer = PRICING_LAYERS.get(layer_name)
        if layer is None:
            layer = build_firestore_layer(layer_name, DATABASE_NAME, project=PROJECT)
            PRICING_LAYERS[layer_name] = layer
        # GCP's SKU API is messy — there's no standard way to look up pricing
        # by machine type. Some fields are parsed from description strings. A
        # mismatch can happen when the pricing DB hasn't been refreshed for a
        # new machine type/region, or when a node reports a machine_type that
        # doesn't match the pricing DB's classification.
        skus = layer.query({"-regions": [region], key_field: [key]}, union=False)
        if len(skus) != 1:
            logger.warning(
                f"Expected 1 SKU for {layer_name} in {region}/{key}, " f"got {len(skus)}; skipping"
            )
            _SKU_PRICE_CACHE[cache_key] = None
            return None
        sku: DBRowDataT = next(iter(skus.values()))
        price = float(str(sku["price_per_unit_usd"]))
        _SKU_PRICE_CACHE[cache_key] = price
        return price

    total_cost = 0.0
    per_wt: dict[str, dict[str, float]] = defaultdict(
        lambda: {"cpu_mem": 0.0, "gpu": 0.0, "node_hours": 0.0}
    )
    for _info in nodes.values():
        node_cpu_mem_hourly = 0.0
        node_gpu_hourly = 0.0
        node_info: DBRowDataT = _info
        machine_class, _, _ = str(node_info["machine_type"]).split("-")
        node_region = str(node_info["region"])
        provisioning_model = PROVISIONING_MAP[str(node_info["provisioning_model"])]
        wt = str(node_info.get("worker_type", "unspecified"))

        cpu = (f"CPU-{provisioning_model}", float(str(node_info["cpu_count"])))
        ram = (f"RAM-{provisioning_model}", float(str(node_info["memory_gib"])))

        for lname, count in [cpu, ram]:
            price = _get_sku_price(lname, node_region, machine_class, "class")
            if price is not None:
                node_cpu_mem_hourly += price * count

        gpu_layer_name = f"GPU-{provisioning_model}"
        for gpu_indentifier, count in node_info["gpus"].items():  # type: ignore
            price = _get_sku_price(
                gpu_layer_name, node_region, str(gpu_indentifier), "gpu_indentifier"
            )
            if price is not None:
                node_gpu_hourly += price * count

        hourly_multiple = (float(str(node_info[run_id])) - _get_start_time(node_info)) / 3600
        per_wt[wt]["cpu_mem"] += hourly_multiple * node_cpu_mem_hourly
        per_wt[wt]["gpu"] += hourly_multiple * node_gpu_hourly
        per_wt[wt]["node_hours"] += hourly_multiple
        total_cost += hourly_multiple * (node_cpu_mem_hourly + node_gpu_hourly)

    global _last_compute_cost  # pylint: disable=global-statement
    _last_compute_cost = total_cost
    global _last_compute_cost_by_wt  # pylint: disable=global-statement
    _last_compute_cost_by_wt = dict(per_wt)
    update: dict = {"compute_cost": total_cost}
    if per_wt:
        update["compute_cost_by_worker_type"] = {
            wt: {
                "cpu_mem": costs["cpu_mem"],
                "gpu": costs["gpu"],
                "node_hours": costs["node_hours"],
            }
            for wt, costs in per_wt.items()
        }
    RUN_DB[run_id] = update


def _billed_egress_bytes(bucket_stats: dict) -> int:
    """Sum egress bytes from cross-region buckets only — what GCP charges for.
    Same-region transfer is free so it's not egress for billing purposes.
    """
    return sum(b["egress_bytes"] for b in bucket_stats.values() if b.get("region_match") is False)


def _gcs_totals(bucket_stats: dict) -> dict:
    """Compute fleet-level GCS totals from a bucket_stats dict."""
    total_a = sum(b["class_a_count"] for b in bucket_stats.values())
    total_b = sum(b["class_b_count"] for b in bucket_stats.values())
    total_egress = _billed_egress_bytes(bucket_stats)
    egress_gib = total_egress / BYTES_PER_GIB
    return {
        "total_class_a": total_a,
        "total_class_b": total_b,
        "total_egress_bytes": total_egress,
        "egress_cost_min": egress_gib * EGRESS_COST_PER_GIB_MIN,
        "egress_cost_max": egress_gib * EGRESS_COST_PER_GIB_MAX,
    }


def _aggregate_gcs_from_pods(pod_docs: dict) -> dict | None:  # pylint: disable=too-many-branches
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
    per_wt_buckets: dict[str, dict] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "class_a_count": 0,
                "class_b_count": 0,
                "egress_bytes": 0,
                "region_match": None,
            }
        )
    )
    pod_count = 0
    for pod_stats in pod_docs.values():
        if not isinstance(pod_stats, dict):
            continue
        buckets_data = pod_stats.get("buckets", {})
        if not buckets_data:
            continue
        worker_type = pod_stats.get("worker_type", "unspecified")
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
            # Per-worker-type accumulation
            wt_bucket = per_wt_buckets[worker_type][bucket]
            wt_bucket["class_a_count"] += stats.get("class_a_count", 0)
            wt_bucket["class_b_count"] += stats.get("class_b_count", 0)
            wt_bucket["egress_bytes"] += stats.get("egress_bytes", 0)
            if wt_bucket["region_match"] is None:
                wt_bucket["region_match"] = stats.get("region_match")
        pod_count += 1

    if pod_count == 0:
        return None

    # Zero out per-bucket egress_bytes for same-region buckets so "egress"
    # consistently means billed egress everywhere. Raw bytes are retained on
    # the per-pod Firestore docs for ad-hoc bandwidth inspection.
    for data in bucket_stats.values():
        if data.get("region_match") is not False:
            data["egress_bytes"] = 0
    for wt_buckets in per_wt_buckets.values():
        for data in wt_buckets.values():
            if data.get("region_match") is not False:
                data["egress_bytes"] = 0

    aggregated: dict = {
        "buckets": {
            bucket: {**data, "operations": dict(data["operations"])}
            for bucket, data in bucket_stats.items()
        },
        "pod_count": pod_count,
    }
    aggregated.update(_gcs_totals(aggregated["buckets"]))
    aggregated["per_worker_type"] = {
        wt: _gcs_totals(buckets) for wt, buckets in per_wt_buckets.items()
    }
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

        # Aggregate GPU stats across all devices on this pod
        gpus = res.get("gpus", {}) if isinstance(res.get("gpus"), dict) else {}
        if gpus:
            gpu_vals = list(gpus.values())
            gpu_avg_util = sum(g.get("avg_utilization_percent", 0) for g in gpu_vals) / len(
                gpu_vals
            )
            gpu_max_util = max(g.get("max_utilization_percent", 0) for g in gpu_vals)
            gpu_total_mem = sum(g.get("memory_total_gib", 0) for g in gpu_vals)
            if gpu_total_mem > 0:
                gpu_avg_mem_pct = (
                    sum(g.get("avg_memory_used_gib", 0) for g in gpu_vals) / gpu_total_mem * 100
                )
                gpu_max_mem_pct = (
                    max(g.get("max_memory_used_gib", 0) for g in gpu_vals)
                    / (max(g.get("memory_total_gib", 0) for g in gpu_vals) or 1)
                    * 100
                )
            else:
                gpu_avg_mem_pct = 0.0
                gpu_max_mem_pct = 0.0
        else:
            gpu_avg_util = None
            gpu_max_util = None
            gpu_avg_mem_pct = None
            gpu_max_mem_pct = None

        compact = {
            "cpu_avg_percent": cpu.get("avg_percent", 0.0),
            "cpu_max_percent": cpu.get("max_percent", 0.0),
            "mem_avg_percent": mem.get("avg_percent", 0.0),
            "mem_max_percent": mem.get("max_percent", 0.0),
            "mem_max_used_gib": mem.get("max_used_gib", 0.0),
            "gpu_avg_util_percent": gpu_avg_util,
            "gpu_max_util_percent": gpu_max_util,
            "gpu_avg_mem_percent": gpu_avg_mem_pct,
            "gpu_max_mem_percent": gpu_max_mem_pct,
            "disk_read_gib": disk.get("total_read_gib", 0.0),
            "disk_write_gib": disk.get("total_write_gib", 0.0),
            "net_sent_gib": net.get("total_bytes_sent_gib", 0.0),
            "net_recv_gib": net.get("total_bytes_recv_gib", 0.0),
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
        result = {
            "avg_cpu_percent": sum(e["cpu_avg_percent"] for e in entries) / len(entries),
            "max_cpu_percent": max(e["cpu_max_percent"] for e in entries),
            "avg_memory_percent": sum(e["mem_avg_percent"] for e in entries) / len(entries),
            "max_memory_percent": max(e["mem_max_percent"] for e in entries),
            "total_net_ingress_gib": sum(e["net_recv_gib"] for e in entries),
            "total_net_egress_gib": sum(e["net_sent_gib"] for e in entries),
        }
        gpu_entries = [e for e in entries if e.get("gpu_avg_util_percent") is not None]
        if gpu_entries:
            result["avg_gpu_util_percent"] = sum(
                e["gpu_avg_util_percent"] for e in gpu_entries
            ) / len(gpu_entries)
            result["max_gpu_util_percent"] = max(e["gpu_max_util_percent"] for e in gpu_entries)
            result["avg_gpu_mem_percent"] = sum(
                e["gpu_avg_mem_percent"] for e in gpu_entries
            ) / len(gpu_entries)
            result["max_gpu_mem_percent"] = max(e["gpu_max_mem_percent"] for e in gpu_entries)
        return result

    return {
        "per_pod": per_pod,
        "fleet": _fleet_rollup(list(per_pod.values())),
        "per_worker_type": {
            wt: _fleet_rollup(entries) for wt, entries in per_worker_entries.items()
        },
    }


def _log_worker_type_table(  # pragma: no cover  # pylint: disable=too-many-locals,too-many-statements
    sema: dict | None,
    resource: dict | None,
    gcs: dict | None = None,
) -> None:
    """Log a per-worker-type table of costs, bottleneck, CPU, memory, GPU, and network stats."""
    sema_by_wt = sema.get("per_worker_type", {}) if sema else {}
    res_by_wt = resource.get("per_worker_type", {}) if resource else {}
    cost_by_wt = _last_compute_cost_by_wt or {}
    gcs_by_wt = gcs.get("per_worker_type", {}) if gcs else {}
    worker_types = sorted(set(sema_by_wt) | set(res_by_wt) | set(cost_by_wt))
    if not worker_types:
        return

    # Column widths (logical; _col adds 1 to compensate for lrpad bounds="" quirk)
    W_WT = 18  # worker type
    W_BN = 17  # bottleneck/2nd
    W_BP = 12  # block/2nd
    W_WL = 21  # wait / lease / ut.
    W_CA = 6  # cpu avg
    W_CM = 6  # cpu max
    W_MA = 6  # mem avg
    W_MM = 6  # mem max
    W_GU = 6  # gpu util avg
    W_GX = 6  # gpu util max
    W_GA = 6  # gpu mem avg
    W_GK = 6  # gpu mem max
    W_NI = 7  # net in
    W_NO = 7  # net out
    # Group widths for top header
    W_SEMA = W_BN + W_BP + W_WL
    W_CPU = W_CA + W_CM
    W_MEM = W_MA + W_MM
    W_GPU_UTIL = W_GU + W_GX
    W_GPU_MEM = W_GA + W_GK
    W_NET = W_NI + W_NO
    # 1 leading space + columns + 2 bounds
    LEN = 1 + W_WT + W_SEMA + W_CPU + W_MEM + W_GPU_UTIL + W_GPU_MEM + W_NET + 2

    def _block_pct(entry: dict) -> float:
        tw = entry.get("total_wait_time", 0)
        tl = entry.get("total_lease_time", 0)
        return tw / (tw + tl) * 100 if (tw + tl) > 0 else 0

    def _fmt_sema(wt_sema: dict) -> tuple[str, str, str]:
        """Returns (bottleneck(2nd), block(2nd), wait_lease_ut_str)."""
        bn = wt_sema.get("bottleneck") or "-"
        per_type = wt_sema.get("per_type", {})
        if bn != "-" and bn in per_type:
            entry = per_type[bn]
            wait_pct = _block_pct(entry)
            avg_w = entry.get("avg_wait", 0)
            avg_l = entry.get("avg_lease", 0)
            util = entry.get("utilization_pct")
            wl = f"{avg_w:.2f} / {avg_l:.2f}"
            if util is not None:
                wl += f" / {util:.0f}%"
            # Find 2nd bottleneck
            others = [
                (k, _block_pct(v))
                for k, v in per_type.items()
                if k != bn and (v.get("total_wait_time", 0) + v.get("total_lease_time", 0)) > 0
            ]
            if others:
                others.sort(key=lambda x: x[1], reverse=True)
                s_name, s_pct = others[0]
                bn_str = f"{bn} / {s_name}"
                bp_str = f"{wait_pct:2.0f}% / {s_pct:2.0f}%"
            else:
                bn_str = bn
                bp_str = f"{wait_pct:2.0f}%"
            return bn_str, bp_str, wl
        return bn, "-", "-"

    def _col(text, width):
        return lrpad(text, level=0, length=width + 1, bounds="")

    def _row(*cols):
        widths = (
            W_WT,
            W_BN,
            W_BP,
            W_WL,
            W_CA,
            W_CM,
            W_MA,
            W_MM,
            W_GU,
            W_GX,
            W_GA,
            W_GK,
            W_NI,
            W_NO,
        )
        return " " + "".join(_col(c, w) for c, w in zip(cols, widths))

    s = ""
    title = "  Worker Stats by Type  "
    pad_total = LEN - 2 - len(title)  # space between + bounds
    pad_left = pad_total // 2
    s += lrpad("=" * pad_left + title, level=0, bounds="+", filler="=", length=LEN) + "\n"
    s += lrpad("", level=0, bounds="|", length=LEN) + "\n"

    # Top header — group names with (s) above Wait/Lease/Ut.
    top = (
        " "
        + _col("", W_WT)
        + _col("Semaphores", W_BN + W_BP)
        + _col("(s, bottleneck)", W_WL)
        + _col("CPU", W_CPU)
        + _col("Memory", W_MEM)
        + _col("CUDA", W_GPU_UTIL)
        + _col("CUDA Mem", W_GPU_MEM)
        + _col("Network (GiB)", W_NET)
    )
    s += lrpad(top, level=0, bounds="|", length=LEN) + "\n"

    # Bottom header — column names
    bottom = _row(
        "Worker Type",
        "Bottleneck/2nd",
        "Block/2nd",
        "Wait / Lease / Ut.",
        "Avg",
        "Max",
        "Avg",
        "Max",
        "Avg",
        "Max",
        "Avg",
        "Max",
        "In",
        "Out",
    )
    s += lrpad(bottom, level=0, bounds="|", length=LEN) + "\n"
    s += lrpad("", level=0, filler="-", bounds="|", length=LEN) + "\n"

    for wt in worker_types:
        wt_sema_data = sema_by_wt.get(wt, {})
        wt_res = res_by_wt.get(wt, {})

        bn_str, bp_str, wl_detail = _fmt_sema(wt_sema_data) if wt_sema_data else ("-", "-", "-")
        cpu_avg = f"{wt_res['avg_cpu_percent']:3.0f}%" if "avg_cpu_percent" in wt_res else "  -"
        cpu_max = f"{wt_res['max_cpu_percent']:3.0f}%" if "max_cpu_percent" in wt_res else "  -"
        mem_avg = (
            f"{wt_res['avg_memory_percent']:3.0f}%" if "avg_memory_percent" in wt_res else "  -"
        )
        mem_max = (
            f"{wt_res['max_memory_percent']:3.0f}%" if "max_memory_percent" in wt_res else "  -"
        )
        gpu_avg = (
            f"{wt_res['avg_gpu_util_percent']:3.0f}%"
            if "avg_gpu_util_percent" in wt_res
            else "  -"
        )
        gpu_max = (
            f"{wt_res['max_gpu_util_percent']:3.0f}%"
            if "max_gpu_util_percent" in wt_res
            else "  -"
        )
        gpu_mem_avg = (
            f"{wt_res['avg_gpu_mem_percent']:3.0f}%" if "avg_gpu_mem_percent" in wt_res else "  -"
        )
        gpu_mem_max = (
            f"{wt_res['max_gpu_mem_percent']:3.0f}%" if "max_gpu_mem_percent" in wt_res else "  -"
        )
        net_ig = (
            f"{wt_res['total_net_ingress_gib']:.2f}" if "total_net_ingress_gib" in wt_res else "-"
        )
        net_eg = (
            f"{wt_res['total_net_egress_gib']:.2f}" if "total_net_egress_gib" in wt_res else "-"
        )

        s += (
            lrpad(
                _row(
                    wt[:W_WT],
                    bn_str,
                    bp_str,
                    wl_detail,
                    cpu_avg,
                    cpu_max,
                    mem_avg,
                    mem_max,
                    gpu_avg,
                    gpu_max,
                    gpu_mem_avg,
                    gpu_mem_max,
                    net_ig,
                    net_eg,
                ),
                level=0,
                bounds="|",
                length=LEN,
            )
            + "\n"
        )

    s += lrpad("", level=0, bounds="|", length=LEN) + "\n"
    s += lrpad("", level=0, bounds="+", filler="=", length=LEN)

    # Cost subtable (only if we have cost data)
    if cost_by_wt or gcs_by_wt:
        W_CH = 8  # node hours
        W_CC = 10  # cpu+mem cost
        W_CG = 10  # gpu cost
        W_CT = 10  # total compute
        W_CI = 10  # ideal compute
        W_EA = 10  # gcs class A ops
        W_EB = 10  # gcs class B ops
        W_EG = 12  # egress cost range
        W_COMPUTE = W_CH + W_CC + W_CG + W_CT + W_CI
        W_GCS = W_EA + W_EB + W_EG
        LEN2 = 1 + W_WT + W_COMPUTE + W_GCS + 2

        def _col2(text, width):
            return lrpad(text, level=0, length=width + 1, bounds="")

        def _row2(*cols):
            widths2 = (W_WT, W_CH, W_CC, W_CG, W_CT, W_CI, W_EA, W_EB, W_EG)
            return " " + "".join(_col2(c, w) for c, w in zip(cols, widths2))

        s += "\n"
        title2 = "  Cost Breakdown by Worker Type  "
        pad2 = (LEN2 - 2 - len(title2)) // 2
        s += lrpad("=" * pad2 + title2, level=0, bounds="+", filler="=", length=LEN2) + "\n"
        s += lrpad("", level=0, bounds="|", length=LEN2) + "\n"
        top2 = " " + _col2("", W_WT) + _col2("Compute ($)", W_COMPUTE) + _col2("GCS", W_GCS)
        s += lrpad(top2, level=0, bounds="|", length=LEN2) + "\n"
        bottom2 = _row2(
            "Worker Type",
            "Hours",
            "CPU+Mem",
            "GPU",
            "Total",
            "Ideal*",
            "Class A",
            "Class B",
            "Egress ($)",
        )
        s += lrpad(bottom2, level=0, bounds="|", length=LEN2) + "\n"
        s += lrpad("", level=0, filler="-", bounds="|", length=LEN2) + "\n"

        all_wts = sorted(set(cost_by_wt) | set(gcs_by_wt))
        total_hrs = 0.0
        total_cm = 0.0
        total_gpu = 0.0
        total_ideal = 0.0
        total_ea = 0
        total_eb = 0
        total_eg_min = 0.0
        total_eg_max = 0.0
        for wt in all_wts:
            wt_cost = cost_by_wt.get(wt, {})
            wt_gcs = gcs_by_wt.get(wt, {})
            wt_res_data = res_by_wt.get(wt, {})
            hrs = wt_cost.get("node_hours", 0.0)
            cm = wt_cost.get("cpu_mem", 0.0)
            gpu = wt_cost.get("gpu", 0.0)
            # Ideal: scale by utilization bottleneck
            cpu_pct = wt_res_data.get("avg_cpu_percent", 100.0)
            mem_pct = wt_res_data.get("avg_memory_percent", 100.0)
            gpu_pct = wt_res_data.get("avg_gpu_util_percent", 100.0)
            ideal_cm = cm * max(cpu_pct, mem_pct, 1.0) / 100.0
            ideal_gpu = gpu * max(gpu_pct, 1.0) / 100.0
            ideal = ideal_cm + ideal_gpu
            ea = wt_gcs.get("total_class_a", 0)
            eb = wt_gcs.get("total_class_b", 0)
            eg_min = wt_gcs.get("egress_cost_min", 0.0)
            eg_max = wt_gcs.get("egress_cost_max", 0.0)
            total_hrs += hrs
            total_cm += cm
            total_gpu += gpu
            total_ideal += ideal
            total_ea += ea
            total_eb += eb
            total_eg_min += eg_min
            total_eg_max += eg_max
            eg_str = f"{eg_min:.2f}-{eg_max:.2f}" if (eg_min or eg_max) else "-"
            s += (
                lrpad(
                    _row2(
                        wt[:W_WT],
                        f"{hrs:.1f}" if hrs else "-",
                        f"{cm:.2f}" if cm else "-",
                        f"{gpu:.2f}" if gpu else "-",
                        f"{cm + gpu:.2f}" if (cm or gpu) else "-",
                        f"{ideal:.2f}*" if (cm or gpu) else "-",
                        f"{ea:,}" if ea else "-",
                        f"{eb:,}" if eb else "-",
                        eg_str,
                    ),
                    level=0,
                    bounds="|",
                    length=LEN2,
                )
                + "\n"
            )

        # Totals row
        s += lrpad("", level=0, filler="-", bounds="|", length=LEN2) + "\n"
        eg_total_str = (
            f"{total_eg_min:.2f}-{total_eg_max:.2f}" if (total_eg_min or total_eg_max) else "-"
        )
        s += (
            lrpad(
                _row2(
                    "TOTAL",
                    f"{total_hrs:.1f}",
                    f"{total_cm:.2f}",
                    f"{total_gpu:.2f}",
                    f"{total_cm + total_gpu:.2f}",
                    f"{total_ideal:.2f}*",
                    f"{total_ea:,}",
                    f"{total_eb:,}",
                    eg_total_str,
                ),
                level=0,
                bounds="|",
                length=LEN2,
            )
            + "\n"
        )
        s += lrpad("", level=0, bounds="|", length=LEN2) + "\n"
        s += lrpad("", level=0, bounds="+", filler="=", length=LEN2)

    logger.info(s)


def _format_compute_cost(cost: float | None) -> str:
    return f"compute=${cost:.2f}" if cost is not None else "compute=unknown"


def _format_gcs_stats(gcs: dict, total_egress_gib: float) -> str | None:
    """One-line gcs summary, or ``None`` when there is no gcs activity."""
    if gcs["pod_count"] <= 0:
        return None
    if not (gcs["total_class_a"] or gcs["total_class_b"] or gcs["total_egress_bytes"]):
        return None
    return (
        f"gcs A={gcs['total_class_a']} B={gcs['total_class_b']} "
        f"egress={total_egress_gib:.2f} GiB "
        f"(${gcs['egress_cost_min']:.2f}-${gcs['egress_cost_max']:.2f}) "
        f"from {gcs['pod_count']} pods, {len(gcs['buckets'])} buckets"
    )


def aggregate_pod_stats(
    run_id: str, semaphore_widths: dict[str, int] | None = None
) -> dict | None:
    """Aggregate gcs/semaphore/resource stats from pod-stats collection.

    Single Firestore read of POD_STATS_DB, single write to RUN_DB[run_id].
    Compute cost for the log line is read from the in-process cache populated
    by compute_costs() to avoid an extra Firestore read.
    """
    global _cached_semaphore_widths  # pylint: disable=global-statement
    if semaphore_widths is None and _cached_semaphore_widths is None:
        try:
            run_doc = RUN_DB[run_id]
            _cached_semaphore_widths = run_doc.get("semaphore_widths")
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    if semaphore_widths is None:
        semaphore_widths = _cached_semaphore_widths

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
        _current = (
            gcs["total_class_a"],
            gcs["total_class_b"],
            gcs["total_egress_bytes"],
            _last_compute_cost,
            bottleneck,
        )
        if any(_current) and _current != _last_gcs_stats.get(run_id):
            _last_gcs_stats[run_id] = _current
            parts = [_format_compute_cost(_last_compute_cost)]
            gcs_part = _format_gcs_stats(gcs, update["total_egress_gib"])
            if gcs_part:
                parts.append(gcs_part)
            logger.info("run cost: " + "; ".join(parts))
            _log_worker_type_table(sema, resource, gcs)

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
