"""
GCS request tracker sidecar.

Tracks and classifies GCS (Google Cloud Storage) requests into Class A and Class B
operations as defined by GCP pricing:

Class A (higher cost): Object create/delete, copy, compose, rewrite, list operations
Class B (lower cost): Object read, metadata read

Uses mitmproxy to intercept HTTPS traffic to storage.googleapis.com.
"""

from __future__ import annotations

import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

import requests

from zetta_utils import log
from zetta_utils.common import RESOURCE_STATS_FILE
from zetta_utils.log import set_verbosity
from zetta_utils.mazepa.semaphores import TimingTracker

from ..gcloud.gcs import get_bucket_location_info, is_region_compatible
from .container import get_main_container_status
from .gcs_tracker_utils import (
    get_pod_name,
    get_worker_type,
    read_existing_pod_stats,
    write_pod_stats,
    write_region_mismatch,
)


@dataclass
class StatsCollector:
    name: str
    collect: Callable[[], "dict | None"]


logger = log.get_logger("zetta_utils")

PROXY_PORT = 8080
UPDATE_INTERVAL = 10  # seconds between stats updates in run_db
CA_CERT_SHARED_PATH = "/tmp/mitmproxy-ca/mitmproxy-ca-cert.pem"
MITMPROXY_CONFDIR = "/tmp/mitmproxy-ca"
STATS_FILE = "/tmp/gcs_tracker_stats.json"


MITM_ADDON_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "mitm_addon.py")


def _read_stats_from_file() -> dict:
    """Read stats from the shared file written by mitmproxy addon."""
    try:
        with open(STATS_FILE, encoding="utf-8") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        logger.warning(f"Stats file not found: {STATS_FILE}")
        return {
            "buckets": {},
            "last_updated": time.time(),
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse stats file: {e}")
        return {
            "buckets": {},
            "last_updated": time.time(),
        }


def _get_compute_region() -> str | None:
    """Get compute region from GCE metadata server."""
    try:
        resp = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/zone",
            headers={"Metadata-Flavor": "Google"},
            timeout=5,
        )
        resp.raise_for_status()
        # Returns: "projects/PROJECT_NUM/zones/us-east1-b"
        zone = resp.text.split("/")[-1]
        region = "-".join(zone.split("-")[:-1])
        return region
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Could not get compute region from metadata server: {e}")
        return None


def _check_and_cache_bucket_region(
    bucket_name: str,
    compute_region: str,
    bucket_region_match: dict[str, bool | None],
    run_id: str,
) -> None:
    """Check bucket region compatibility and cache result."""
    if bucket_name.startswith("_") or bucket_name in bucket_region_match:
        return
    try:
        location, data_locations = get_bucket_location_info(bucket_name)
        region_match = is_region_compatible(location, compute_region, data_locations)
        bucket_region_match[bucket_name] = region_match
        if not region_match:
            logger.warning(
                f"Bucket '{bucket_name}' in '{location}' not compatible "
                f"with compute region '{compute_region}'"
            )
            write_region_mismatch(run_id, bucket_name, location, compute_region)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Could not get location for bucket '{bucket_name}': {e}")
        bucket_region_match[bucket_name] = None


# Semaphore types tracked in the main container. See mazepa/semaphores.py.
# The sidecar reads `/dev/shm/zetta_semaphore_timing_{pid}_{name}`, which is visible
# because k8s pod containers share `/dev/shm` via an `emptyDir` (medium=Memory) volume.
_SEMA_TYPES = ("read", "write", "cuda", "cpu", "tensorrt")
_SHM_PREFIX = "zetta_semaphore_timing_"


def _collect_semaphore_stats() -> dict | None:
    """Read semaphore timing from /dev/shm.

    Returns a dict mapping semaphore type → timing fields, or None when no
    timing shared-memory files are present (main container has not yet
    called configure_semaphores()).
    """
    try:
        files = glob.glob(f"/dev/shm/{_SHM_PREFIX}*")
        if not files:
            return None
        # Filename format: zetta_semaphore_timing_{pid}_{name}
        basename = os.path.basename(files[0])
        pid_str = basename[len(_SHM_PREFIX) :].split("_", 1)[0]
        pid = int(pid_str)

        out: dict[str, dict] = {}
        for sema_type in _SEMA_TYPES:
            try:
                wait, lease, count, start = TimingTracker(
                    name=sema_type, pid=pid
                ).get_timing_data()
                out[sema_type] = {
                    "total_wait_time": wait,
                    "total_lease_time": lease,
                    "lease_count": count,
                    "start_time": start,
                }
            except RuntimeError:
                # This semaphore type not initialized in this run.
                continue
        return out or None
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Semaphore stats collector failed: {e}", exc_info=True)
        return None


def _collect_resource_stats() -> dict | None:
    """Read resource summary written by the main container to RESOURCE_STATS_FILE.

    The main container runs `monitor_resources()` which writes
    `ResourceMonitor.get_summary_stats()` output to this shared file every
    `resource_monitor_interval` seconds. Returns None when the file does not
    yet exist (main container hasn't started writing, or resource monitoring
    is disabled).
    """
    try:
        with open(RESOURCE_STATS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Resource stats collector failed: {e}", exc_info=True)
        return None


def _make_gcs_collector(run_id: str, compute_region: str | None) -> StatsCollector:
    """Build the GCS stats collector. Holds the per-bucket region cache in a closure."""
    bucket_region_match: dict[str, bool | None] = {}

    def collect() -> dict | None:
        try:
            stats = _read_stats_from_file()
            if compute_region:
                for bucket_name in stats.get("buckets", {}):
                    _check_and_cache_bucket_region(
                        bucket_name, compute_region, bucket_region_match, run_id
                    )
                    if bucket_name in bucket_region_match:
                        stats["buckets"][bucket_name]["region_match"] = bucket_region_match[
                            bucket_name
                        ]
            return stats
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"GCS stats collector failed: {e}", exc_info=True)
            return None

    return StatsCollector(name="gcs_stats", collect=collect)


def _collect_all(collectors: list[StatsCollector]) -> dict:
    """Run all collectors and merge their results into a single payload."""
    payload: dict = {}
    for collector in collectors:
        try:
            result = collector.collect()
            if result is not None:
                # Phase 1: GCS collector merges flat at top level to preserve
                # existing Firestore document shape. Later phases will nest
                # under collector.name.
                if collector.name == "gcs_stats":
                    payload.update(result)
                else:
                    payload[collector.name] = result
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Collector {collector.name} failed: {e}", exc_info=True)
    return payload


def _update_firestore_loop(
    run_id: str,
    stop_event: threading.Event,
    collectors: list[StatsCollector],
):
    """Periodically run all collectors and write a unified payload to Firestore."""
    pod_name = get_pod_name()
    worker_type = get_worker_type()
    logger.info(f"Starting Firestore update loop for pod {pod_name}, run {run_id}")

    while not stop_event.wait(UPDATE_INTERVAL):
        payload = _collect_all(collectors)
        if not payload:
            continue
        payload["worker_type"] = worker_type
        try:
            write_pod_stats(run_id, pod_name, payload)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Failed to write pod stats to Firestore: {e}", exc_info=True)


def _setup_mitmproxy_ca() -> None:
    """Generate mitmproxy's CA certificate into the shared volume.

    Runs mitmdump briefly to produce the CA, then the main container reads
    it via wait_for_ca. GCS tracking is a hard requirement for customer
    billing, so any setup failure here raises: K8s will restart the sidecar
    container (via restart_policy=Always), and a persistent failure shows
    up as a visible pod-level crash-loop instead of a silent "running
    direct" state.
    """
    os.makedirs(MITMPROXY_CONFDIR, exist_ok=True)
    logger.info("Generating mitmproxy CA certificate...")

    process = subprocess.Popen(  # pylint: disable=consider-using-with
        [
            "mitmdump",
            "--set",
            f"confdir={MITMPROXY_CONFDIR}",
            "--mode",
            "regular",
            "--listen-port",
            "18080",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    for _ in range(30):
        if os.path.exists(CA_CERT_SHARED_PATH):
            break
        time.sleep(0.5)

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()

    if not os.path.exists(CA_CERT_SHARED_PATH):
        raise RuntimeError(
            f"Failed to generate mitmproxy CA at {CA_CERT_SHARED_PATH}; "
            "GCS tracking is required and cannot run without it."
        )
    logger.info(f"CA certificate generated at {CA_CERT_SHARED_PATH}")
    _write_ready_file()


def _write_ready_file() -> None:
    """Write ready marker so main container can proceed."""
    os.makedirs(MITMPROXY_CONFDIR, exist_ok=True)
    with open(os.path.join(MITMPROXY_CONFDIR, "ready"), "w", encoding="utf-8") as f:
        f.write("1")


def _initialize_stats_file(run_id: str, pod_name: str):
    """Initialize stats file, loading existing stats if pod restarted."""
    existing = read_existing_pod_stats(run_id, pod_name)
    if existing and existing.get("buckets"):
        bucket_count = len(existing.get("buckets", {}))
        logger.info(f"Resumed GCS stats: {bucket_count} bucket(s)")
        try:
            with open(STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(existing, f)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Failed to write initial stats file: {e}")
    else:
        # Write empty stats file
        try:
            with open(STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "buckets": {},
                        "last_updated": time.time(),
                    },
                    f,
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Failed to write initial stats file: {e}")


def _write_final_stats(
    run_id: str,
    pod_name: str,
    collectors: list[StatsCollector],
) -> None:
    """Write final pod stats on shutdown by running all collectors once."""
    try:
        payload = _collect_all(collectors)
        if not payload:
            return
        payload["worker_type"] = get_worker_type()
        write_pod_stats(run_id, pod_name, payload)
        logger.info("Final pod stats written")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Failed final pod stats update: {e}")


MITMPROXY_MAX_RESTARTS = 20
MITMPROXY_RESTART_BACKOFF_SEC = 5


def _run_main_loop(
    process_ref: list[subprocess.Popen],
    stop_event: threading.Event,
) -> None:
    """Run the sidecar main loop until main container exits or shutdown.

    If mitmdump dies while main is still up, restart it in-place (bounded
    by MITMPROXY_MAX_RESTARTS). This keeps the sidecar Python process alive
    so main's traffic routing through port 8080 recovers without depending
    on K8s container-restart policy (which is Never under KEDA Jobs).

    `process_ref` is a single-element list holding the current mitmdump
    process handle. Updated on each restart so callers (e.g. the shutdown
    signal handler) always see the live instance.
    """
    restart_count = 0
    while not stop_event.is_set():
        process = process_ref[0]
        if process.poll() is not None:
            if restart_count >= MITMPROXY_MAX_RESTARTS:
                logger.error(
                    f"mitmproxy crashed {restart_count} times; giving up "
                    "and letting the sidecar container exit"
                )
                return
            restart_count += 1
            logger.warning(
                f"mitmproxy exited with code {process.returncode}; "
                f"restarting ({restart_count}/{MITMPROXY_MAX_RESTARTS})"
            )
            if stop_event.wait(MITMPROXY_RESTART_BACKOFF_SEC):
                return
            process_ref[0] = _start_mitmproxy()
            continue
        status = get_main_container_status()
        if status != -1:
            logger.info(f"Main container exited with code {status}")
            return
        time.sleep(5)


def _start_mitmproxy() -> subprocess.Popen:
    """Start mitmproxy as a subprocess and return the handle."""
    mitm_cmd = [
        "mitmdump",
        "--quiet",
        "--mode",
        "regular",
        "--listen-port",
        str(PROXY_PORT),
        "--set",
        f"confdir={MITMPROXY_CONFDIR}",
        # Stream response bodies > 64KB straight through instead of buffering
        # them for the addon. Egress accounting still works — Part 11 prefers
        # Content-Length over body length, exactly the case where streaming
        # makes the body unavailable. Keeps mproxy memory bounded under load.
        "--set",
        "stream_large_bodies=64k",
        "-s",
        MITM_ADDON_SCRIPT_PATH,
    ]
    logger.info(f"Starting mitmproxy on port {PROXY_PORT}")
    return subprocess.Popen(  # pylint: disable=consider-using-with
        mitm_cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def run_sidecar(run_id: str):
    """Run the pod stats sidecar.

    Sets up mitmproxy (hard requirement — raises on failure), starts the GCS
    interception, and starts the periodic stats-push loop for all collectors.
    """
    logger.info(f"Starting pod stats sidecar for run {run_id}")

    pod_name = get_pod_name()

    collectors: list[StatsCollector] = [
        StatsCollector(name="semaphore_stats", collect=_collect_semaphore_stats),
        StatsCollector(name="resource_stats", collect=_collect_resource_stats),
    ]

    _setup_mitmproxy_ca()
    _initialize_stats_file(run_id, pod_name)
    compute_region = _get_compute_region()
    if compute_region:
        logger.info(f"Compute region: {compute_region}")
    else:
        logger.warning("Could not determine compute region, region_match will be unavailable")
    collectors.append(_make_gcs_collector(run_id, compute_region))
    # _run_main_loop may replace the mitmdump process on restart, so wrap
    # in a mutable container so handle_shutdown always sees the current one.
    mitm_process_ref = [_start_mitmproxy()]

    stop_event = threading.Event()
    update_thread = threading.Thread(
        target=_update_firestore_loop,
        args=(run_id, stop_event, collectors),
        daemon=True,
    )
    update_thread.start()

    def handle_shutdown(_signum, _frame):
        logger.info("Shutting down pod stats sidecar...")
        stop_event.set()
        mitm_process_ref[0].terminate()
        _write_final_stats(run_id, pod_name, collectors)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    _run_main_loop(mitm_process_ref, stop_event)

    stop_event.set()
    mitm_process_ref[0].terminate()
    try:
        mitm_process_ref[0].wait(timeout=5)
    except subprocess.TimeoutExpired:
        mitm_process_ref[0].kill()

    _write_final_stats(run_id, pod_name, collectors)


if __name__ == "__main__":
    set_verbosity("INFO")

    _run_id = os.environ.get("RUN_ID")
    if not _run_id:
        raise RuntimeError("RUN_ID environment variable must be set")
    run_sidecar(_run_id)
