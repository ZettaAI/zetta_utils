"""
GCS request tracker sidecar.

Tracks and classifies GCS (Google Cloud Storage) requests into Class A and Class B
operations as defined by GCP pricing:

Class A (higher cost): Object create/delete, copy, compose, rewrite, list operations
Class B (lower cost): Object read, metadata read

Uses mitmproxy to intercept HTTPS traffic to storage.googleapis.com.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

import requests

from zetta_utils import log
from zetta_utils.log import set_verbosity

from ..gcloud.gcs import get_bucket_location, is_region_compatible
from .container import get_main_container_status
from .gcs_tracker_utils import (
    get_pod_name,
    read_existing_stats,
    write_region_mismatch,
    write_stats,
)

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
        location = get_bucket_location(bucket_name)
        region_match = is_region_compatible(location, compute_region)
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


def _update_firestore_loop(run_id: str, compute_region: str | None, stop_event: threading.Event):
    """Periodically update Firestore with GCS stats from file."""
    pod_name = get_pod_name()
    logger.info(f"Starting Firestore update loop for pod {pod_name}, run {run_id}")
    bucket_region_match: dict[str, bool | None] = {}

    while not stop_event.wait(UPDATE_INTERVAL):
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

            write_stats(run_id, pod_name, "gcs_stats_proxy", stats)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Failed to update GCS stats in Firestore: {e}", exc_info=True)


def _check_mitmproxy_installed() -> bool:
    """Check if mitmproxy is installed."""
    return shutil.which("mitmdump") is not None


def _setup_mitmproxy_ca():
    """
    Initialize mitmproxy's CA certificate in a shared location.

    This runs mitmdump briefly to generate the CA certificate, then copies it
    to the shared volume so the main container can trust it.
    """
    os.makedirs(MITMPROXY_CONFDIR, exist_ok=True)

    if not _check_mitmproxy_installed():
        logger.warning(
            "mitmproxy (mitmdump) is not installed. GCS request tracking will be disabled. "
            "To enable tracking, install mitmproxy: pip install mitmproxy"
        )
        _write_ready_file(disabled=True)
        return False

    logger.info("Generating mitmproxy CA certificate...")
    try:
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
    except FileNotFoundError:
        logger.warning("mitmdump not found. GCS tracking disabled.")
        _write_ready_file(disabled=True)
        return False

    for _ in range(30):
        if os.path.exists(CA_CERT_SHARED_PATH):
            break
        time.sleep(0.5)

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()

    if os.path.exists(CA_CERT_SHARED_PATH):
        logger.info(f"CA certificate generated at {CA_CERT_SHARED_PATH}")
        _write_ready_file(disabled=False)
        return True
    else:
        logger.error("Failed to generate mitmproxy CA certificate")
        _write_ready_file(disabled=True)
        return False


def _write_ready_file(disabled: bool = False):
    """Write ready marker file so main container can proceed."""
    os.makedirs(MITMPROXY_CONFDIR, exist_ok=True)
    with open(os.path.join(MITMPROXY_CONFDIR, "ready"), "w", encoding="utf-8") as f:
        f.write("disabled" if disabled else "1")


def _wait_for_main_container_exit():
    """Wait for main container to exit (used when tracking is disabled)."""
    logger.info("Waiting for main container to exit...")
    while True:
        status = get_main_container_status()
        if status != -1:
            logger.info(f"Main container exited with code {status}")
            break
        time.sleep(10)


def _initialize_stats_file(run_id: str, pod_name: str):
    """Initialize stats file, loading existing stats if pod restarted."""
    existing = read_existing_stats(run_id, pod_name, "gcs_stats_proxy")
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


def _write_final_stats(run_id: str, pod_name: str) -> None:
    """Write final GCS stats on shutdown."""
    try:
        stats = _read_stats_from_file()
        write_stats(run_id, pod_name, "gcs_stats_proxy", stats)
        bucket_count = len(stats.get("buckets", {}))
        logger.info(f"Final GCS stats: {bucket_count} bucket(s)")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Failed final GCS stats update: {e}")


def _run_main_loop(process: subprocess.Popen) -> None:
    """Main loop: wait for proxy or main container to exit."""
    while True:
        if process.poll() is not None:
            logger.error(f"mitmproxy exited with code {process.returncode}")
            break
        status = get_main_container_status()
        if status != -1:
            logger.info(f"Main container exited with code {status}")
            break
        time.sleep(5)


def run_proxy(run_id: str):
    """
    Run the mitmproxy-based GCS tracker.

    This function starts mitmproxy in transparent/regular proxy mode and
    tracks all GCS requests, periodically updating Firestore with stats.
    """
    logger.info(f"Starting GCS tracker for run {run_id}")

    pod_name = get_pod_name()
    _initialize_stats_file(run_id, pod_name)

    if not _setup_mitmproxy_ca():
        logger.warning("GCS tracking disabled - mitmproxy not available")
        _wait_for_main_container_exit()
        return

    compute_region = _get_compute_region()
    if compute_region:
        logger.info(f"Compute region: {compute_region}")
    else:
        logger.warning("Could not determine compute region, region_match will be unavailable")

    stop_event = threading.Event()
    update_thread = threading.Thread(
        target=_update_firestore_loop,
        args=(run_id, compute_region, stop_event),
        daemon=True,
    )
    update_thread.start()

    mitm_cmd = [
        "mitmdump",
        "--quiet",
        "--mode",
        "regular",
        "--listen-port",
        str(PROXY_PORT),
        "--set",
        f"confdir={MITMPROXY_CONFDIR}",
        "--set",
        "stream_large_bodies=1m",
        "-s",
        MITM_ADDON_SCRIPT_PATH,
    ]

    logger.info(f"Starting mitmproxy on port {PROXY_PORT}")
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        mitm_cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    def handle_shutdown(_signum, _frame):
        logger.info("Shutting down GCS tracker...")
        stop_event.set()
        process.terminate()
        _write_final_stats(run_id, pod_name)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    _run_main_loop(process)

    stop_event.set()
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()

    _write_final_stats(run_id, pod_name)


if __name__ == "__main__":
    set_verbosity("INFO")

    _run_id = os.environ.get("RUN_ID")
    if not _run_id:
        raise RuntimeError("RUN_ID environment variable must be set")
    run_proxy(_run_id)
