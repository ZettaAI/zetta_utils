# pylint: disable=unspecified-encoding, broad-exception-caught
import os
import re
import time
from collections import deque

from kubernetes import client, config  # type: ignore

from .log_pod_runtime import log_pod_runtime

POD_NAME = os.environ.get("POD_NAME")
NAMESPACE = "default"
CHECK_INTERVAL = 1
WINDOW_SIZE = 60
GROWTH_THRESHOLD_PERCENT = 10
PERCENTAGE_THRESHOLD = 0.85


def get_memory_limit_bytes() -> int:
    def _parse_quantity_to_bytes(quantity) -> int:
        units = {
            "Ki": 1024,
            "Mi": 1024 ** 2,
            "Gi": 1024 ** 3,
            "Ti": 1024 ** 4,
            "Pi": 1024 ** 5,
            "Ei": 1024 ** 6,
            "K": 1000,
            "M": 1000 ** 2,
            "G": 1000 ** 3,
            "T": 1000 ** 4,
            "P": 1000 ** 5,
            "E": 1000 ** 6,
        }

        match = re.match(r"^([0-9.]+)([a-zA-Z]+)?$", quantity)
        if not match:
            raise ValueError(f"Invalid quantity format: {quantity}")

        number, suffix = match.groups()
        number = float(number)

        if not suffix:
            return int(number)  # plain number (assumed to be bytes)
        if suffix not in units:
            raise ValueError(f"Unknown suffix: {suffix}")

        return int(number * units[suffix])

    config.load_incluster_config()
    v1 = client.CoreV1Api()
    pod = v1.read_namespaced_pod(name=POD_NAME, namespace=NAMESPACE)

    for container in pod.spec.containers:
        if container.name != "runtime":
            resources = container.resources
            if resources.limits and "memory" in resources.limits:
                return _parse_quantity_to_bytes(resources.limits["memory"])
    return -1


def get_node_memory_total_bytes():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                parts = line.split()
                return int(parts[1]) * 1024
    raise RuntimeError("Could not read /proc/meminfo")


def parse_quantity(mem_str):
    units = {"Ki": 1024, "Mi": 1024 ** 2, "Gi": 1024 ** 3}
    for suffix, factor in units.items():
        if mem_str.endswith(suffix):
            return int(float(mem_str[: -len(suffix)]) * factor)
    return int(mem_str)


def get_pod_memory_usage():
    config.load_incluster_config()
    metrics_api = client.CustomObjectsApi()
    metrics = metrics_api.get_namespaced_custom_object(
        group="metrics.k8s.io",
        version="v1beta1",
        namespace=NAMESPACE,
        plural="pods",
        name=POD_NAME,
    )
    return sum(parse_quantity(c["usage"]["memory"]) for c in metrics["containers"])


def _get_main_container_status() -> int:
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    pod = v1.read_namespaced_pod(name=POD_NAME, namespace=NAMESPACE)
    for container_status in pod.status.container_statuses:
        if container_status.name == "main":
            state = container_status.state
            if state.terminated:
                return int(state.terminated.exit_code)
    return -1


def monitor_loop():
    log_pod_runtime()
    history: deque = deque(maxlen=WINDOW_SIZE)
    total_bytes = get_memory_limit_bytes()
    if total_bytes < 0:
        total_bytes = get_node_memory_total_bytes()

    while _get_main_container_status() == -1:
        try:
            usage_bytes = get_pod_memory_usage()
            usage_pct = usage_bytes / total_bytes
            history.append(usage_pct)
            growth_alert = False
            if len(history) >= 2:
                mem_old = history[0]
                mem_new = history[-1]
                delta_pct = mem_new - mem_old
                if delta_pct * 100 > GROWTH_THRESHOLD_PERCENT:
                    growth_alert = True

            if growth_alert and usage_pct > PERCENTAGE_THRESHOLD:
                log_pod_runtime()
        except Exception as e:
            print(e)

        time.sleep(CHECK_INTERVAL)
    log_pod_runtime()


if __name__ == "__main__":
    if not POD_NAME:
        raise RuntimeError("POD_NAME environment variable must be set.")
    monitor_loop()
