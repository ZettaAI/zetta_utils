# pylint: disable=line-too-long, too-many-locals, too-many-statements
from __future__ import annotations

import time
from typing import Any, Dict

import attrs
import psutil

try:  # pragma: no cover # logging only
    import pynvml

    pynvml.nvmlInit()
    NVIDIA_GPU_AVAILABLE = True
except (ImportError, Exception):  # pylint: disable=broad-except
    NVIDIA_GPU_AVAILABLE = False
    pynvml = None

from zetta_utils import log
from zetta_utils.common.pprint import lrpad

logger = log.get_logger("mazepa")


@attrs.define
class ResourceMonitor:  # pragma: no cover # logging only
    """
    Monitor system resource usage (CPU, RAM, Disk, GPU);
    should be run in an asynchronous thread that calls
    ResourceMonitor.log_usage method every log_interval_seconds,
    using a RepeatTimer.
    """

    log_interval_seconds: float
    gpu_handles: list = attrs.field(factory=list)
    samples: list[Dict[str, Any]] = attrs.field(factory=list)

    # Track previous I/O values for rate calculations
    prev_disk_io: Dict[str, Any] | None = None
    prev_net_io: Dict[str, Any] | None = None
    prev_time: float | None = None

    def __attrs_post_init__(self):
        if NVIDIA_GPU_AVAILABLE:
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)]
                logger.info(f"Initialized GPU monitoring for {gpu_count} devices.")
            except Exception as e:  # pylint:disable=broad-except
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.gpu_handles = []

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=None)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024 ** 3),
            "used_gb": mem.used / (1024 ** 3),
            "available_gb": mem.available / (1024 ** 3),
            "percent": mem.percent,
        }

    def get_disk_io(self) -> Dict[str, Any]:
        """Get current disk I/O statistics."""
        disk_io = psutil.disk_io_counters()
        if not disk_io:
            return {
                "read_bytes": 0,
                "write_bytes": 0,
                "read_count": 0,
                "write_count": 0,
                "read_time": 0,
                "write_time": 0,
            }

        return {
            "read_bytes": disk_io.read_bytes,
            "write_bytes": disk_io.write_bytes,
            "read_count": disk_io.read_count,
            "write_count": disk_io.write_count,
            "read_time": disk_io.read_time,  # milliseconds
            "write_time": disk_io.write_time,  # milliseconds
        }

    def get_network_usage(self) -> Dict[str, Any]:
        """Get current network usage information."""
        net_io = psutil.net_io_counters()
        if not net_io:
            return {
                "bytes_sent": 0,
                "bytes_recv": 0,
                "packets_sent": 0,
                "packets_recv": 0,
                "errin": 0,
                "errout": 0,
                "dropin": 0,
                "dropout": 0,
            }

        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "errin": net_io.errin,
            "errout": net_io.errout,
            "dropin": net_io.dropin,
            "dropout": net_io.dropout,
        }

    def get_gpu_usage(self) -> list[Dict[str, Any]]:
        """Get current GPU usage information."""
        if not self.gpu_handles:
            return []

        gpu_stats = []
        for i, handle in enumerate(self.gpu_handles):
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                gpu_stats.append(
                    {
                        "gpu_id": i,
                        "gpu_percent": util.gpu,
                        "memory_percent": util.memory,
                        "memory_total_gb": mem_info.total / (1024 ** 3),
                        "memory_used_gb": mem_info.used / (1024 ** 3),
                        "memory_free_gb": mem_info.free / (1024 ** 3),
                    }
                )
            except Exception as e:  # pylint:disable=broad-except
                logger.warning(f"Failed to get stats for GPU {i}: {e}")

        return gpu_stats

    def get_all_usage(self) -> Dict[str, Any]:
        """Get comprehensive resource usage snapshot."""
        return {
            "timestamp": time.time(),
            "cpu_percent": self.get_cpu_usage(),
            "memory": self.get_memory_usage(),
            "disk_io": self.get_disk_io(),
            "network": self.get_network_usage(),
            "gpus": self.get_gpu_usage(),
        }

    def log_usage(self) -> None:
        """Log current resource usage and optionally collect for summary."""
        stats = self.get_all_usage()
        current_time = time.time()

        # Calculate I/O rates if we have previous measurements
        if (
            self.prev_disk_io is not None
            and self.prev_net_io is not None
            and self.prev_time is not None
        ):
            time_delta = current_time - self.prev_time
            if time_delta > 0:
                # Calculate disk I/O rates (MB/s)
                disk_read_rate_mbps = (
                    (stats["disk_io"]["read_bytes"] - self.prev_disk_io["read_bytes"])
                    / (1024 ** 2)
                    / time_delta
                )
                disk_write_rate_mbps = (
                    (stats["disk_io"]["write_bytes"] - self.prev_disk_io["write_bytes"])
                    / (1024 ** 2)
                    / time_delta
                )

                # Calculate network I/O rates (MB/s)
                net_recv_rate_mbps = (
                    (stats["network"]["bytes_recv"] - self.prev_net_io["bytes_recv"])
                    / (1024 ** 2)
                    / time_delta
                )
                net_send_rate_mbps = (
                    (stats["network"]["bytes_sent"] - self.prev_net_io["bytes_sent"])
                    / (1024 ** 2)
                    / time_delta
                )

                # Add rates to stats
                stats["disk_io"]["read_rate_mbps"] = disk_read_rate_mbps
                stats["disk_io"]["write_rate_mbps"] = disk_write_rate_mbps
                stats["network"]["recv_rate_mbps"] = net_recv_rate_mbps
                stats["network"]["send_rate_mbps"] = net_send_rate_mbps
            else:
                # First measurement or no time elapsed, set rates to 0
                stats["disk_io"]["read_rate_mbps"] = 0.0
                stats["disk_io"]["write_rate_mbps"] = 0.0
                stats["network"]["recv_rate_mbps"] = 0.0
                stats["network"]["send_rate_mbps"] = 0.0
        else:
            # First measurement, no rates available
            stats["disk_io"]["read_rate_mbps"] = 0.0
            stats["disk_io"]["write_rate_mbps"] = 0.0
            stats["network"]["recv_rate_mbps"] = 0.0
            stats["network"]["send_rate_mbps"] = 0.0

        # Update previous values for next calculation
        self.prev_disk_io = stats["disk_io"].copy()
        self.prev_net_io = stats["network"].copy()
        self.prev_time = current_time

        # Store sample for summary
        self.samples.append(stats)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics from collected samples."""
        if not self.samples:
            return {}

        def calculate_avg(values):
            return sum(values) / len(values) if values else 0.0

        def calculate_max(values):
            return max(values) if values else 0.0

        def calculate_min(values):
            return min(values) if values else 0.0

        def calculate_25th_percentile(values):
            if not values:
                return 0.0
            sorted_values = sorted(values)
            n = len(sorted_values)
            index = int(0.25 * (n - 1))
            # Linear interpolation for more accurate percentile
            if index < n - 1:
                lower = sorted_values[index]
                upper = sorted_values[index + 1]
                fraction = 0.25 * (n - 1) - index
                return lower + fraction * (upper - lower)
            return sorted_values[index]

        cpu_values = [s["cpu_percent"] for s in self.samples]
        mem_used = [s["memory"]["used_gb"] for s in self.samples]
        mem_total = self.samples[0]["memory"]["total_gb"] if self.samples else 0.0
        mem_percent = [s["memory"]["percent"] for s in self.samples]

        # Disk I/O stats (calculate deltas for totals)
        disk_read_bytes = [s["disk_io"]["read_bytes"] for s in self.samples]
        disk_write_bytes = [s["disk_io"]["write_bytes"] for s in self.samples]
        disk_read_count = [s["disk_io"]["read_count"] for s in self.samples]
        disk_write_count = [s["disk_io"]["write_count"] for s in self.samples]

        # Network stats (calculate deltas for rates)
        net_bytes_sent = [s["network"]["bytes_sent"] for s in self.samples]
        net_bytes_recv = [s["network"]["bytes_recv"] for s in self.samples]
        net_packets_sent = [s["network"]["packets_sent"] for s in self.samples]
        net_packets_recv = [s["network"]["packets_recv"] for s in self.samples]

        # I/O rates (per-interval rates in MB/s)
        disk_read_rate_mbps = [s["disk_io"]["read_rate_mbps"] for s in self.samples]
        disk_write_rate_mbps = [s["disk_io"]["write_rate_mbps"] for s in self.samples]
        net_recv_rate_mbps = [s["network"]["recv_rate_mbps"] for s in self.samples]
        net_send_rate_mbps = [s["network"]["send_rate_mbps"] for s in self.samples]

        duration_seconds = len(self.samples) * self.log_interval_seconds

        summary = {
            "sample_count": len(self.samples),
            "duration_seconds": duration_seconds,
            "cpu": {
                "avg_percent": calculate_avg(cpu_values),
                "max_percent": calculate_max(cpu_values),
                "min_percent": calculate_min(cpu_values),
                "p25_percent": calculate_25th_percentile(cpu_values),
            },
            "memory": {
                "total_gb": mem_total,
                "avg_used_gb": calculate_avg(mem_used),
                "max_used_gb": calculate_max(mem_used),
                "min_used_gb": calculate_min(mem_used),
                "p25_used_gb": calculate_25th_percentile(mem_used),
                "avg_percent": calculate_avg(mem_percent),
                "max_percent": calculate_max(mem_percent),
                "min_percent": calculate_min(mem_percent),
                "p25_percent": calculate_25th_percentile(mem_percent),
            },
            "disk_io": {
                "total_read_gb": (disk_read_bytes[-1] - disk_read_bytes[0]) / (1024 ** 3)
                if len(disk_read_bytes) > 1
                else 0.0,
                "total_write_gb": (disk_write_bytes[-1] - disk_write_bytes[0]) / (1024 ** 3)
                if len(disk_write_bytes) > 1
                else 0.0,
                "total_read_ops": disk_read_count[-1] - disk_read_count[0]
                if len(disk_read_count) > 1
                else 0,
                "total_write_ops": disk_write_count[-1] - disk_write_count[0]
                if len(disk_write_count) > 1
                else 0,
                "avg_read_rate_mbps": (
                    (disk_read_bytes[-1] - disk_read_bytes[0])
                    / (1024 ** 2)
                    / max(duration_seconds, 1)
                )
                if len(disk_read_bytes) > 1
                else 0.0,
                "avg_write_rate_mbps": (
                    (disk_write_bytes[-1] - disk_write_bytes[0])
                    / (1024 ** 2)
                    / max(duration_seconds, 1)
                )
                if len(disk_write_bytes) > 1
                else 0.0,
                "avg_read_rate_interval_mbps": calculate_avg(disk_read_rate_mbps),
                "max_read_rate_mbps": calculate_max(disk_read_rate_mbps),
                "p25_read_rate_mbps": calculate_25th_percentile(disk_read_rate_mbps),
                "avg_write_rate_interval_mbps": calculate_avg(disk_write_rate_mbps),
                "max_write_rate_mbps": calculate_max(disk_write_rate_mbps),
                "p25_write_rate_mbps": calculate_25th_percentile(disk_write_rate_mbps),
            },
            "network": {
                "total_bytes_sent_gb": (net_bytes_sent[-1] - net_bytes_sent[0]) / (1024 ** 3)
                if len(net_bytes_sent) > 1
                else 0.0,
                "total_bytes_recv_gb": (net_bytes_recv[-1] - net_bytes_recv[0]) / (1024 ** 3)
                if len(net_bytes_recv) > 1
                else 0.0,
                "total_packets_sent": net_packets_sent[-1] - net_packets_sent[0]
                if len(net_packets_sent) > 1
                else 0,
                "total_packets_recv": net_packets_recv[-1] - net_packets_recv[0]
                if len(net_packets_recv) > 1
                else 0,
                "avg_send_rate_mbps": (
                    (net_bytes_sent[-1] - net_bytes_sent[0])
                    / (1024 ** 2)
                    / max(duration_seconds, 1)
                )
                if len(net_bytes_sent) > 1
                else 0.0,
                "avg_recv_rate_mbps": (
                    (net_bytes_recv[-1] - net_bytes_recv[0])
                    / (1024 ** 2)
                    / max(duration_seconds, 1)
                )
                if len(net_bytes_recv) > 1
                else 0.0,
                "avg_recv_rate_interval_mbps": calculate_avg(net_recv_rate_mbps),
                "max_recv_rate_mbps": calculate_max(net_recv_rate_mbps),
                "p25_recv_rate_mbps": calculate_25th_percentile(net_recv_rate_mbps),
                "avg_send_rate_interval_mbps": calculate_avg(net_send_rate_mbps),
                "max_send_rate_mbps": calculate_max(net_send_rate_mbps),
                "p25_send_rate_mbps": calculate_25th_percentile(net_send_rate_mbps),
            },
        }

        # GPU summary if available
        if self.samples and self.samples[0]["gpus"]:
            gpu_summary = {}
            for gpu_id in range(len(self.samples[0]["gpus"])):
                gpu_util = [
                    s["gpus"][gpu_id]["gpu_percent"]
                    for s in self.samples
                    if gpu_id < len(s["gpus"])
                ]
                gpu_mem_used = [
                    s["gpus"][gpu_id]["memory_used_gb"]
                    for s in self.samples
                    if gpu_id < len(s["gpus"])
                ]
                gpu_mem_total = (
                    self.samples[0]["gpus"][gpu_id]["memory_total_gb"]
                    if self.samples[0]["gpus"]
                    else 0.0
                )

                gpu_summary[f"gpu_{gpu_id}"] = {
                    "memory_total_gb": gpu_mem_total,
                    "avg_utilization_percent": calculate_avg(gpu_util),
                    "max_utilization_percent": calculate_max(gpu_util),
                    "min_utilization_percent": calculate_min(gpu_util),
                    "p25_utilization_percent": calculate_25th_percentile(gpu_util),
                    "avg_memory_used_gb": calculate_avg(gpu_mem_used),
                    "max_memory_used_gb": calculate_max(gpu_mem_used),
                    "min_memory_used_gb": calculate_min(gpu_mem_used),
                    "p25_memory_used_gb": calculate_25th_percentile(gpu_mem_used),
                }
            summary["gpus"] = gpu_summary

        return summary

    def log_summary(self) -> None:
        """Log a pretty-printed summary of resource usage."""
        if not self.samples:
            logger.debug("No resource monitoring samples available for summary.")
            return

        stats = self.get_summary_stats()
        if not stats:
            return

        summary = ""
        summary += lrpad("  Resource Usage Summary  ", bounds="+", filler="=", length=80) + "\n"
        summary += lrpad("", bounds="|", length=80) + "\n"
        summary += (
            lrpad(
                f"Monitoring Duration: {stats['duration_seconds']:.1f}s ({stats['sample_count']} samples)",
                bounds="|",
                length=80,
            )
            + "\n"
        )
        summary += lrpad("", bounds="|", length=80) + "\n"

        # CPU/Memory/GPU Table
        cpu = stats["cpu"]
        mem = stats["memory"]

        summary += lrpad("System Resources", bounds="|", length=80) + "\n"
        summary += lrpad("", filler="-", bounds="|", length=80) + "\n"

        # Table header
        header = f"{'Resource':<12} {'Total':>9} {'Peak':>9} {'Average':>11} {'Low (25%)':>11} {'Minimum':>11}"
        summary += lrpad(header, bounds="|", length=80) + "\n"
        summary += lrpad("", filler="-", bounds="|", length=80) + "\n"

        # CPU row
        cpu_avg = f"{cpu['avg_percent']:.1f}%"
        cpu_max = f"{cpu['max_percent']:.1f}%"
        cpu_p25 = f"{cpu['p25_percent']:.1f}%"
        cpu_min = f"{cpu['min_percent']:.1f}%"
        cpu_row = f"{'CPU':<12} {'-':>9} {cpu_max:>9} {cpu_avg:>11} {cpu_p25:>11} {cpu_min:>11}"
        summary += lrpad(cpu_row, bounds="|", length=80) + "\n"

        # Memory row
        mem_total = f"{mem['total_gb']:.1f}GB"
        mem_avg = f"{mem['avg_used_gb']:.1f}GB"
        mem_max = f"{mem['max_used_gb']:.1f}GB"
        mem_p25 = f"{mem['p25_used_gb']:.1f}GB"
        mem_min = f"{mem['min_used_gb']:.1f}GB"
        mem_row = (
            f"{'Memory':<12} {mem_total:>9} {mem_max:>9} {mem_avg:>11} {mem_p25:>11} {mem_min:>11}"
        )
        summary += lrpad(mem_row, bounds="|", length=80) + "\n"

        # GPU rows if available
        if "gpus" in stats:
            for gpu_name, gpu_stats in stats["gpus"].items():
                gpu_id = gpu_name.split("_")[1]

                # GPU utilization row
                gpu_util_avg = f"{gpu_stats['avg_utilization_percent']:.1f}%"
                gpu_util_max = f"{gpu_stats['max_utilization_percent']:.1f}%"
                gpu_util_p25 = f"{gpu_stats['p25_utilization_percent']:.1f}%"
                gpu_util_min = f"{gpu_stats['min_utilization_percent']:.1f}%"
                gpu_util_row = f"{'GPU ' + gpu_id:<12} {'-':>9} {gpu_util_max:>9} {gpu_util_avg:>11} {gpu_util_p25:>11} {gpu_util_min:>11}"
                summary += lrpad(gpu_util_row, bounds="|", length=80) + "\n"

                # GPU memory row
                gpu_mem_total = f"{gpu_stats['memory_total_gb']:.1f}GB"
                gpu_mem_avg = f"{gpu_stats['avg_memory_used_gb']:.1f}GB"
                gpu_mem_max = f"{gpu_stats['max_memory_used_gb']:.1f}GB"
                gpu_mem_p25 = f"{gpu_stats['p25_memory_used_gb']:.1f}GB"
                gpu_mem_min = f"{gpu_stats['min_memory_used_gb']:.1f}GB"
                gpu_mem_row = f"{'GPU ' + gpu_id + ' Mem':<12} {gpu_mem_total:>9} {gpu_mem_max:>9} {gpu_mem_avg:>11} {gpu_mem_p25:>11} {gpu_mem_min:>11}"
                summary += lrpad(gpu_mem_row, bounds="|", length=80) + "\n"

        summary += lrpad("", bounds="|", length=80) + "\n"

        # I/O Operations Table
        disk_io = stats["disk_io"]
        net = stats["network"]

        summary += lrpad("I/O Operations", bounds="|", length=80) + "\n"
        summary += lrpad("", filler="-", bounds="|", length=80) + "\n"

        # I/O Table header
        io_header = f"{'Operation':<12} {'Count':>9} {'Total':>9} {'Avg Rate':>11} {'Low Rate':>11} {'Peak Rate':>11}"
        summary += lrpad(io_header, bounds="|", length=80) + "\n"
        summary += lrpad("", filler="-", bounds="|", length=80) + "\n"

        # Disk Read row
        disk_read_ops = f"{disk_io['total_read_ops']:,}"
        disk_read_bytes = f"{disk_io['total_read_gb']:.2f}GB"
        disk_read_rate = f"{disk_io['avg_read_rate_mbps']:.1f}MB/s"
        disk_read_rate_p25 = f"{disk_io.get('p25_read_rate_mbps', 0):.1f}MB/s"
        disk_read_rate_max = f"{disk_io.get('max_read_rate_mbps', 0):.1f}MB/s"
        disk_read_row = f"{'Disk Read':<12} {disk_read_ops:>9} {disk_read_bytes:>9} {disk_read_rate:>11} {disk_read_rate_p25:>11} {disk_read_rate_max:>11}"
        summary += lrpad(disk_read_row, bounds="|", length=80) + "\n"

        # Disk Write row
        disk_write_ops = f"{disk_io['total_write_ops']:,}"
        disk_write_bytes = f"{disk_io['total_write_gb']:.2f}GB"
        disk_write_rate = f"{disk_io['avg_write_rate_mbps']:.1f}MB/s"
        disk_write_rate_p25 = f"{disk_io.get('p25_write_rate_mbps', 0):.1f}MB/s"
        disk_write_rate_max = f"{disk_io.get('max_write_rate_mbps', 0):.1f}MB/s"
        disk_write_row = f"{'Disk Write':<12} {disk_write_ops:>9} {disk_write_bytes:>9} {disk_write_rate:>11} {disk_write_rate_p25:>11} {disk_write_rate_max:>11}"
        summary += lrpad(disk_write_row, bounds="|", length=80) + "\n"

        # Network Receive row
        net_recv_ops = f"{net['total_packets_recv']:,}"
        net_recv_bytes = f"{net['total_bytes_recv_gb']:.2f}GB"
        net_recv_rate = f"{net['avg_recv_rate_mbps']:.1f}MB/s"
        net_recv_rate_p25 = f"{net.get('p25_recv_rate_mbps', 0):.1f}MB/s"
        net_recv_rate_max = f"{net.get('max_recv_rate_mbps', 0):.1f}MB/s"
        net_recv_row = f"{'Net Receive':<12} {net_recv_ops:>9} {net_recv_bytes:>9} {net_recv_rate:>11} {net_recv_rate_p25:>11} {net_recv_rate_max:>11}"
        summary += lrpad(net_recv_row, bounds="|", length=80) + "\n"

        # Network Send row
        net_send_ops = f"{net['total_packets_sent']:,}"
        net_send_bytes = f"{net['total_bytes_sent_gb']:.2f}GB"
        net_send_rate = f"{net['avg_send_rate_mbps']:.1f}MB/s"
        net_send_rate_p25 = f"{net.get('p25_send_rate_mbps', 0):.1f}MB/s"
        net_send_rate_max = f"{net.get('max_send_rate_mbps', 0):.1f}MB/s"
        net_send_row = f"{'Net Send':<12} {net_send_ops:>9} {net_send_bytes:>9} {net_send_rate:>11} {net_send_rate_p25:>11} {net_send_rate_max:>11}"
        summary += lrpad(net_send_row, bounds="|", length=80) + "\n"

        summary += lrpad("", bounds="|", length=80) + "\n"
        summary += lrpad("", bounds="+", filler="=", length=80)

        logger.info(summary)
