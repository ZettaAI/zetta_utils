from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml

    pynvml.nvmlInit()
    NVIDIA_GPU_AVAILABLE = True
except (ImportError, Exception):
    NVIDIA_GPU_AVAILABLE = False
    pynvml = None

from zetta_utils import log
from zetta_utils.common.pprint import lrpad

logger = log.get_logger("mazepa")


class ResourceMonitor:
    """Monitor system resource usage (CPU, RAM, Disk, GPU) for worker pools."""

    def __init__(self, log_interval_seconds: float = 1.0, collect_summary: bool = False):
        self.log_interval = log_interval_seconds
        self.collect_summary = collect_summary
        self.gpu_handles = []
        self.samples = [] if collect_summary else None

        if psutil is None:
            raise ImportError(
                "psutil is required for resource monitoring. Install with: pip install psutil"
            )

        if NVIDIA_GPU_AVAILABLE:
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)]
                logger.info(f"Initialized GPU monitoring for {gpu_count} devices")
            except Exception as e:
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
            except Exception as e:
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

        # Store sample if collecting summary
        if self.collect_summary and self.samples is not None:
            self.samples.append(stats)
            return  # Don't log individual entries when collecting summary

        # Format log message for individual logging
        cpu_pct = stats["cpu_percent"]
        mem = stats["memory"]
        disk_io = stats["disk_io"]
        net = stats["network"]

        log_msg = (
            f"Resources: CPU {cpu_pct:.1f}%, "
            f"RAM {mem['used_gb']:.1f}/{mem['total_gb']:.1f}GB ({mem['percent']:.1f}%), "
            f"Disk ↓{disk_io['read_bytes']/(1024**2):.1f}MB ↑{disk_io['write_bytes']/(1024**2):.1f}MB, "
            f"Net ↑{net['bytes_sent']/(1024**2):.1f}MB ↓{net['bytes_recv']/(1024**2):.1f}MB"
        )

        if stats["gpus"]:
            gpu_info = []
            for gpu in stats["gpus"]:
                gpu_info.append(
                    f"GPU{gpu['gpu_id']} {gpu['gpu_percent']}% "
                    f"({gpu['memory_used_gb']:.1f}/{gpu['memory_total_gb']:.1f}GB)"
                )
            log_msg += f", {', '.join(gpu_info)}"

        logger.info(log_msg)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics from collected samples."""
        if not self.collect_summary or not self.samples:
            return {}

        def calculate_avg(values):
            return sum(values) / len(values) if values else 0.0

        def calculate_max(values):
            return max(values) if values else 0.0

        def calculate_min(values):
            return min(values) if values else 0.0

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

        duration_seconds = len(self.samples) * self.log_interval

        summary = {
            "sample_count": len(self.samples),
            "duration_seconds": duration_seconds,
            "cpu": {
                "avg_percent": calculate_avg(cpu_values),
                "max_percent": calculate_max(cpu_values),
                "min_percent": calculate_min(cpu_values),
            },
            "memory": {
                "total_gb": mem_total,
                "avg_used_gb": calculate_avg(mem_used),
                "max_used_gb": calculate_max(mem_used),
                "min_used_gb": calculate_min(mem_used),
                "avg_percent": calculate_avg(mem_percent),
                "max_percent": calculate_max(mem_percent),
                "min_percent": calculate_min(mem_percent),
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
                    "avg_memory_used_gb": calculate_avg(gpu_mem_used),
                    "max_memory_used_gb": calculate_max(gpu_mem_used),
                    "min_memory_used_gb": calculate_min(gpu_mem_used),
                }
            summary["gpus"] = gpu_summary

        return summary

    def log_summary(self) -> None:
        """Log a pretty-printed summary of resource usage."""
        if not self.collect_summary or not self.samples:
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
        header = f"{'Resource':<12} {'Total':>12} {'Average':>12} {'Peak':>12} {'Minimum':>12}"
        summary += lrpad(header, bounds="|", length=80) + "\n"
        summary += lrpad("", filler="-", bounds="|", length=80) + "\n"

        # CPU row
        cpu_avg = f"{cpu['avg_percent']:.1f}%"
        cpu_max = f"{cpu['max_percent']:.1f}%"
        cpu_min = f"{cpu['min_percent']:.1f}%"
        cpu_row = f"{'CPU':<12} {'-':>12} {cpu_avg:>12} {cpu_max:>12} {cpu_min:>12}"
        summary += lrpad(cpu_row, bounds="|", length=80) + "\n"

        # Memory row
        mem_total = f"{mem['total_gb']:.1f}GB"
        mem_avg = f"{mem['avg_used_gb']:.1f}GB"
        mem_max = f"{mem['max_used_gb']:.1f}GB"
        mem_min = f"{mem['min_used_gb']:.1f}GB"
        mem_row = f"{'Memory':<12} {mem_total:>12} {mem_avg:>12} {mem_max:>12} {mem_min:>12}"
        summary += lrpad(mem_row, bounds="|", length=80) + "\n"

        # GPU rows if available
        if "gpus" in stats:
            for gpu_name, gpu_stats in stats["gpus"].items():
                gpu_id = gpu_name.split("_")[1]

                # GPU utilization row
                gpu_util_avg = f"{gpu_stats['avg_utilization_percent']:.1f}%"
                gpu_util_max = f"{gpu_stats['max_utilization_percent']:.1f}%"
                gpu_util_min = f"{gpu_stats['min_utilization_percent']:.1f}%"
                gpu_util_row = f"{'GPU ' + gpu_id:<12} {'-':>12} {gpu_util_avg:>12} {gpu_util_max:>12} {gpu_util_min:>12}"
                summary += lrpad(gpu_util_row, bounds="|", length=80) + "\n"

                # GPU memory row
                gpu_mem_total = f"{gpu_stats['memory_total_gb']:.1f}GB"
                gpu_mem_avg = f"{gpu_stats['avg_memory_used_gb']:.1f}GB"
                gpu_mem_max = f"{gpu_stats['max_memory_used_gb']:.1f}GB"
                gpu_mem_min = f"{gpu_stats['min_memory_used_gb']:.1f}GB"
                gpu_mem_row = f"{'GPU ' + gpu_id + ' Mem':<12} {gpu_mem_total:>12} {gpu_mem_avg:>12} {gpu_mem_max:>12} {gpu_mem_min:>12}"
                summary += lrpad(gpu_mem_row, bounds="|", length=80) + "\n"

        summary += lrpad("", bounds="|", length=80) + "\n"

        # I/O Operations Table
        disk_io = stats["disk_io"]
        net = stats["network"]

        summary += lrpad("I/O Operations", bounds="|", length=80) + "\n"
        summary += lrpad("", filler="-", bounds="|", length=80) + "\n"

        # I/O Table header
        io_header = f"{'Operation':<15} {'Count':>15} {'Total Bytes':>15} {'Avg Rate':>15}"
        summary += lrpad(io_header, bounds="|", length=80) + "\n"
        summary += lrpad("", filler="-", bounds="|", length=80) + "\n"

        # Disk Read row
        disk_read_ops = f"{disk_io['total_read_ops']:,}"
        disk_read_bytes = f"{disk_io['total_read_gb']:.3f}GB"
        disk_read_rate = f"{disk_io['avg_read_rate_mbps']:.2f}MB/s"
        disk_read_row = (
            f"{'Disk Read':<15} {disk_read_ops:>15} {disk_read_bytes:>15} {disk_read_rate:>15}"
        )
        summary += lrpad(disk_read_row, bounds="|", length=80) + "\n"

        # Disk Write row
        disk_write_ops = f"{disk_io['total_write_ops']:,}"
        disk_write_bytes = f"{disk_io['total_write_gb']:.3f}GB"
        disk_write_rate = f"{disk_io['avg_write_rate_mbps']:.2f}MB/s"
        disk_write_row = (
            f"{'Disk Write':<15} {disk_write_ops:>15} {disk_write_bytes:>15} {disk_write_rate:>15}"
        )
        summary += lrpad(disk_write_row, bounds="|", length=80) + "\n"

        # Network Receive row
        net_recv_ops = f"{net['total_packets_recv']:,}"
        net_recv_bytes = f"{net['total_bytes_recv_gb']:.3f}GB"
        net_recv_rate = f"{net['avg_recv_rate_mbps']:.2f}MB/s"
        net_recv_row = (
            f"{'Net Receive':<15} {net_recv_ops:>15} {net_recv_bytes:>15} {net_recv_rate:>15}"
        )
        summary += lrpad(net_recv_row, bounds="|", length=80) + "\n"

        # Network Send row
        net_send_ops = f"{net['total_packets_sent']:,}"
        net_send_bytes = f"{net['total_bytes_sent_gb']:.3f}GB"
        net_send_rate = f"{net['avg_send_rate_mbps']:.2f}MB/s"
        net_send_row = (
            f"{'Net Send':<15} {net_send_ops:>15} {net_send_bytes:>15} {net_send_rate:>15}"
        )
        summary += lrpad(net_send_row, bounds="|", length=80) + "\n"

        summary += lrpad("", bounds="|", length=80) + "\n"
        summary += lrpad("", bounds="+", filler="=", length=80)

        logger.info(summary)
