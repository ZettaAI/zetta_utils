# pylint: disable=protected-access
import builtins
import time
from unittest.mock import mock_open

from zetta_utils.common import resource_monitor as rm
from zetta_utils.common.resource_monitor import ResourceMonitor, monitor_resources


def _stub_file_reads(monkeypatch, values):
    """Patch open() so only `values` paths return content; other /sys/fs/cgroup paths FNF."""
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        s = str(path)
        if s in values:
            return mock_open(read_data=values[s])()
        if s.startswith("/sys/fs/cgroup/"):
            raise FileNotFoundError(s)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_resource_monitor_basic():
    monitor = ResourceMonitor(log_interval_seconds=0.1)

    assert monitor.log_interval_seconds == 0.1
    assert len(monitor.samples) == 0
    assert monitor.prev_time is None

    usage = monitor.get_all_usage()
    assert "timestamp" in usage
    assert "cpu_percent" in usage
    assert "memory" in usage
    assert "disk_io" in usage
    assert "network" in usage
    assert "gpus" in usage

    monitor.log_usage()
    time.sleep(0.1)
    monitor.log_usage()
    time.sleep(0.1)
    monitor.log_usage()

    summary = monitor.get_summary_stats()

    assert len(monitor.samples) == 3
    assert "sample_count" in summary
    assert summary["sample_count"] == 3
    assert "duration_seconds" in summary
    assert "cpu" in summary
    assert "memory" in summary

    monitor.log_summary()


def test_resource_monitor_empty():
    monitor = ResourceMonitor(log_interval_seconds=0.1)
    summary = monitor.get_summary_stats()
    assert not summary


def test_monitor_resources_none():
    with monitor_resources():
        pass


def test_monitor_resources_notnone():
    with monitor_resources(0.1):
        time.sleep(0.2)


# ---------- cgroup-aware memory reporting ----------


def test_cgroup_v2_read(monkeypatch):
    _stub_file_reads(
        monkeypatch,
        {
            "/sys/fs/cgroup/memory.current": "5368709120",  # 5 GiB
            "/sys/fs/cgroup/memory.max": "13958643712",  # 13 GiB
        },
    )
    result = rm._read_cgroup_memory()
    assert result is not None
    used, total = result
    assert used == 5368709120
    assert total == 13958643712


def test_cgroup_v1_fallback(monkeypatch):
    _stub_file_reads(
        monkeypatch,
        {
            "/sys/fs/cgroup/memory/memory.usage_in_bytes": "1073741824",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes": "4294967296",
        },
    )
    result = rm._read_cgroup_memory()
    assert result is not None
    used, total = result
    assert used == 1073741824
    assert total == 4294967296


def test_cgroup_no_limit_returns_none(monkeypatch):
    """Sentinel value (no constraint) → caller falls back to psutil."""
    _stub_file_reads(
        monkeypatch,
        {
            "/sys/fs/cgroup/memory.current": "1234",
            "/sys/fs/cgroup/memory.max": str(2 ** 63 - 1),
        },
    )
    assert rm._read_cgroup_memory() is None


def test_cgroup_missing_returns_none(monkeypatch):
    _stub_file_reads(monkeypatch, {})
    assert rm._read_cgroup_memory() is None


def test_cgroup_garbage_value_returns_none(monkeypatch):
    """cgroup v2 sometimes contains the literal "max" string; treat as missing."""
    _stub_file_reads(
        monkeypatch,
        {
            "/sys/fs/cgroup/memory.current": "1234",
            "/sys/fs/cgroup/memory.max": "max",
        },
    )
    assert rm._read_cgroup_memory() is None


def test_get_memory_usage_uses_cgroup_when_available(monkeypatch):
    _stub_file_reads(
        monkeypatch,
        {
            "/sys/fs/cgroup/memory.current": str(13 * (1024 ** 3) // 2),
            "/sys/fs/cgroup/memory.max": str(13 * (1024 ** 3)),
        },
    )
    monitor = ResourceMonitor(log_interval_seconds=1.0)
    info = monitor.get_memory_usage()
    assert info["total_gib"] == 13.0
    assert info["used_gib"] == 6.5
    assert info["percent"] == 50.0


def test_get_memory_usage_falls_back_to_psutil(monkeypatch):
    _stub_file_reads(monkeypatch, {})
    monitor = ResourceMonitor(log_interval_seconds=1.0)
    info = monitor.get_memory_usage()
    assert info["total_gib"] > 0
    assert 0 <= info["percent"] <= 100
