import time

from zetta_utils.common.resource_monitor import ResourceMonitor


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
