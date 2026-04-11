"""Tests for write_resource_stats_file context manager."""

import json
import time

from zetta_utils.common import resource_stats_file
from zetta_utils.common.resource_stats_file import write_resource_stats_file


def test_noop_when_interval_none(tmp_path):
    path = tmp_path / "should_not_exist.json"
    with write_resource_stats_file(interval=None, path=str(path)):
        time.sleep(0.05)
    assert not path.exists()


def test_writes_summary_to_file(tmp_path):
    path = tmp_path / "resource_stats.json"
    with write_resource_stats_file(interval=0.1, path=str(path)):
        # Two ticks worth
        time.sleep(0.3)

    assert path.exists()
    data = json.loads(path.read_text())
    assert "sample_count" in data
    assert "cpu" in data
    assert "memory" in data


def test_bad_path_logs_warning_without_raising(tmp_path, mocker):
    bad_path = str(tmp_path / "nonexistent_dir" / "out.json")
    warning_spy = mocker.patch.object(resource_stats_file.logger, "warning")

    with write_resource_stats_file(interval=0.1, path=bad_path):
        time.sleep(0.3)

    warning_spy.assert_called()
    # No exception escaped the context manager
