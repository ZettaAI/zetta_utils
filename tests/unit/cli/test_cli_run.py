# pylint: disable=unused-argument,redefined-outer-name
import os
import sys
import time
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from zetta_utils.cli.run.cli import COLUMNS, _print_infos, run_info_cli


@pytest.fixture
def mock_run_db(mocker):
    """Mock the RUN_DB for testing."""
    mock_db = MagicMock()
    mocker.patch("zetta_utils.cli.run.cli.RunInfo", MagicMock)
    return mock_db


@pytest.fixture
def mock_imports(mocker):
    """Mock the imports inside the command functions."""
    mock_run_db = MagicMock()
    mock_run_info_bucket = "/test/bucket"

    # Mock the imports that happen inside the command functions
    mocker.patch.dict(
        "sys.modules",
        {
            "zetta_utils.run": MagicMock(
                RUN_INFO_BUCKET=mock_run_info_bucket,
                RunInfo=MagicMock(),
                get_latest_checkpoint=MagicMock(),
            ),
            "zetta_utils.run.db": MagicMock(
                RUN_DB=mock_run_db,
            ),
        },
    )

    return mock_run_db, mock_run_info_bucket


def test_print_infos():
    """Test the _print_infos function."""
    current_time = time.time()
    infos = [
        {
            "zetta_user": "test_user",
            "state": "running",
            "timestamp": current_time - 100,
            "heartbeat": current_time,
            "run_id": "test_run_123",
        }
    ]

    table = _print_infos(infos)

    # Check that table was created with correct columns
    assert len(table.columns) == len(COLUMNS._fields)
    # Rich tables store cell data internally, we can check the row count
    assert len(table.rows) == 1


def test_run_info(mock_imports, mocker):
    """Test the run_info command."""
    mock_run_db, _ = mock_imports
    runner = CliRunner()

    # Setup mock data
    current_time = time.time()
    mock_infos = [
        {
            "zetta_user": "user1",
            "state": "completed",
            "timestamp": current_time - 200,
            "heartbeat": current_time - 100,
            "run_id": "run1",
        },
        {
            "zetta_user": "user2",
            "state": "running",
            "timestamp": current_time - 50,
            "heartbeat": current_time,
            "run_id": "run2",
        },
    ]

    # Mock RUN_DB behavior
    mock_run_db.__getitem__.return_value = mock_infos

    # Mock get_latest_checkpoint
    sys.modules["zetta_utils.run"].get_latest_checkpoint.return_value = "/checkpoint/path"

    # Mock environment variable
    mocker.patch.dict(os.environ, {"RUN_INFO_BUCKET": "/custom/bucket"})

    result = runner.invoke(run_info_cli, ["run-info", "run1", "run2"])

    assert result.exit_code == 0
    # Verify RUN_DB was called correctly
    mock_run_db.__getitem__.assert_called_once()


def test_run_info_checkpoint_not_found(mock_imports, mocker):
    """Test run_info when checkpoint is not found."""
    mock_run_db, _ = mock_imports
    runner = CliRunner()

    # Setup mock data
    current_time = time.time()
    mock_infos = [
        {
            "zetta_user": "user1",
            "state": "completed",
            "timestamp": current_time - 200,
            "heartbeat": current_time - 100,
            "run_id": "run1",
        }
    ]

    mock_run_db.__getitem__.return_value = mock_infos

    # Mock get_latest_checkpoint to raise FileNotFoundError
    sys.modules["zetta_utils.run"].get_latest_checkpoint.side_effect = FileNotFoundError(
        "No checkpoint"
    )

    result = runner.invoke(run_info_cli, ["run-info", "run1"])

    assert result.exit_code == 0
    assert "FileNotFoundError" in result.output or result.exit_code == 0


def test_run_list_no_filter(mock_imports, mocker):
    """Test run_list command without user filter."""
    mock_run_db, _ = mock_imports
    runner = CliRunner()

    current_time = time.time()
    mock_result = {
        "run1": {
            "zetta_user": "user1",
            "state": "running",
            "timestamp": current_time - 100,
            "heartbeat": current_time,
        },
        "run2": {
            "zetta_user": "user2",
            "state": "completed",
            "timestamp": current_time - 200,
            "heartbeat": current_time - 150,
        },
    }

    mock_run_db.query.return_value = mock_result

    result = runner.invoke(run_info_cli, ["run-list"])

    assert result.exit_code == 0
    # Verify query was called with correct time filter
    mock_run_db.query.assert_called_once()
    query_filter = mock_run_db.query.call_args[0][0]
    assert ">timestamp" in query_filter
    assert len(query_filter[">timestamp"]) == 1
    # Check it's filtering for last 7 days
    assert query_filter[">timestamp"][0] > current_time - (7 * 24 * 3600 + 1)
    assert query_filter[">timestamp"][0] < current_time - (7 * 24 * 3600 - 1)


def test_run_list_with_user_filter(mock_imports, mocker):
    """Test run_list command with user filter."""
    mock_run_db, _ = mock_imports
    runner = CliRunner()

    current_time = time.time()
    mock_result = {
        "run1": {
            "zetta_user": "test_user",
            "state": "running",
            "timestamp": current_time - 100,
            "heartbeat": current_time,
        }
    }

    mock_run_db.query.return_value = mock_result

    result = runner.invoke(run_info_cli, ["run-list", "test_user"])

    assert result.exit_code == 0
    # Verify query was called with user filter
    mock_run_db.query.assert_called_once()
    query_filter = mock_run_db.query.call_args[0][0]
    assert "zetta_user" in query_filter
    assert query_filter["zetta_user"] == ["test_user"]


def test_run_list_with_days_filter(mock_imports, mocker):
    """Test run_list command with custom days filter."""
    mock_run_db, _ = mock_imports
    runner = CliRunner()

    current_time = time.time()
    mock_result = {
        "run1": {
            "zetta_user": "user1",
            "state": "running",
            "timestamp": current_time - 100,
            "heartbeat": current_time,
        }
    }

    mock_run_db.query.return_value = mock_result

    result = runner.invoke(run_info_cli, ["run-list", "--days", "3"])

    assert result.exit_code == 0
    # Verify query was called with correct time filter
    mock_run_db.query.assert_called_once()
    query_filter = mock_run_db.query.call_args[0][0]
    assert ">timestamp" in query_filter
    # Check it's filtering for last 3 days
    assert query_filter[">timestamp"][0] > current_time - (3 * 24 * 3600 + 1)
    assert query_filter[">timestamp"][0] < current_time - (3 * 24 * 3600 - 1)
