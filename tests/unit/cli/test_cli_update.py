# pylint: disable=unused-argument,redefined-outer-name
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from kubernetes.client import ApiException

from zetta_utils.cli.run.cli_update import run_update_cli


@pytest.fixture
def mock_k8s(mocker):
    """Mock k8s module."""
    mock = MagicMock()
    mocker.patch("zetta_utils.cli.run.cli_update.k8s", mock)
    return mock


@pytest.fixture
def mock_logger(mocker):
    """Mock logger."""
    mock = MagicMock()
    mocker.patch("zetta_utils.cli.run.cli_update.logger", mock)
    return mock


def test_run_update_default(mock_k8s, mock_logger):
    """Test run_update command with default settings (no worker groups)."""
    runner = CliRunner()

    result = runner.invoke(run_update_cli, ["run-update", "test-run-123", "--max-workers", "10"])

    assert result.exit_code == 0
    # Should patch the default scaled job
    mock_k8s.patch_scaledjob.assert_called_once_with(
        "run-test-run-123-sj", patch_body={"spec": {"maxReplicaCount": 10}}
    )


def test_run_update_with_worker_groups(mock_k8s, mock_logger):
    """Test run_update command with specific worker groups."""
    runner = CliRunner()

    result = runner.invoke(
        run_update_cli,
        [
            "run-update",
            "test-run-123",
            "--max-workers",
            "20",
            "--worker-groups",
            "group1",
            "--worker-groups",
            "group2",
        ],
    )

    assert result.exit_code == 0
    # Should patch each worker group
    assert mock_k8s.patch_scaledjob.call_count == 2

    calls = mock_k8s.patch_scaledjob.call_args_list
    expected_patch = {"spec": {"maxReplicaCount": 20}}

    assert calls[0][0][0] == "run-test-run-123-group1-sj"
    assert calls[0][1]["patch_body"] == expected_patch

    assert calls[1][0][0] == "run-test-run-123-group2-sj"
    assert calls[1][1]["patch_body"] == expected_patch


def test_run_update_with_underscores(mock_k8s, mock_logger):
    """Test that underscores in names are replaced with hyphens."""
    runner = CliRunner()

    result = runner.invoke(
        run_update_cli,
        ["run-update", "test_run_123", "--max-workers", "15", "--worker-groups", "worker_group_1"],
    )

    assert result.exit_code == 0
    # Check that underscores were replaced with hyphens
    mock_k8s.patch_scaledjob.assert_called_once_with(
        "run-test-run-123-worker-group-1-sj", patch_body={"spec": {"maxReplicaCount": 15}}
    )


def test_run_update_resource_not_found(mock_k8s, mock_logger):
    """Test handling of 404 errors (resource not found)."""
    runner = CliRunner()

    # Mock 404 error
    mock_k8s.patch_scaledjob.side_effect = ApiException(status=404)

    result = runner.invoke(
        run_update_cli, ["run-update", "nonexistent-run", "--max-workers", "10"]
    )

    assert result.exit_code == 0
    # Should log info about non-existent resource
    mock_logger.info.assert_called_once()
    assert "Resource does not exist" in mock_logger.info.call_args[0][0]


def test_run_update_other_api_error(mock_k8s, mock_logger):
    """Test handling of non-404 API errors."""
    runner = CliRunner()

    # Mock other API error
    mock_k8s.patch_scaledjob.side_effect = ApiException(status=403, reason="Forbidden")

    result = runner.invoke(run_update_cli, ["run-update", "test-run", "--max-workers", "10"])

    assert result.exit_code == 0
    # Should log warning about failure
    mock_logger.warning.assert_called_once()
    assert "Failed to update k8s resource" in mock_logger.warning.call_args[0][0]


def test_run_update_multiple_groups_partial_failure(mock_k8s, mock_logger):
    """Test updating multiple groups where some fail."""
    runner = CliRunner()

    # First call succeeds, second fails with 404, third fails with 403
    mock_k8s.patch_scaledjob.side_effect = [
        None,  # Success
        ApiException(status=404),  # Not found
        ApiException(status=403),  # Forbidden
    ]

    result = runner.invoke(
        run_update_cli,
        [
            "run-update",
            "test-run-123",
            "--max-workers",
            "25",
            "--worker-groups",
            "group1",
            "--worker-groups",
            "group2",
            "--worker-groups",
            "group3",
        ],
    )

    assert result.exit_code == 0
    assert mock_k8s.patch_scaledjob.call_count == 3
    # Should have one info log and one warning log
    assert mock_logger.info.call_count == 1
    assert mock_logger.warning.call_count == 1
