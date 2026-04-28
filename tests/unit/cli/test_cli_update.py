# pylint: disable=redefined-outer-name,unused-argument
import pytest
from click.testing import CliRunner
from kubernetes.client.exceptions import ApiException

from zetta_utils.cli.run.cli_update import run_update_cli
from zetta_utils.cloud_management.resource_allocation.k8s.autoscaler import (
    MAX_REPLICAS_ANNOTATION,
    MIN_REPLICAS_ANNOTATION,
    SCALE_DOWN_STABILIZATION_SEC_ANNOTATION,
)


@pytest.fixture
def mock_apps_api(mocker):
    apps_api_class = mocker.patch("zetta_utils.cli.run.cli_update.k8s_client.AppsV1Api")
    return apps_api_class.return_value


@pytest.fixture
def mock_logger(mocker):
    return mocker.patch("zetta_utils.cli.run.cli_update.logger")


def test_run_update_max_workers(mock_apps_api, mock_logger):
    runner = CliRunner()
    result = runner.invoke(
        run_update_cli,
        ["run-update", "test-run-123", "-g", "io", "--max-workers", "10"],
    )
    assert result.exit_code == 0, result.output
    mock_apps_api.patch_namespaced_deployment.assert_called_once_with(
        name="run-test-run-123-io",
        namespace="default",
        body={"metadata": {"annotations": {MAX_REPLICAS_ANNOTATION: "10"}}},
    )


def test_run_update_multiple_groups(mock_apps_api, mock_logger):
    runner = CliRunner()
    result = runner.invoke(
        run_update_cli,
        ["run-update", "test-run", "-g", "io", "-g", "gpu", "--max-workers", "5"],
    )
    assert result.exit_code == 0, result.output
    assert mock_apps_api.patch_namespaced_deployment.call_count == 2
    names = {c.kwargs["name"] for c in mock_apps_api.patch_namespaced_deployment.call_args_list}
    assert names == {"run-test-run-io", "run-test-run-gpu"}


def test_run_update_underscore_group_name_normalized(mock_apps_api, mock_logger):
    runner = CliRunner()
    result = runner.invoke(
        run_update_cli,
        ["run-update", "my-run", "-g", "gpu_pool", "--max-workers", "1"],
    )
    assert result.exit_code == 0, result.output
    mock_apps_api.patch_namespaced_deployment.assert_called_once_with(
        name="run-my-run-gpu-pool",
        namespace="default",
        body={"metadata": {"annotations": {MAX_REPLICAS_ANNOTATION: "1"}}},
    )


def test_run_update_all_flags(mock_apps_api, mock_logger):
    runner = CliRunner()
    result = runner.invoke(
        run_update_cli,
        [
            "run-update",
            "test-run",
            "-g",
            "io",
            "--max-workers",
            "100",
            "--min-workers",
            "5",
            "--scale-down-stabilization-sec",
            "120",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_apps_api.patch_namespaced_deployment.assert_called_once_with(
        name="run-test-run-io",
        namespace="default",
        body={
            "metadata": {
                "annotations": {
                    MAX_REPLICAS_ANNOTATION: "100",
                    MIN_REPLICAS_ANNOTATION: "5",
                    SCALE_DOWN_STABILIZATION_SEC_ANNOTATION: "120",
                }
            }
        },
    )


def test_run_update_no_flags_raises_usage_error(mock_apps_api, mock_logger):
    runner = CliRunner()
    result = runner.invoke(run_update_cli, ["run-update", "test-run", "-g", "io"])
    assert result.exit_code != 0
    mock_apps_api.patch_namespaced_deployment.assert_not_called()


def test_run_update_404_logs_info(mock_apps_api, mock_logger):
    mock_apps_api.patch_namespaced_deployment.side_effect = ApiException(status=404)
    runner = CliRunner()
    result = runner.invoke(
        run_update_cli,
        ["run-update", "missing", "-g", "io", "--max-workers", "1"],
    )
    assert result.exit_code == 0
    mock_logger.info.assert_called()
    mock_logger.warning.assert_not_called()


def test_run_update_403_logs_warning(mock_apps_api, mock_logger):
    mock_apps_api.patch_namespaced_deployment.side_effect = ApiException(
        status=403, reason="Forbidden"
    )
    runner = CliRunner()
    result = runner.invoke(
        run_update_cli,
        ["run-update", "test-run", "-g", "io", "--max-workers", "1"],
    )
    assert result.exit_code == 0
    mock_logger.warning.assert_called()
