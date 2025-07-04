"""Tests for segmentation stats updater worker"""

# pylint: disable=unused-argument,redefined-outer-name

from datetime import datetime, timedelta

from click.testing import CliRunner
from slack_sdk.errors import SlackApiError

from zetta_utils.task_management.automated_workers.segmentation_stats_updater import (
    create_task_dashboard_link,
    process_task,
    run_worker,
    send_slack_error_notification,
)

# Module path constant to avoid long lines
_SSU = "zetta_utils.task_management.automated_workers.segmentation_stats_updater"


class TestUtilityFunctions:
    """Test utility functions"""

    def test_create_task_dashboard_link(self):
        """Test creating task dashboard link"""
        link = create_task_dashboard_link("test_project", "task123")
        expected = "https://zetta-task-manager.vercel.app/test_project/tasks/task123"
        assert link == expected

    def test_send_slack_error_notification_success(self, mocker):
        """Test sending error notification to Slack"""
        mock_slack = mocker.patch(f"{_SSU}.slack_client")
        mock_slack.token = "test_token"

        send_slack_error_notification("test-channel", "test_project", "Connection timeout", 35)

        mock_slack.chat_postMessage.assert_called_once()
        call_args = mock_slack.chat_postMessage.call_args[1]
        assert call_args["channel"] == "test-channel"
        assert "Connection timeout" in call_args["text"]
        assert "35 minutes" in call_args["text"]
        assert "test_project" in call_args["text"]

    def test_send_slack_error_notification_no_channel(self, mocker):
        """Test skipping notification when no channel"""
        mock_slack = mocker.patch(f"{_SSU}.slack_client")
        send_slack_error_notification(None, "test_project", "Error", 30)

        mock_slack.chat_postMessage.assert_not_called()

    def test_send_slack_error_notification_no_token(self, mocker):
        """Test skipping notification when no token"""
        mock_slack = mocker.patch(f"{_SSU}.slack_client")
        mock_slack.token = None

        send_slack_error_notification("test-channel", "test_project", "Error", 30)

        mock_slack.chat_postMessage.assert_not_called()

    def test_send_slack_error_notification_api_error(self, mocker):
        """Test handling Slack API error"""
        mock_slack = mocker.patch(f"{_SSU}.slack_client")
        mock_slack.token = "test_token"
        mock_slack.chat_postMessage.side_effect = SlackApiError("Invalid channel", None)

        # Should not raise exception
        send_slack_error_notification("test-channel", "test_project", "Error", 30)


class TestProcessTask:
    """Test process_task function"""

    def test_process_task_success(self, mocker):
        """Test successfully processing a task"""
        mock_get_task = mocker.patch(f"{_SSU}.get_task")
        mock_update_stats = mocker.patch(f"{_SSU}.update_segment_statistics")
        mock_release = mocker.patch(f"{_SSU}.release_task")

        mock_get_task.return_value = {
            "task_id": "task123",
            "task_type": "seg_stats_update_v0",
            "extra_data": {"seed_id": 12345},
        }

        mock_update_stats.return_value = {
            "skeleton_path_length_mm": 2.5,
            "pre_synapse_count": 10,
            "post_synapse_count": 5,
        }

        process_task("task123", "test_project", "worker_user", 1)

        mock_update_stats.assert_called_once_with(project_name="test_project", seed_id=12345)

        mock_release.assert_called_once_with(
            project_name="test_project",
            task_id="task123",
            user_id="worker_user",
            completion_status="Done",
        )

    def test_process_task_with_errors(self, mocker):
        """Test processing task with partial errors"""
        mock_get_task = mocker.patch(f"{_SSU}.get_task")
        mock_update_stats = mocker.patch(f"{_SSU}.update_segment_statistics")
        mock_release = mocker.patch(f"{_SSU}.release_task")
        mock_get_task.return_value = {
            "task_id": "task123",
            "task_type": "seg_stats_update_v0",
            "extra_data": {"seed_id": 12345},
        }

        mock_update_stats.return_value = {
            "skeleton_error": "Failed to compute skeleton",
            "pre_synapse_count": 10,
            "post_synapse_count": 5,
        }

        process_task("task123", "test_project", "worker_user", 1)

        # Should still release as done
        mock_release.assert_called_once_with(
            project_name="test_project",
            task_id="task123",
            user_id="worker_user",
            completion_status="Done",
        )

    def test_process_task_no_extra_data(self, mocker):
        """Test processing task with no extra_data"""
        mock_get_task = mocker.patch(f"{_SSU}.get_task")
        mock_release = mocker.patch(f"{_SSU}.release_task")
        mock_get_task.return_value = {
            "task_id": "task123",
            "task_type": "seg_stats_update_v0",
            # No extra_data
        }

        process_task("task123", "test_project", "worker_user", 1)

        # Should release as done even though couldn't process
        mock_release.assert_called_once_with(
            project_name="test_project",
            task_id="task123",
            user_id="worker_user",
            completion_status="Done",
        )

    def test_process_task_no_seed_id(self, mocker):
        """Test processing task with no seed_id"""
        mock_get_task = mocker.patch(f"{_SSU}.get_task")
        mock_release = mocker.patch(f"{_SSU}.release_task")
        mock_get_task.return_value = {
            "task_id": "task123",
            "task_type": "seg_stats_update_v0",
            "extra_data": {},  # No seed_id
        }

        process_task("task123", "test_project", "worker_user", 1)

        # Should release as done
        mock_release.assert_called_once_with(
            project_name="test_project",
            task_id="task123",
            user_id="worker_user",
            completion_status="Done",
        )

    def test_process_task_update_exception(self, mocker):
        """Test handling exception during stats update"""
        mock_get_task = mocker.patch(f"{_SSU}.get_task")
        mock_update_stats = mocker.patch(f"{_SSU}.update_segment_statistics")
        mock_release = mocker.patch(f"{_SSU}.release_task")
        mock_get_task.return_value = {
            "task_id": "task123",
            "task_type": "seg_stats_update_v0",
            "extra_data": {"seed_id": 12345},
        }

        mock_update_stats.side_effect = Exception("Database connection failed")

        process_task("task123", "test_project", "worker_user", 1)

        # Should still release task
        mock_release.assert_called_once_with(
            project_name="test_project",
            task_id="task123",
            user_id="worker_user",
            completion_status="Done",
        )

    def test_process_task_with_general_error(self, mocker):
        """Test processing task with general error result"""
        mock_get_task = mocker.patch(f"{_SSU}.get_task")
        mock_update_stats = mocker.patch(f"{_SSU}.update_segment_statistics")
        mock_release = mocker.patch(f"{_SSU}.release_task")
        mock_get_task.return_value = {
            "task_id": "task123",
            "task_type": "seg_stats_update_v0",
            "extra_data": {"seed_id": 12345},
        }

        mock_update_stats.return_value = {"error": "No current_segment_id"}

        process_task("task123", "test_project", "worker_user", 1)

        # Should still complete
        mock_release.assert_called_once()


class TestRunWorker:
    """Test run_worker function"""

    def test_run_worker_with_tasks(self, mocker):
        """Test worker processing tasks"""
        mock_start_task = mocker.patch(f"{_SSU}.start_task")
        mock_process_task = mocker.patch(f"{_SSU}.process_task")
        mocker.patch(f"{_SSU}.time.sleep")
        # Simulate getting a task then keyboard interrupt
        mock_start_task.side_effect = ["task123", KeyboardInterrupt]

        runner = CliRunner()
        result = runner.invoke(
            run_worker, ["--user_id", "worker1", "--project_name", "test_project"]
        )

        # Should have processed one task
        mock_process_task.assert_called_once_with("task123", "test_project", "worker1", 1)
        assert "Tasks Processed: 1" in result.output

    def test_run_worker_no_tasks(self, mocker):
        """Test worker when no tasks available"""
        mock_start_task = mocker.patch(f"{_SSU}.start_task")
        mock_sleep = mocker.patch(f"{_SSU}.time.sleep")
        # Return None (no tasks) then keyboard interrupt
        mock_start_task.side_effect = [None, KeyboardInterrupt]

        runner = CliRunner()
        result = runner.invoke(
            run_worker, ["--user_id", "worker1", "--project_name", "test_project"]
        )

        # Should have slept
        mock_sleep.assert_called_with(5.0)
        assert "Tasks Processed: 0" in result.output

    def test_run_worker_with_persistent_error(self, mocker):
        """Test worker handling persistent errors and sending Slack notification"""
        mock_start_task = mocker.patch(f"{_SSU}.start_task")
        mock_process_task = mocker.patch(f"{_SSU}.process_task")
        mock_sleep = mocker.patch(f"{_SSU}.time.sleep")
        mock_datetime = mocker.patch(f"{_SSU}.datetime")
        mock_slack_notify = mocker.patch(f"{_SSU}.send_slack_error_notification")
        # Mock datetime to control timing
        base_time = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.side_effect = [
            base_time,  # First exception
            base_time + timedelta(minutes=35),  # After 35 minutes
            base_time + timedelta(minutes=36),  # Continue
        ]

        # Simulate error, then another error after threshold, then interrupt
        mock_start_task.side_effect = ["task123", "task456", KeyboardInterrupt]
        mock_process_task.side_effect = [
            Exception("Connection failed"),
            Exception("Still failing"),
        ]

        runner = CliRunner()
        runner.invoke(
            run_worker,
            [
                "--user_id",
                "worker1",
                "--project_name",
                "test_project",
                "--slack_channel",
                "test-channel",
            ],
        )

        # Should have sent Slack notification after 30 minutes
        mock_slack_notify.assert_called_once()
        call_args = mock_slack_notify.call_args[0]
        assert call_args[0] == "test-channel"
        assert call_args[1] == "test_project"
        # The last exception message is sent
        assert call_args[2] in ["Connection failed", "Still failing"]
        assert call_args[3] >= 30  # Duration in minutes

        # Should have slept for retry delay
        assert mock_sleep.call_args_list[0][0][0] == 60  # RETRY_DELAY_SECONDS

    def test_run_worker_error_recovery(self, mocker):
        """Test worker recovering from errors"""
        mock_start_task = mocker.patch(f"{_SSU}.start_task")
        mock_process_task = mocker.patch(f"{_SSU}.process_task")
        mocker.patch(f"{_SSU}.time.sleep")
        # Error, then success, then interrupt
        mock_start_task.side_effect = ["task123", "task456", KeyboardInterrupt]
        mock_process_task.side_effect = [
            Exception("Temporary error"),
            None,  # Success
        ]

        runner = CliRunner()
        result = runner.invoke(
            run_worker, ["--user_id", "worker1", "--project_name", "test_project"]
        )

        # Should have processed both tasks (one failed, one succeeded)
        assert mock_process_task.call_count == 2
        # Tasks processed counter should only count successful starts
        assert "Tasks Processed: 2" in result.output

    def test_run_worker_cli_options(self):
        """Test CLI options parsing"""
        runner = CliRunner()
        result = runner.invoke(run_worker, ["--help"])

        assert result.exit_code == 0
        assert "--user_id" in result.output
        assert "--project_name" in result.output
        assert "--polling_period" in result.output
        assert "--slack_channel" in result.output

    def test_run_worker_default_user_id(self, mocker):
        """Test default user_id value"""
        runner = CliRunner()

        # Mock to capture the call and prevent actual execution
        mock_start = mocker.patch(f"{_SSU}.start_task")
        mock_start.side_effect = KeyboardInterrupt

        result = runner.invoke(run_worker, ["--project_name", "test_project"])

        # Check the startup panel shows default user
        assert "automated_worker" in result.output
