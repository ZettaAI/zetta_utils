"""Tests for segmentation auto verifier worker"""

# pylint: disable=unused-argument,redefined-outer-name

from unittest.mock import Mock

from click.testing import CliRunner

from zetta_utils.task_management.automated_workers.segmentation_auto_verifier import (
    SegmentSkeleton,
    create_task_dashboard_link,
    extract_target_location,
    get_skeleton,
    get_task_timesheet_summary,
    is_unavailable_error,
    log_retry_attempt,
    process_task,
    run_worker,
)
from zetta_utils.task_management.db.models import TimesheetModel

# Module path constant to avoid long lines
_SAV = "zetta_utils.task_management.automated_workers.segmentation_auto_verifier"


class TestSegmentSkeleton:
    """Test SegmentSkeleton class"""

    def test_segment_skeleton_creation(self):
        """Test creating a SegmentSkeleton instance"""
        mock_skeleton = Mock()
        seg_skel = SegmentSkeleton(segment_id=12345, skeleton=mock_skeleton)

        assert seg_skel.segment_id == 12345
        assert seg_skel.skeleton == mock_skeleton


class TestUtilityFunctions:
    """Test utility functions"""

    def test_create_task_dashboard_link(self):
        """Test creating task dashboard link"""
        link = create_task_dashboard_link("test_project", "task123")
        expected = "https://zetta-task-manager.vercel.app/test_project/tasks/task123"
        assert link == expected

    def test_is_unavailable_error(self):
        """Test checking if error contains 'unavailable'"""
        assert is_unavailable_error("Service unavailable")
        assert is_unavailable_error("SERVICE UNAVAILABLE")
        assert is_unavailable_error("The system is currently unavailable")
        assert not is_unavailable_error("Connection refused")
        assert not is_unavailable_error("Timeout error")

    def test_log_retry_attempt(self):
        """Test logging retry attempts"""
        retry_state = Mock()
        retry_state.attempt_number = 3
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = Exception("Service unavailable")

        # Should not raise any errors
        log_retry_attempt(retry_state)

    def test_extract_target_location(self):
        """Test extracting target location from ng_dict"""
        # Test with valid target location
        ng_dict = {
            "layers": [
                {"name": "Some Layer"},
                {"name": "Target Location", "annotations": [{"point": [100.0, 200.0, 300.5]}]},
            ]
        }
        location = extract_target_location(ng_dict)
        assert location == [100.0, 200.0, 300.0]  # z-coordinate adjusted by -0.5

        # Test with no target location layer
        ng_dict_no_target = {"layers": [{"name": "Some Layer"}]}
        assert extract_target_location(ng_dict_no_target) is None

        # Test with empty annotations
        ng_dict_empty = {"layers": [{"name": "Target Location", "annotations": []}]}
        assert extract_target_location(ng_dict_empty) is None


class TestGetSkeleton:
    """Test get_skeleton function"""

    def test_get_skeleton_success(self, mocker):
        """Test successfully getting skeleton"""
        # Setup mocks
        import numpy as np  # pylint: disable=import-outside-toplevel

        mock_build_cv = mocker.patch(f"{_SAV}.build_cv_layer")
        mocker.patch(f"{_SAV}.CAVEclient")
        mock_pcg_skel = mocker.patch(f"{_SAV}.pcg_skel")

        mock_layer = Mock()
        # Mock the slicing operation to return a numpy array
        mock_layer.__getitem__ = Mock(return_value=np.array([[[67890]]]))
        mock_build_cv.return_value = mock_layer

        mock_skeleton = Mock()
        mock_skeleton.path_length.return_value = 1500000  # 1.5 mm
        mock_pcg_skel.pcg_skeleton.return_value = mock_skeleton

        ng_dict = {
            "layers": [
                {"name": "Target Location", "annotations": [{"point": [100.0, 200.0, 300.5]}]}
            ]
        }

        result = get_skeleton(ng_dict, "gs://test/path", [8, 8, 40], [16, 16, 40])

        assert isinstance(result, SegmentSkeleton)
        assert result.segment_id == 67890
        assert result.skeleton == mock_skeleton

    def test_get_skeleton_no_target_location(self, mocker):
        """Test get_skeleton with no target location"""
        mock_build_cv = mocker.patch(f"{_SAV}.build_cv_layer")

        ng_dict: dict[str, list] = {"layers": []}
        result = get_skeleton(ng_dict, "gs://test", [8, 8, 40], [16, 16, 40])
        assert result is None
        mock_build_cv.assert_not_called()

    def test_get_skeleton_retry_on_unavailable(self, mocker):
        """Test get_skeleton retries on unavailable error"""
        # Setup layer mock
        import numpy as np  # pylint: disable=import-outside-toplevel

        mock_build_cv = mocker.patch(f"{_SAV}.build_cv_layer")
        mocker.patch(f"{_SAV}.CAVEclient")
        mock_pcg_skel = mocker.patch(f"{_SAV}.pcg_skel")

        mock_layer = Mock()
        mock_layer.__getitem__ = Mock(return_value=np.array([[[67890]]]))
        mock_build_cv.return_value = mock_layer

        # First call fails with unavailable, second succeeds
        mock_skeleton = Mock()
        mock_pcg_skel.pcg_skeleton.side_effect = [Exception("Service unavailable"), mock_skeleton]

        ng_dict = {
            "layers": [
                {"name": "Target Location", "annotations": [{"point": [100.0, 200.0, 300.5]}]}
            ]
        }

        result = get_skeleton(ng_dict, "gs://test", [8, 8, 40], [16, 16, 40])

        assert result is not None and result.skeleton == mock_skeleton
        assert mock_pcg_skel.pcg_skeleton.call_count == 2


class TestProcessTask:
    """Test process_task function"""

    def test_process_task_pass(self, mocker):
        """Test processing a task that passes verification"""
        # Setup mocks
        mock_get_slack_client = mocker.patch(f"{_SAV}.get_slack_client")
        mock_slack = Mock()
        mock_get_slack_client.return_value = mock_slack

        mock_release = mocker.patch(f"{_SAV}.release_task")
        mock_get_task = mocker.patch(f"{_SAV}.get_task")
        mock_get_skeleton = mocker.patch(f"{_SAV}.get_skeleton")
        mock_timesheet = mocker.patch(f"{_SAV}.get_task_timesheet_summary")

        # Setup task details
        mock_get_task.side_effect = [
            {
                "task_id": "verify123",
                "task_type": "segmentation_auto_verify",
                "ng_state": {"layers": []},
                "extra_data": {"trace_task_id": "trace123"},
            },
            {"task_id": "trace123", "completed_user_id": "user1"},
        ]

        # Setup skeleton
        mock_skeleton = Mock()
        mock_skeleton.skeleton.path_length.return_value = 2500000  # 2.5 mm
        mock_get_skeleton.return_value = mock_skeleton
        mock_skeleton.segment_id = 12345

        # Setup timesheet
        mock_timesheet.return_value = {"total_seconds": 3661, "formatted_total": "1h 1m 1s"}

        # Setup slack response
        mock_slack.chat_postMessage.return_value = {"ts": "123.456"}

        process_task(
            "verify123",
            "test_project",
            "worker_user",
            2,
            "test-channel",
            1,
            "gs://test",
            [8, 8, 40],
            [16, 16, 40],
            ("user1", "user2"),
        )

        # Verify task was released as pass
        mock_release.assert_called_once_with(
            project_name="test_project",
            task_id="verify123",
            user_id="worker_user",
            completion_status="pass",
        )

        # Verify slack message was sent
        assert mock_slack.chat_postMessage.called
        call_args = mock_slack.chat_postMessage.call_args[1]
        assert "✅" in call_args["text"]
        assert "2.50 mm" in call_args["text"]

    def test_process_task_fail(self, mocker):
        """Test processing a task that fails verification"""
        # Setup mocks
        mock_get_slack_client = mocker.patch(f"{_SAV}.get_slack_client")
        mock_slack = Mock()
        mock_get_slack_client.return_value = mock_slack

        mock_release = mocker.patch(f"{_SAV}.release_task")
        mock_get_task = mocker.patch(f"{_SAV}.get_task")
        mock_get_skeleton = mocker.patch(f"{_SAV}.get_skeleton")
        mock_timesheet = mocker.patch(f"{_SAV}.get_task_timesheet_summary")

        # Setup task details
        mock_get_task.side_effect = [
            {
                "task_id": "verify123",
                "task_type": "segmentation_auto_verify",
                "ng_state": {"layers": []},
                "extra_data": {"trace_task_id": "trace123"},
            },
            {"task_id": "trace123", "completed_user_id": "user1"},
        ]

        # Setup skeleton with short length
        mock_skeleton = Mock()
        mock_skeleton.skeleton.path_length.return_value = 500000  # 0.5 mm
        mock_get_skeleton.return_value = mock_skeleton
        mock_skeleton.segment_id = 12345

        # Setup timesheet
        mock_timesheet.return_value = {"total_seconds": 0, "formatted_total": "0s"}

        # Setup slack response
        mock_slack.chat_postMessage.return_value = {"ts": "123.456"}

        process_task(
            "verify123",
            "test_project",
            "worker_user",
            2,
            "test-channel",
            1,
            "gs://test",
            [8, 8, 40],
            [16, 16, 40],
            ("user1", "user2"),
        )

        # Verify task was released as fail
        mock_release.assert_called_once_with(
            project_name="test_project",
            task_id="verify123",
            user_id="worker_user",
            completion_status="fail",
        )

        # Verify slack messages
        assert mock_slack.chat_postMessage.call_count == 2  # Main + thread
        main_call = mock_slack.chat_postMessage.call_args_list[0][1]
        assert "❌" in main_call["text"]
        assert "0.50 mm" in main_call["text"]

        # Check thread message
        thread_call = mock_slack.chat_postMessage.call_args_list[1][1]
        assert thread_call["thread_ts"] == "123.456"
        assert "<@user1>" in thread_call["text"]
        assert "<@user2>" in thread_call["text"]

    def test_process_task_no_skeleton(self, mocker):
        """Test processing task when skeleton retrieval fails"""
        # Setup mocks
        mock_release = mocker.patch(f"{_SAV}.release_task")
        mock_get_task = mocker.patch(f"{_SAV}.get_task")
        mock_get_skeleton = mocker.patch(f"{_SAV}.get_skeleton")
        mock_timesheet = mocker.patch(f"{_SAV}.get_task_timesheet_summary")

        mock_get_task.side_effect = [
            {
                "task_id": "verify123",
                "task_type": "segmentation_auto_verify",
                "ng_state": {"layers": []},
                "extra_data": {"trace_task_id": "trace123"},
            },
            {"task_id": "trace123", "completed_user_id": "user1"},
        ]

        # Skeleton retrieval fails
        mock_get_skeleton.return_value = None

        mock_timesheet.return_value = {"total_seconds": 0, "formatted_total": "0s"}

        process_task(
            "verify123",
            "test_project",
            "worker_user",
            2,
            None,
            1,  # No slack channel
            "gs://test",
            [8, 8, 40],
            [16, 16, 40],
            (),
        )

        # Should still release as fail
        mock_release.assert_called_once_with(
            project_name="test_project",
            task_id="verify123",
            user_id="worker_user",
            completion_status="fail",
        )

    def test_process_task_missing_trace_id(self, mocker):
        """Test processing task with missing trace_task_id"""
        mock_get_task = mocker.patch(f"{_SAV}.get_task")

        mock_get_task.return_value = {
            "task_id": "verify123",
            "task_type": "segmentation_auto_verify",
            "ng_state": {"layers": []},
            "extra_data": {},  # Missing trace_task_id
        }

        # Should return early without processing
        process_task(
            "verify123",
            "test_project",
            "worker_user",
            2,
            None,
            1,
            "gs://test",
            [8, 8, 40],
            [16, 16, 40],
            (),
        )

        # Should only call get_task once
        assert mock_get_task.call_count == 1


class TestGetTaskTimesheetSummary:
    """Test get_task_timesheet_summary function"""

    def test_format_duration(self):
        """Test format_duration functionality via get_task_timesheet_summary"""
        # We can't test format_duration directly as it's nested,
        # but we can verify it works through the main function
        # Tested via test_get_task_timesheet_summary

    def test_get_task_timesheet_summary(self, mocker):
        """Test getting timesheet summary"""
        mock_session_context = mocker.patch(f"{_SAV}.get_session_context")

        # Create mock timesheets
        timesheet1 = Mock(spec=TimesheetModel)
        timesheet1.seconds_spent = 1800
        timesheet1.user = "user1"

        timesheet2 = Mock(spec=TimesheetModel)
        timesheet2.seconds_spent = 3600
        timesheet2.user = "user2"

        timesheet3 = Mock(spec=TimesheetModel)
        timesheet3.seconds_spent = 600
        timesheet3.user = "user1"

        # Setup session mock
        mock_session = Mock()
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            timesheet1,
            timesheet2,
            timesheet3,
        ]
        mock_session_context.return_value.__enter__.return_value = mock_session

        result = get_task_timesheet_summary("test_project", "task123")

        assert result["total_seconds"] == 6000
        assert result["user_breakdown"] == {"user1": 2400, "user2": 3600}
        assert result["formatted_total"] == "1h 40m 0s"
        # Check formatted breakdown contains the users and times
        breakdown_str = " ".join(result["formatted_breakdown"])
        assert "user1" in breakdown_str
        assert "40m 0s" in breakdown_str
        assert "user2" in breakdown_str
        assert "1h 0m 0s" in breakdown_str

    def test_get_task_timesheet_summary_no_entries(self, mocker):
        """Test timesheet summary with no entries"""
        mock_session_context = mocker.patch(f"{_SAV}.get_session_context")

        mock_session = Mock()
        mock_session.execute.return_value.scalars.return_value.all.return_value = []
        mock_session_context.return_value.__enter__.return_value = mock_session

        result = get_task_timesheet_summary("test_project", "task123")

        assert result["total_seconds"] == 0
        assert not result["user_breakdown"]
        assert result["formatted_total"] == "0s"
        assert not result["formatted_breakdown"]


class TestRunWorker:
    """Test run_worker function"""

    def test_run_worker_with_tasks(self, mocker):
        """Test worker processing tasks"""
        # Setup mocks
        mocker.patch(f"{_SAV}.time.sleep")
        mock_process_task = mocker.patch(f"{_SAV}.process_task")
        mock_start_task = mocker.patch(f"{_SAV}.start_task")

        # Simulate getting a task then keyboard interrupt
        mock_start_task.side_effect = ["task123", KeyboardInterrupt]

        runner = CliRunner()
        result = runner.invoke(
            run_worker,
            [
                "--user_id",
                "worker1",
                "--project_name",
                "test_project",
                "--min_skeleton_length_mm",
                "2",
                "--graphene_path",
                "gs://test",
                "--ng_resolution",
                "8",
                "8",
                "40",
                "--sv_resolution",
                "16",
                "16",
                "40",
            ],
        )

        # Should have processed one task
        mock_process_task.assert_called_once()
        assert "Tasks Processed: 1" in result.output

    def test_run_worker_no_tasks(self, mocker):
        """Test worker when no tasks available"""
        # Setup mocks
        mock_sleep = mocker.patch(f"{_SAV}.time.sleep")
        mock_start_task = mocker.patch(f"{_SAV}.start_task")

        # Return None (no tasks) then keyboard interrupt
        mock_start_task.side_effect = [None, KeyboardInterrupt]

        runner = CliRunner()
        result = runner.invoke(
            run_worker,
            [
                "--user_id",
                "worker1",
                "--project_name",
                "test_project",
                "--graphene_path",
                "gs://test",
                "--ng_resolution",
                "8",
                "8",
                "40",
                "--sv_resolution",
                "16",
                "16",
                "40",
            ],
        )

        # Should have slept
        mock_sleep.assert_called_with(5.0)
        assert "Tasks Processed: 0" in result.output

    def test_run_worker_with_error(self, mocker):
        """Test worker handling errors during processing"""
        # Setup mocks
        mock_sleep = mocker.patch(f"{_SAV}.time.sleep")
        mock_process_task = mocker.patch(f"{_SAV}.process_task")
        mock_start_task = mocker.patch(f"{_SAV}.start_task")

        mock_start_task.side_effect = ["task123", None, KeyboardInterrupt]
        mock_process_task.side_effect = Exception("Processing error")

        runner = CliRunner()
        result = runner.invoke(
            run_worker,
            [
                "--user_id",
                "worker1",
                "--project_name",
                "test_project",
                "--graphene_path",
                "gs://test",
                "--ng_resolution",
                "8",
                "8",
                "40",
                "--sv_resolution",
                "16",
                "16",
                "40",
                "--slack_channel",
                "test-channel",
                "--slack_users",
                "user1",
                "--slack_users",
                "user2",
            ],
        )

        # Should continue after error
        assert mock_sleep.called
        assert "Error processing task" in result.output

    def test_run_worker_cli_options(self):
        """Test CLI options parsing"""
        runner = CliRunner()
        result = runner.invoke(run_worker, ["--help"])

        assert result.exit_code == 0
        assert "--user_id" in result.output
        assert "--project_name" in result.output
        assert "--polling_period" in result.output
        assert "--min_skeleton_length_mm" in result.output
        assert "--slack_channel" in result.output
        assert "--graphene_path" in result.output
        assert "--ng_resolution" in result.output
        assert "--sv_resolution" in result.output
        assert "--slack_users" in result.output
