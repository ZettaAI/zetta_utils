# pylint: disable=redefined-outer-name,unused-argument
import time

import pytest

from zetta_utils.task_management.exceptions import UserValidationError
from zetta_utils.task_management.project import get_collection
from zetta_utils.task_management.subtask import start_subtask
from zetta_utils.task_management.timesheet import submit_timesheet
from zetta_utils.task_management.types import User
from zetta_utils.task_management.user import create_user


def test_submit_timesheet_success(project_name, existing_user, existing_subtask):
    """Test successful timesheet submission"""
    # Start the subtask
    start_subtask(project_name, "test_user_1", "subtask_1")

    submit_timesheet(project_name, "test_user_1", 3600, "subtask_1")

    # Verify last_leased_ts was updated
    subtask_doc = get_collection(project_name, "subtasks").document("subtask_1").get()
    assert subtask_doc.exists
    subtask_data = subtask_doc.to_dict()
    assert subtask_data["last_leased_ts"] > time.time() - 10  # Updated within last 10 seconds


def test_submit_timesheet_update_existing(project_name, existing_user, existing_subtask):
    """Test updating an existing timesheet entry with additional duration"""
    # Start the subtask
    start_subtask(project_name, "test_user_1", "subtask_1")

    # Submit first timesheet entry
    initial_duration = 3600
    submit_timesheet(project_name, "test_user_1", initial_duration, "subtask_1")

    # Submit second timesheet entry for the same subtask
    additional_duration = 1800
    submit_timesheet(project_name, "test_user_1", additional_duration, "subtask_1")

    # Verify the timesheet entry was updated with the combined duration
    timesheet_doc = (
        get_collection(project_name, "timesheets").document("test_user_1_subtask_1").get()
    )
    assert timesheet_doc.exists
    timesheet_data = timesheet_doc.to_dict()
    assert timesheet_data["duration_seconds"] == initial_duration + additional_duration
    assert timesheet_data["user_id"] == "test_user_1"
    assert timesheet_data["subtask_id"] == "subtask_1"
    assert "last_updated_ts" in timesheet_data
    assert timesheet_data["last_updated_ts"] > time.time() - 10  # Updated within last 10 seconds


def test_submit_timesheet_no_active_subtask(project_name, existing_user):
    """Test submitting timesheet without an active subtask"""
    with pytest.raises(UserValidationError, match="User does not have an active subtask"):
        submit_timesheet(project_name, "test_user_1", 3600, "subtask_1")


def test_submit_timesheet_no_user_subtask(project_name, existing_user, existing_subtask):
    with pytest.raises(UserValidationError, match="User does not have an active subtask"):
        submit_timesheet(project_name, "test_user_1", 3600, "subtask_1")


def test_submit_timesheet_negative_duration(project_name, existing_user, existing_subtask):
    """Test submitting timesheet with negative duration"""
    start_subtask(project_name, "test_user_1", "subtask_1")

    with pytest.raises(ValueError, match="Duration must be positive"):
        submit_timesheet(project_name, "test_user_1", -3600, "subtask_1")


def test_submit_timesheet_nonexistent_user(project_name, existing_subtask):
    """Test submitting timesheet for a user that doesn't exist"""
    with pytest.raises(UserValidationError, match="User nonexistent_user not found"):
        submit_timesheet(project_name, "nonexistent_user", 3600, "subtask_1")


def test_submit_timesheet_nonexistent_subtask(project_name, existing_user):
    """Test submitting timesheet when user has nonexistent subtask"""
    # Manually set user's active_subtask to a nonexistent one
    user_ref = get_collection(project_name, "users").document("test_user_1")
    user_ref.update({"active_subtask": "nonexistent_subtask"})

    with pytest.raises(UserValidationError, match="Subtask nonexistent_subtask not found"):
        submit_timesheet(project_name, "test_user_1", 3600, "nonexistent_subtask")


def test_submit_timesheet_wrong_user(project_name, existing_user, existing_subtask):
    """Test submitting timesheet for subtask assigned to different user"""
    # Start subtask with one user
    start_subtask(project_name, "test_user_1", "subtask_1")

    # Create another user
    other_user = User(
        **{
            "user_id": "other_user",
            "hourly_rate": 50.0,
            "active_subtask": "subtask_1",  # Manually set to first user's subtask
            "qualified_subtask_types": ["test_type"],
        }
    )
    create_user(project_name, other_user)

    with pytest.raises(UserValidationError, match="Subtask not assigned to this user"):
        submit_timesheet(project_name, "other_user", 3600, "subtask_1")


def test_submit_timesheet_mismatched_subtask_id(project_name, existing_user, existing_subtask):
    """Test submitting timesheet with a subtask_id that doesn't match user's active subtask"""
    # Start subtask with one ID
    start_subtask(project_name, "test_user_1", "subtask_1")

    # Try to submit timesheet with a different subtask_id
    with pytest.raises(
        UserValidationError,
        match="Provided subtask_id subtask_2 does not match user's active subtask subtask_1",
    ):
        submit_timesheet(project_name, "test_user_1", 3600, "subtask_2")
