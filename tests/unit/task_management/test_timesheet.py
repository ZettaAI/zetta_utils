# pylint: disable=redefined-outer-name,unused-argument

import pytest
from sqlalchemy import select

from zetta_utils.task_management.db.models import TimesheetModel
from zetta_utils.task_management.exceptions import UserValidationError
from zetta_utils.task_management.subtask import start_subtask
from zetta_utils.task_management.timesheet import submit_timesheet
from zetta_utils.task_management.types import User
from zetta_utils.task_management.user import create_user, update_user


def test_submit_timesheet_success(
    clean_db, project_name, db_session, existing_user, existing_subtask
):
    """Test successful timesheet submission"""
    # Start the subtask
    start_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
    )

    submit_timesheet(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        duration_seconds=3600,
        subtask_id="subtask_1",
    )

    # TODO: Add verification once we have a get_timesheet function


def test_submit_timesheet_update_existing(
    clean_db, project_name, db_session, existing_user, existing_subtask
):
    """Test updating an existing timesheet entry with additional duration"""
    # Start the subtask
    start_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
    )

    # Submit first timesheet entry
    initial_duration = 3600
    submit_timesheet(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        duration_seconds=initial_duration,
        subtask_id="subtask_1",
    )

    # Submit second timesheet entry for the same subtask
    additional_duration = 1800
    submit_timesheet(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        duration_seconds=additional_duration,
        subtask_id="subtask_1",
    )

    # TODO: Add verification once we have a get_timesheet function


def test_submit_timesheet_no_active_subtask(clean_db, project_name, db_session, existing_user):
    """Test submitting timesheet without an active subtask"""
    with pytest.raises(UserValidationError, match="User does not have an active subtask"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=3600,
            subtask_id="subtask_1",
        )


def test_submit_timesheet_no_user_subtask(
    clean_db, project_name, db_session, existing_user, existing_subtask
):
    with pytest.raises(UserValidationError, match="User does not have an active subtask"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=3600,
            subtask_id="subtask_1",
        )


def test_submit_timesheet_negative_duration(
    clean_db, project_name, db_session, existing_user, existing_subtask
):
    """Test submitting timesheet with negative duration"""
    start_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
    )

    with pytest.raises(ValueError, match="Duration must be positive"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=-3600,
            subtask_id="subtask_1",
        )


def test_submit_timesheet_nonexistent_user(clean_db, project_name, db_session, existing_subtask):
    """Test submitting timesheet for a user that doesn't exist"""
    with pytest.raises(UserValidationError, match="User nonexistent_user not found"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="nonexistent_user",
            duration_seconds=3600,
            subtask_id="subtask_1",
        )


def test_submit_timesheet_nonexistent_subtask(clean_db, project_name, db_session, existing_user):
    """Test submitting timesheet when user has nonexistent subtask"""
    # This test needs to be adjusted since we can't manually set active_subtask without SQL
    # We'll test when the user claims to have a different active subtask
    with pytest.raises(UserValidationError, match="User does not have an active subtask"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=3600,
            subtask_id="nonexistent_subtask",
        )


def test_submit_timesheet_wrong_user(
    clean_db, project_name, db_session, existing_user, existing_subtask
):
    """Test submitting timesheet for subtask assigned to different user"""
    # Start subtask with one user
    start_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
    )

    # Create another user
    other_user = User(
        **{
            "user_id": "other_user",
            "hourly_rate": 50.0,
            "active_subtask": "",
            "qualified_subtask_types": ["segmentation_proofread"],
        }
    )
    create_user(db_session=db_session, project_name=project_name, data=other_user)

    with pytest.raises(UserValidationError, match="User does not have an active subtask"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="other_user",
            duration_seconds=3600,
            subtask_id="subtask_1",
        )


def test_submit_timesheet_mismatched_subtask_id(
    clean_db, project_name, db_session, existing_user, existing_subtask
):
    """Test submitting timesheet with a subtask_id that doesn't match user's active subtask"""
    # Start subtask with one ID
    start_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
    )

    # Try to submit timesheet with a different subtask_id - this should fail since
    # user's active subtask is subtask_1 but we're submitting for subtask_2
    with pytest.raises(
        UserValidationError,
        match="Provided subtask_id subtask_2 does not match user's active subtask subtask_1",
    ):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=3600,
            subtask_id="subtask_2",
        )


def test_submit_timesheet_subtask_not_assigned_to_user(
    clean_db, project_name, db_session, existing_user, existing_subtask
):
    """Test that timesheet submission fails when user does not have an active subtask"""
    # Create another user
    create_user(
        db_session=db_session,
        project_name=project_name,
        data={
            "user_id": "another_user",
            "hourly_rate": 50.0,
            "active_subtask": "",
            "qualified_subtask_types": ["segmentation_proofread"],
        },
    )

    # Start the subtask with a different user
    start_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="another_user",
        subtask_id="subtask_1",
    )

    # Try to submit timesheet as the original user (should fail)
    with pytest.raises(UserValidationError, match="User does not have an active subtask"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=3600.0,
            subtask_id="subtask_1",
        )


def test_submit_timesheet_subtask_assigned_to_different_user(
    clean_db, project_name, db_session, existing_user, existing_subtask
):
    """Test that timesheet submission fails when subtask is assigned to a different user"""
    # Create another user
    create_user(
        db_session=db_session,
        project_name=project_name,
        data={
            "user_id": "another_user",
            "hourly_rate": 50.0,
            "active_subtask": "",
            "qualified_subtask_types": ["segmentation_proofread"],
        },
    )

    # Start the subtask with another user
    start_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="another_user",
        subtask_id="subtask_1",
    )

    # Update the first user to have this subtask as active (simulating inconsistent state)
    update_user(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        data={"active_subtask": "subtask_1"},
    )

    # Try to submit timesheet as the first user (should fail)
    with pytest.raises(UserValidationError, match="Subtask not assigned to this user"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=3600.0,
            subtask_id="subtask_1",
        )


def test_submit_timesheet_database_failure(
    clean_db, project_name, db_session, existing_user, existing_subtask, mocker
):
    """Test that timesheet submission handles database failures gracefully"""
    # Start the subtask
    start_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
    )

    # Mock the commit to raise an exception
    mock_commit = mocker.patch.object(db_session, "commit")
    mock_rollback = mocker.patch.object(db_session, "rollback")
    mock_commit.side_effect = Exception("Database connection lost")

    with pytest.raises(RuntimeError, match="Failed to submit timesheet: Database connection lost"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=3600.0,
            subtask_id="subtask_1",
        )

    # Verify rollback was called
    mock_rollback.assert_called_once()


def test_submit_timesheet_session_add_failure(
    clean_db, project_name, db_session, existing_user, existing_subtask, mocker
):
    # Start the subtask
    start_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
    )

    # Mock session.add to raise an exception to trigger the exception handling path
    mock_add = mocker.patch.object(db_session, "add")
    mock_rollback = mocker.patch.object(db_session, "rollback")
    mock_add.side_effect = Exception("Database constraint violation")

    with pytest.raises(
        RuntimeError, match="Failed to submit timesheet: Database constraint violation"
    ):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=3600.0,
            subtask_id="subtask_1",
        )

    # Verify rollback was called (this should cover line 86)
    mock_rollback.assert_called_once()


def test_submit_timesheet_update_existing_entry_verification(
    clean_db, project_name, db_session, existing_user, existing_subtask
):
    start_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
    )

    # Submit first timesheet entry
    initial_duration = 3600
    submit_timesheet(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        duration_seconds=initial_duration,
        subtask_id="subtask_1",
    )

    # Verify one timesheet entry exists with correct duration
    query = (
        select(TimesheetModel)
        .where(TimesheetModel.project_name == project_name)
        .where(TimesheetModel.user == "test_user")
        .where(TimesheetModel.subtask_id == "subtask_1")
    )
    timesheets = db_session.execute(query).scalars().all()
    assert len(timesheets) == 1
    assert timesheets[0].seconds_spent == initial_duration

    # Submit second timesheet entry for the same subtask (should update existing)
    additional_duration = 1800
    submit_timesheet(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        duration_seconds=additional_duration,
        subtask_id="subtask_1",
    )

    # Verify still only one timesheet entry exists with accumulated duration
    timesheets = db_session.execute(query).scalars().all()
    assert len(timesheets) == 1  # Should still be only one entry
    assert (
        timesheets[0].seconds_spent == initial_duration + additional_duration
    )  # Should be accumulated

    # Submit third timesheet entry to further verify accumulation
    third_duration = 900
    submit_timesheet(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        duration_seconds=third_duration,
        subtask_id="subtask_1",
    )

    # Verify accumulation continues to work
    timesheets = db_session.execute(query).scalars().all()
    assert len(timesheets) == 1
    assert timesheets[0].seconds_spent == initial_duration + additional_duration + third_duration
