# pylint: disable=redefined-outer-name,unused-argument

import pytest
from sqlalchemy import select

from zetta_utils.task_management.db.models import TimesheetModel
from zetta_utils.task_management.exceptions import UserValidationError
from zetta_utils.task_management.task import start_task
from zetta_utils.task_management.timesheet import submit_timesheet
from zetta_utils.task_management.types import User
from zetta_utils.task_management.user import create_user, update_user


def test_submit_timesheet_success(
    clean_db, project_name, db_session, existing_user, existing_task
):
    """Test successful timesheet submission"""
    # Start the task
    start_task(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
    )

    submit_timesheet(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        duration_seconds=3600,
        task_id="task_1",
    )

    # TODO: Add verification once we have a get_timesheet function


def test_submit_timesheet_update_existing(
    clean_db, project_name, db_session, existing_user, existing_task
):
    """Test updating an existing timesheet entry with additional duration"""
    # Start the task
    start_task(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
    )

    # Submit first timesheet entry
    initial_duration = 3600
    submit_timesheet(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        duration_seconds=initial_duration,
        task_id="task_1",
    )

    # Submit second timesheet entry for the same task
    additional_duration = 1800
    submit_timesheet(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        duration_seconds=additional_duration,
        task_id="task_1",
    )

    # TODO: Add verification once we have a get_timesheet function


def test_submit_timesheet_no_active_task(clean_db, project_name, db_session, existing_user):
    """Test submitting timesheet without an active task"""
    with pytest.raises(UserValidationError, match="User does not have an active task"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=3600,
            task_id="task_1",
        )


def test_submit_timesheet_no_user_task(
    clean_db, project_name, db_session, existing_user, existing_task
):
    with pytest.raises(UserValidationError, match="User does not have an active task"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=3600,
            task_id="task_1",
        )


def test_submit_timesheet_negative_duration(
    clean_db, project_name, db_session, existing_user, existing_task
):
    """Test submitting timesheet with negative duration"""
    start_task(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
    )

    with pytest.raises(ValueError, match="Duration must be positive"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=-3600,
            task_id="task_1",
        )


def test_submit_timesheet_nonexistent_user(clean_db, project_name, db_session, existing_task):
    """Test submitting timesheet for a user that doesn't exist"""
    with pytest.raises(UserValidationError, match="User nonexistent_user not found"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="nonexistent_user",
            duration_seconds=3600,
            task_id="task_1",
        )


def test_submit_timesheet_nonexistent_task(clean_db, project_name, db_session, existing_user):
    """Test submitting timesheet when user has nonexistent task"""
    # This test needs to be adjusted since we can't manually set active_task without SQL
    # We'll test when the user claims to have a different active task
    with pytest.raises(UserValidationError, match="User does not have an active task"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=3600,
            task_id="nonexistent_task",
        )


def test_submit_timesheet_wrong_user(
    clean_db, project_name, db_session, existing_user, existing_task
):
    """Test submitting timesheet for task assigned to different user"""
    # Start task with one user
    start_task(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
    )

    # Create another user
    other_user = User(
        **{
            "user_id": "other_user",
            "hourly_rate": 50.0,
            "active_task": "",
            "qualified_task_types": ["segmentation_proofread"],
        }
    )
    create_user(db_session=db_session, project_name=project_name, data=other_user)

    with pytest.raises(UserValidationError, match="User does not have an active task"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="other_user",
            duration_seconds=3600,
            task_id="task_1",
        )


def test_submit_timesheet_mismatched_task_id(
    clean_db, project_name, db_session, existing_user, existing_task
):
    """Test submitting timesheet with a task_id that doesn't match user's active task"""
    # Start task with one ID
    start_task(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
    )

    # Try to submit timesheet with a different task_id - this should fail since
    # user's active task is task_1 but we're submitting for task_2
    with pytest.raises(
        UserValidationError,
        match="Provided task_id task_2 does not match user's active task task_1",
    ):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=3600,
            task_id="task_2",
        )


def test_submit_timesheet_task_not_assigned_to_user(
    clean_db, project_name, db_session, existing_user, existing_task
):
    """Test that timesheet submission fails when user does not have an active task"""
    # Create another user
    create_user(
        db_session=db_session,
        project_name=project_name,
        data={
            "user_id": "another_user",
            "hourly_rate": 50.0,
            "active_task": "",
            "qualified_task_types": ["segmentation_proofread"],
        },
    )

    # Start the task with a different user
    start_task(
        db_session=db_session,
        project_name=project_name,
        user_id="another_user",
        task_id="task_1",
    )

    # Try to submit timesheet as the original user (should fail)
    with pytest.raises(UserValidationError, match="User does not have an active task"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=3600.0,
            task_id="task_1",
        )


def test_submit_timesheet_task_assigned_to_different_user(
    clean_db, project_name, db_session, existing_user, existing_task
):
    """Test that timesheet submission fails when task is assigned to a different user"""
    # Create another user
    create_user(
        db_session=db_session,
        project_name=project_name,
        data={
            "user_id": "another_user",
            "hourly_rate": 50.0,
            "active_task": "",
            "qualified_task_types": ["segmentation_proofread"],
        },
    )

    # Start the task with another user
    start_task(
        db_session=db_session,
        project_name=project_name,
        user_id="another_user",
        task_id="task_1",
    )

    # Update the first user to have this task as active (simulating inconsistent state)
    update_user(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        data={"active_task": "task_1"},
    )

    # Try to submit timesheet as the first user (should fail)
    with pytest.raises(UserValidationError, match="Task not assigned to this user"):
        submit_timesheet(
            db_session=db_session,
            project_name=project_name,
            user_id="test_user",
            duration_seconds=3600.0,
            task_id="task_1",
        )


def test_submit_timesheet_database_failure(
    clean_db, project_name, db_session, existing_user, existing_task, mocker
):
    """Test that timesheet submission handles database failures gracefully"""
    # Start the task
    start_task(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
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
            task_id="task_1",
        )

    # Verify rollback was called
    mock_rollback.assert_called_once()


def test_submit_timesheet_session_add_failure(
    clean_db, project_name, db_session, existing_user, existing_task, mocker
):
    # Start the task
    start_task(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
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
            task_id="task_1",
        )

    # Verify rollback was called (this should cover line 86)
    mock_rollback.assert_called_once()


def test_submit_timesheet_update_existing_entry_verification(
    clean_db, project_name, db_session, existing_user, existing_task
):
    start_task(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
    )

    # Submit first timesheet entry
    initial_duration = 3600
    submit_timesheet(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        duration_seconds=initial_duration,
        task_id="task_1",
    )

    # Verify one timesheet entry exists with correct duration
    query = (
        select(TimesheetModel)
        .where(TimesheetModel.project_name == project_name)
        .where(TimesheetModel.user == "test_user")
        .where(TimesheetModel.task_id == "task_1")
    )
    timesheets = db_session.execute(query).scalars().all()
    assert len(timesheets) == 1
    assert timesheets[0].seconds_spent == initial_duration

    # Submit second timesheet entry for the same task (should update existing)
    additional_duration = 1800
    submit_timesheet(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        duration_seconds=additional_duration,
        task_id="task_1",
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
        task_id="task_1",
    )

    # Verify accumulation continues to work
    timesheets = db_session.execute(query).scalars().all()
    assert len(timesheets) == 1
    assert timesheets[0].seconds_spent == initial_duration + additional_duration + third_duration
