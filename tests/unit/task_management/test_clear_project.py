"""Tests for clear_project module"""

# pylint: disable=unused-argument,redefined-outer-name

import pytest
from sqlalchemy.exc import SQLAlchemyError

from zetta_utils.task_management.clear_project import (
    clear_project_data,
    clear_project_task_types,
    clear_project_users,
)
from zetta_utils.task_management.db.models import (
    DependencyModel,
    TaskModel,
    TaskTypeModel,
    TimesheetModel,
    UserModel,
)
from zetta_utils.task_management.dependency import create_dependency
from zetta_utils.task_management.task import create_task, start_task
from zetta_utils.task_management.task_type import create_task_type
from zetta_utils.task_management.timesheet import submit_timesheet
from zetta_utils.task_management.types import Dependency, Task, TaskType, User
from zetta_utils.task_management.user import create_user


def test_clear_project_data_success(
    clean_db, project_name, existing_user, existing_task_type, db_session
):
    """Test clearing all project data successfully"""
    # Create test data
    task1: Task = {
        "task_id": "task1",
        "task_type": "segmentation_proofread",
        "completion_status": "",
        "assigned_user_id": "",
        "active_user_id": "",
        "completed_user_id": "",
        "ng_state": {},
        "ng_state_initial": {},
        "priority": 1,
        "batch_id": "test_batch",
        "last_leased_ts": 0,
        "is_active": True,
    }
    task2: Task = {
        "task_id": "task2",
        "task_type": "segmentation_proofread",
        "completion_status": "",
        "assigned_user_id": "",
        "active_user_id": "",
        "completed_user_id": "",
        "ng_state": {},
        "ng_state_initial": {},
        "priority": 2,
        "batch_id": "test_batch",
        "last_leased_ts": 0,
        "is_active": True,
    }

    create_task(project_name=project_name, data=task1, db_session=db_session)
    create_task(project_name=project_name, data=task2, db_session=db_session)

    # Create dependency
    dep: Dependency = {
        "dependency_id": "dep1",
        "task_id": "task2",
        "dependent_on_task_id": "task1",
        "is_satisfied": False,
        "required_completion_status": "done",
    }
    create_dependency(project_name=project_name, data=dep, db_session=db_session)

    # Start task and create timesheet
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task1",
        db_session=db_session,
    )
    submit_timesheet(
        project_name=project_name,
        user_id="test_user",
        task_id="task1",
        duration_seconds=100,
        db_session=db_session,
    )

    # Clear project data
    deleted_counts = clear_project_data(project_name=project_name, db_session=db_session)

    # Verify counts
    assert deleted_counts["timesheets"] == 1
    assert deleted_counts["dependencies"] == 1
    assert deleted_counts["tasks"] == 2

    # Verify data is actually deleted
    assert db_session.query(TaskModel).filter_by(project_name=project_name).count() == 0
    assert db_session.query(DependencyModel).filter_by(project_name=project_name).count() == 0
    assert db_session.query(TimesheetModel).filter_by(project_name=project_name).count() == 0


def test_clear_project_data_empty_project(clean_db, project_name, db_session):
    """Test clearing data from an empty project"""
    deleted_counts = clear_project_data(project_name=project_name, db_session=db_session)

    assert deleted_counts["timesheets"] == 0
    assert deleted_counts["dependencies"] == 0
    assert deleted_counts["tasks"] == 0


def test_clear_project_data_rollback_on_error(clean_db, project_name, db_session, mocker):
    """Test that errors are properly raised"""
    # Mock session.execute to raise an error
    mocker.patch.object(db_session, "execute", side_effect=SQLAlchemyError("Test error"))

    with pytest.raises(SQLAlchemyError, match="Test error"):
        clear_project_data(project_name=project_name, db_session=db_session)


def test_clear_project_users_success(clean_db, project_name, db_session):
    """Test clearing all users for a project"""
    # Create test users
    user1: User = {
        "user_id": "user1",
        "hourly_rate": 50.0,
        "active_task": "",
        "qualified_task_types": [],
    }
    user2: User = {
        "user_id": "user2",
        "hourly_rate": 60.0,
        "active_task": "",
        "qualified_task_types": [],
    }

    create_user(project_name=project_name, data=user1, db_session=db_session)
    create_user(project_name=project_name, data=user2, db_session=db_session)

    # Clear users
    count = clear_project_users(project_name=project_name, db_session=db_session)

    assert count == 2
    assert db_session.query(UserModel).filter_by(project_name=project_name).count() == 0


def test_clear_project_users_empty_project(clean_db, project_name, db_session):
    """Test clearing users from a project with no users"""
    count = clear_project_users(project_name=project_name, db_session=db_session)
    assert count == 0


def test_clear_project_users_rollback_on_error(clean_db, project_name, db_session, mocker):
    """Test that errors are properly raised"""
    # Mock session.execute to raise an error
    mocker.patch.object(db_session, "execute", side_effect=SQLAlchemyError("Test error"))

    with pytest.raises(SQLAlchemyError, match="Test error"):
        clear_project_users(project_name=project_name, db_session=db_session)


def test_clear_project_task_types_success(clean_db, project_name, db_session):
    """Test clearing all task types for a project"""
    # Create test task types
    type1: TaskType = {"task_type": "type1", "completion_statuses": ["pending", "completed"]}
    type2: TaskType = {"task_type": "type2", "completion_statuses": ["pending", "completed"]}

    create_task_type(project_name=project_name, data=type1, db_session=db_session)
    create_task_type(project_name=project_name, data=type2, db_session=db_session)

    # Clear task types
    count = clear_project_task_types(project_name=project_name, db_session=db_session)

    assert count == 2
    assert db_session.query(TaskTypeModel).filter_by(project_name=project_name).count() == 0


def test_clear_project_task_types_empty_project(clean_db, project_name, db_session):
    """Test clearing task types from a project with no task types"""
    count = clear_project_task_types(project_name=project_name, db_session=db_session)
    assert count == 0


def test_clear_project_task_types_rollback_on_error(clean_db, project_name, db_session, mocker):
    """Test that errors are properly raised"""
    # Mock session.execute to raise an error
    mocker.patch.object(db_session, "execute", side_effect=SQLAlchemyError("Test error"))

    with pytest.raises(SQLAlchemyError, match="Test error"):
        clear_project_task_types(project_name=project_name, db_session=db_session)


def test_clear_project_data_multiple_projects(clean_db, db_session):
    """Test that clearing one project doesn't affect another"""
    project1 = "project1"
    project2 = "project2"

    # Create data for both projects
    for project in [project1, project2]:
        user: User = {
            "user_id": f"user_{project}",
            "hourly_rate": 50.0,
            "active_task": "",
            "qualified_task_types": [],
        }
        create_user(project_name=project, data=user, db_session=db_session)

        task_type: TaskType = {
            "task_type": f"type_{project}",
            "completion_statuses": ["pending", "completed"],
        }
        create_task_type(project_name=project, data=task_type, db_session=db_session)

        task: Task = {
            "task_id": f"task_{project}",
            "task_type": f"type_{project}",
            "completion_status": "",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": {},
            "ng_state_initial": {},
            "priority": 1,
            "batch_id": "test_batch",
            "last_leased_ts": 0,
            "is_active": True,
        }
        create_task(project_name=project, data=task, db_session=db_session)

    # Clear only project1
    clear_project_data(project_name=project1, db_session=db_session)

    # Verify project1 data is gone
    assert db_session.query(TaskModel).filter_by(project_name=project1).count() == 0

    # Verify project2 data still exists
    assert db_session.query(TaskModel).filter_by(project_name=project2).count() == 1
    assert db_session.query(UserModel).filter_by(project_name=project2).count() == 1
    assert db_session.query(TaskTypeModel).filter_by(project_name=project2).count() == 1
