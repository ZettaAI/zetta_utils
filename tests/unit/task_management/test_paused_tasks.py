# pylint: disable=redefined-outer-name,unused-argument
"""Tests for paused task functionality."""

from zetta_utils.task_management.project import create_project
from zetta_utils.task_management.task import (
    create_task,
    get_paused_tasks_by_user,
    update_task,
)
from zetta_utils.task_management.task_type import create_task_type
from zetta_utils.task_management.types import Task, TaskType, TaskUpdate
from zetta_utils.task_management.user import create_user


def test_get_paused_tasks_by_user(clean_db, postgres_session, project_name):
    """Test getting all paused tasks for a specific user."""

    # Create the project first
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=postgres_session,
    )

    # Create task type
    create_task_type(
        project_name=project_name,
        data=TaskType(task_type="test_type", completion_statuses=["done", "failed"]),
        db_session=postgres_session,
    )

    # Create users
    user1_id = "user1"
    user2_id = "user2"
    create_user(
        project_name=project_name,
        data={
            "user_id": user1_id,
            "hourly_rate": 50.0,
            "active_task": "",
            "qualified_task_types": ["test_type"],
        },
        db_session=postgres_session,
    )
    create_user(
        project_name=project_name,
        data={
            "user_id": user2_id,
            "hourly_rate": 50.0,
            "active_task": "",
            "qualified_task_types": ["test_type"],
        },
        db_session=postgres_session,
    )

    # Create tasks assigned to user1
    # Paused tasks for user1
    paused_tasks_user1 = []
    for i in range(3):
        task_id = f"paused_user1_{i}"
        task = Task(
            task_id=task_id,
            task_type="test_type",
            priority=10 - i,  # Different priorities
            batch_id="test_batch",
            ng_state={"test": "state"},
            ng_state_initial={"test": "state"},
            completion_status="",
            assigned_user_id=user1_id,
            active_user_id="",
            completed_user_id="",
            last_leased_ts=0.0,
            is_active=True,
            is_paused=True,
        )
        create_task(project_name=project_name, data=task, db_session=postgres_session)
        paused_tasks_user1.append(task_id)

    # Active (not paused) tasks for user1
    for i in range(2):
        task = Task(
            task_id=f"active_user1_{i}",
            task_type="test_type",
            priority=5,
            batch_id="test_batch",
            ng_state={"test": "state"},
            ng_state_initial={"test": "state"},
            completion_status="",
            assigned_user_id=user1_id,
            active_user_id="",
            completed_user_id="",
            last_leased_ts=0.0,
            is_active=True,
            is_paused=False,
        )
        create_task(project_name=project_name, data=task, db_session=postgres_session)

    # Paused tasks for user2
    for i in range(2):
        task = Task(
            task_id=f"paused_user2_{i}",
            task_type="test_type",
            priority=7,
            batch_id="test_batch",
            ng_state={"test": "state"},
            ng_state_initial={"test": "state"},
            completion_status="",
            assigned_user_id=user2_id,
            active_user_id="",
            completed_user_id="",
            last_leased_ts=0.0,
            is_active=True,
            is_paused=True,
        )
        create_task(project_name=project_name, data=task, db_session=postgres_session)

    # Completed paused task (should not be returned)
    task = Task(
        task_id="completed_paused",
        task_type="test_type",
        priority=1,
        batch_id="test_batch",
        ng_state={"test": "state"},
        ng_state_initial={"test": "state"},
        completion_status="done",
        assigned_user_id=user1_id,
        active_user_id="",
        completed_user_id=user1_id,
        last_leased_ts=0.0,
        is_active=True,
        is_paused=True,
    )
    create_task(project_name=project_name, data=task, db_session=postgres_session)

    # Get paused tasks for user1
    paused_tasks = get_paused_tasks_by_user(
        project_name=project_name,
        user_id=user1_id,
        db_session=postgres_session,
    )

    # Should return 3 paused tasks for user1
    assert len(paused_tasks) == 3

    # Check that all returned tasks are paused and assigned to user1
    for task in paused_tasks:
        assert task["is_paused"] is True
        assert task["assigned_user_id"] == user1_id
        assert task["completion_status"] == ""
        assert task["is_active"] is True

    # Check ordering (by priority desc, then task_id)
    task_ids = [task["task_id"] for task in paused_tasks]
    priorities = [task["priority"] for task in paused_tasks]

    # Should be ordered by priority descending
    assert priorities == sorted(priorities, reverse=True)

    # Verify the specific tasks returned
    assert set(task_ids) == set(paused_tasks_user1)

    # Get paused tasks for user2
    paused_tasks_user2 = get_paused_tasks_by_user(
        project_name=project_name,
        user_id=user2_id,
        db_session=postgres_session,
    )

    # Should return 2 paused tasks for user2
    assert len(paused_tasks_user2) == 2
    for task in paused_tasks_user2:
        assert task["assigned_user_id"] == user2_id

    # Get paused tasks for non-existent user
    paused_tasks_none = get_paused_tasks_by_user(
        project_name=project_name,
        user_id="non_existent_user",
        db_session=postgres_session,
    )

    # Should return empty list
    assert len(paused_tasks_none) == 0


def test_get_paused_tasks_by_user_empty(clean_db, postgres_session, project_name):
    """Test getting paused tasks when there are none."""

    # Create a user
    create_user(
        project_name=project_name,
        data={
            "user_id": "test_user",
            "hourly_rate": 50.0,
            "active_task": "",
            "qualified_task_types": ["test_type"],
        },
        db_session=postgres_session,
    )

    # Get paused tasks
    paused_tasks = get_paused_tasks_by_user(
        project_name=project_name,
        user_id="test_user",
        db_session=postgres_session,
    )

    # Should return empty list
    assert paused_tasks == []


def test_pause_unpause_task(clean_db, postgres_session, project_name):
    """Test pausing and unpausing tasks."""

    # Create the project first
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=postgres_session,
    )

    # Create task type
    create_task_type(
        project_name=project_name,
        data=TaskType(task_type="test_type", completion_statuses=["done", "failed"]),
        db_session=postgres_session,
    )

    # Create user
    user_id = "test_user"
    create_user(
        project_name=project_name,
        data={
            "user_id": user_id,
            "hourly_rate": 50.0,
            "active_task": "",
            "qualified_task_types": ["test_type"],
        },
        db_session=postgres_session,
    )

    # Create an active task
    task_id = "test_task"
    task = Task(
        task_id=task_id,
        task_type="test_type",
        priority=5,
        batch_id="test_batch",
        ng_state={"test": "state"},
        ng_state_initial={"test": "state"},
        completion_status="",
        assigned_user_id=user_id,
        active_user_id="",
        completed_user_id="",
        last_leased_ts=0.0,
        is_active=True,
        is_paused=False,
    )
    create_task(project_name=project_name, data=task, db_session=postgres_session)

    # Initially, no paused tasks
    paused_tasks = get_paused_tasks_by_user(
        project_name=project_name,
        user_id=user_id,
        db_session=postgres_session,
    )
    assert len(paused_tasks) == 0

    # Pause the task
    update_task(
        project_name=project_name,
        task_id=task_id,
        data=TaskUpdate(is_paused=True),
        db_session=postgres_session,
    )

    # Now should have 1 paused task
    paused_tasks = get_paused_tasks_by_user(
        project_name=project_name,
        user_id=user_id,
        db_session=postgres_session,
    )
    assert len(paused_tasks) == 1
    assert paused_tasks[0]["task_id"] == task_id

    # Unpause the task
    update_task(
        project_name=project_name,
        task_id=task_id,
        data=TaskUpdate(is_paused=False),
        db_session=postgres_session,
    )

    # Should have no paused tasks again
    paused_tasks = get_paused_tasks_by_user(
        project_name=project_name,
        user_id=user_id,
        db_session=postgres_session,
    )
    assert len(paused_tasks) == 0
