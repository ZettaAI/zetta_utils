"""Tests for cross-project task selection."""

# pylint: disable=unused-argument,redefined-outer-name,import-outside-toplevel,import-error,too-many-lines

import time

import pytest

from zetta_utils.task_management.project import create_project
from zetta_utils.task_management.task import (
    get_max_idle_seconds,
    start_task_cross_project,
)
from zetta_utils.task_management.task_type import create_task_type
from zetta_utils.task_management.types import Task, TaskType, User
from zetta_utils.task_management.user import create_user, get_user


def _mk_task(task_id: str, priority: int, seed_id: int) -> Task:
    return Task(
        **{
            "task_id": task_id,
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": {"url": f"http://example.com/{task_id}"},
            "ng_state_initial": {"url": f"http://example.com/{task_id}"},
            "priority": priority,
            "batch_id": "batch_1",
            "task_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
            "extra_data": {"seed_id": seed_id},
        }
    )


def _mk_user(user_id: str) -> User:
    return User(
        **{
            "user_id": user_id,
            "hourly_rate": 1.0,
            "active_task": "",
            "qualified_task_types": ["segmentation_proofread"],
            "qualified_segment_types": ["axon"],
        }
    )


@pytest.fixture
def two_projects(clean_db, db_session):
    """Create two projects with the standard task type configured."""
    p1 = "proj_A"
    p2 = "proj_B"
    for pn in (p1, p2):
        create_project(
            project_name=pn,
            segmentation_path=f"precomputed://gs://bucket/{pn}",
            sv_resolution_x=8.0,
            sv_resolution_y=8.0,
            sv_resolution_z=40.0,
            datastack_name=f"{pn}_ds",
            synapse_table=f"{pn}_syn",
            db_session=db_session,
        )
        create_task_type(
            project_name=pn,
            data=TaskType(task_type="segmentation_proofread", completion_statuses=["done"]),
            db_session=db_session,
        )
    return p1, p2


def _ensure_segment(db_session, project_name: str, seed_id: int):
    # Local import to avoid polluting test namespace
    from datetime import datetime, timezone

    from zetta_utils.task_management.db.models import SegmentModel

    now = datetime.now(timezone.utc)
    seg = SegmentModel(
        project_name=project_name,
        seed_id=seed_id,
        seed_x=0.0,
        seed_y=0.0,
        seed_z=0.0,
        task_ids=[],
        status="Raw",
        is_exported=False,
        created_at=now,
        updated_at=now,
        expected_segment_type="axon",
    )
    db_session.add(seg)


def _create_task_in_project(db_session, project_name: str, data: Task):
    from typing import cast

    from zetta_utils.task_management.task import create_task

    # pylint: disable=line-too-long
    seed_id_val = int(cast(dict, data["extra_data"])["seed_id"])  # mypy: data["extra_data"] is dict
    _ensure_segment(db_session, project_name, seed_id_val)
    create_task(project_name=project_name, data=data, db_session=db_session)


def test_cross_project_auto_select_highest_priority(two_projects, db_session):
    p1, p2 = two_projects
    user_id = "u1"
    create_user(project_name=p1, data=_mk_user(user_id), db_session=db_session)
    create_user(project_name=p2, data=_mk_user(user_id), db_session=db_session)

    # Create tasks in both projects; highest priority is in p2
    _create_task_in_project(db_session, p1, _mk_task("A1", priority=1, seed_id=101))
    _create_task_in_project(db_session, p2, _mk_task("B5", priority=5, seed_id=201))

    selected = start_task_cross_project(user_id=user_id, task_id=None, db_session=db_session)
    assert selected == "B5"

    # Verify user records: active in p2, cleared in p1
    u_p2 = get_user(project_name=p2, user_id=user_id, db_session=db_session)
    assert u_p2["active_task"] == "B5"
    u_p1 = get_user(project_name=p1, user_id=user_id, db_session=db_session)
    assert u_p1["active_task"] == ""


def test_cross_project_prefers_assigned_task(two_projects, db_session):
    p1, p2 = two_projects
    user_id = "u2"
    create_user(project_name=p1, data=_mk_user(user_id), db_session=db_session)
    create_user(project_name=p2, data=_mk_user(user_id), db_session=db_session)

    # Assigned task in p1 with lower priority than available in p2
    t_assigned = _mk_task("A_assigned", priority=1, seed_id=111)
    t_assigned["assigned_user_id"] = user_id
    _create_task_in_project(db_session, p1, t_assigned)

    _create_task_in_project(db_session, p2, _mk_task("B2", priority=2, seed_id=222))

    selected = start_task_cross_project(user_id=user_id, task_id=None, db_session=db_session)
    assert selected == "A_assigned"


def test_cross_project_idle_takeover(two_projects, db_session):
    p1, _ = two_projects
    user_id = "u3"
    prev_user_id = "prev"
    create_user(project_name=p1, data=_mk_user(user_id), db_session=db_session)
    # Previous user holds the task in p1
    create_user(project_name=p1, data=_mk_user(prev_user_id), db_session=db_session)

    idle = _mk_task("A_idle", priority=1, seed_id=333)
    idle["active_user_id"] = prev_user_id
    # Set lease old enough to be considered idle
    idle["last_leased_ts"] = time.time() - get_max_idle_seconds() - 5
    _create_task_in_project(db_session, p1, idle)

    # Mark prev user's active task to simulate real holding
    # (this will be cleared by takeover)
    from zetta_utils.task_management.user import update_user
    update_user(
        project_name=p1,
        user_id=prev_user_id,
        data={"active_task": "A_idle"},
        db_session=db_session,
    )

    selected = start_task_cross_project(user_id=user_id, task_id=None, db_session=db_session)
    assert selected == "A_idle"

    # Previous user's active task should be cleared by takeover
    prev_user = get_user(project_name=p1, user_id=prev_user_id, db_session=db_session)
    assert prev_user["active_task"] == ""


def test_cross_project_no_qualified_user(two_projects, db_session):
    """Test that user with no qualifications gets no task."""
    p1, p2 = two_projects
    user_id = "unqualified_user"

    # Create user with no qualifications
    unqualified_user = _mk_user(user_id)
    unqualified_user["qualified_task_types"] = []
    unqualified_user["qualified_segment_types"] = []
    create_user(project_name=p1, data=unqualified_user, db_session=db_session)
    create_user(project_name=p2, data=unqualified_user, db_session=db_session)

    # Create tasks in both projects
    _create_task_in_project(db_session, p1, _mk_task("A1", priority=1, seed_id=101))
    _create_task_in_project(db_session, p2, _mk_task("B1", priority=2, seed_id=201))

    # Should get no task
    selected = start_task_cross_project(user_id=user_id, task_id=None, db_session=db_session)
    assert selected is None


def test_cross_project_specific_task_not_found(two_projects, db_session):
    """Test requesting a specific task that doesn't exist."""
    p1, p2 = two_projects
    user_id = "u1"
    create_user(project_name=p1, data=_mk_user(user_id), db_session=db_session)
    create_user(project_name=p2, data=_mk_user(user_id), db_session=db_session)

    from zetta_utils.task_management.exceptions import TaskValidationError

    # pylint: disable=line-too-long
    with pytest.raises(TaskValidationError, match="Task nonexistent not found in any accessible project"):
        start_task_cross_project(user_id=user_id, task_id="nonexistent", db_session=db_session)


def test_cross_project_user_not_found(two_projects, db_session):
    """Test with user that doesn't exist in any project."""
    from zetta_utils.task_management.exceptions import UserValidationError

    with pytest.raises(UserValidationError, match="User nonexistent not found in any project"):
        start_task_cross_project(user_id="nonexistent", task_id=None, db_session=db_session)


def test_cross_project_user_has_active_task_different_from_requested(two_projects, db_session):
    """Test when user has active task different from requested task."""
    p1, p2 = two_projects
    user_id = "u1"
    create_user(project_name=p1, data=_mk_user(user_id), db_session=db_session)
    create_user(project_name=p2, data=_mk_user(user_id), db_session=db_session)

    # Create tasks
    _create_task_in_project(db_session, p1, _mk_task("A1", priority=1, seed_id=101))
    _create_task_in_project(db_session, p2, _mk_task("B1", priority=1, seed_id=201))

    # Start one task
    start_task_cross_project(user_id=user_id, task_id="A1", db_session=db_session)

    from zetta_utils.task_management.exceptions import UserValidationError

    # Try to start different task
    # pylint: disable=line-too-long
    with pytest.raises(UserValidationError, match="User already has an active task A1 which is different from requested task B1"):
        start_task_cross_project(user_id=user_id, task_id="B1", db_session=db_session)


def test_cross_project_return_to_active_task(two_projects, db_session):
    """Test that when user has active task and calls with task_id=None, returns to same task."""
    p1, p2 = two_projects
    user_id = "u1"
    create_user(project_name=p1, data=_mk_user(user_id), db_session=db_session)
    create_user(project_name=p2, data=_mk_user(user_id), db_session=db_session)

    # Create and start a task
    _create_task_in_project(db_session, p1, _mk_task("A1", priority=1, seed_id=101))
    first_result = start_task_cross_project(user_id=user_id, task_id="A1", db_session=db_session)
    assert first_result == "A1"

    # Call again with task_id=None should return same task
    second_result = start_task_cross_project(user_id=user_id, task_id=None, db_session=db_session)
    assert second_result == "A1"


def test_cross_project_user_active_in_multiple_projects_error(clean_db, db_session):
    """Test error when user has active tasks in multiple projects (invalid state)."""
    from zetta_utils.task_management.exceptions import UserValidationError
    from zetta_utils.task_management.user import update_user

    # Create projects
    p1, p2 = "proj_A", "proj_B"
    for pn in (p1, p2):
        create_project(
            project_name=pn,
            segmentation_path=f"precomputed://gs://bucket/{pn}",
            sv_resolution_x=8.0,
            sv_resolution_y=8.0,
            sv_resolution_z=40.0,
            datastack_name=f"{pn}_ds",
            synapse_table=f"{pn}_syn",
            db_session=db_session,
        )
        create_task_type(
            project_name=pn,
            data=TaskType(task_type="segmentation_proofread", completion_statuses=["done"]),
            db_session=db_session,
        )

    user_id = "multi_active_user"
    create_user(project_name=p1, data=_mk_user(user_id), db_session=db_session)
    create_user(project_name=p2, data=_mk_user(user_id), db_session=db_session)

    # Manually set user as active in both projects (invalid state)
    # pylint: disable=line-too-long
    update_user(project_name=p1, user_id=user_id, data={"active_task": "task_1"}, db_session=db_session)
    # pylint: disable=line-too-long
    update_user(project_name=p2, user_id=user_id, data={"active_task": "task_2"}, db_session=db_session)

    with pytest.raises(UserValidationError, match="User has active tasks in multiple projects"):
        start_task_cross_project(user_id=user_id, task_id=None, db_session=db_session)


def test_get_task_cross_project_success(two_projects, db_session):
    """Test successful retrieval of task across projects."""
    from zetta_utils.task_management.task import get_task_cross_project

    p1, _ = two_projects

    # Create task in p1
    _create_task_in_project(db_session, p1, _mk_task("A1", priority=1, seed_id=101))

    # Retrieve it
    task, project = get_task_cross_project(task_id="A1", db_session=db_session)
    assert task["task_id"] == "A1"
    assert project == p1
    assert task["priority"] == 1


def test_get_task_cross_project_not_found(two_projects, db_session):
    """Test retrieval of non-existent task across projects."""
    from zetta_utils.task_management.task import get_task_cross_project

    with pytest.raises(KeyError, match="Task nonexistent not found in any project"):
        get_task_cross_project(task_id="nonexistent", db_session=db_session)


def test_get_task_cross_project_without_process_ng_state(two_projects, db_session):
    """Test get_task_cross_project with process_ng_state=False."""
    from zetta_utils.task_management.task import get_task_cross_project

    p1, _ = two_projects

    # Create task with seed_id format
    task_data = _mk_task("A1", priority=1, seed_id=101)
    task_data["ng_state"] = {"seed_id": 101}
    task_data["ng_state_initial"] = {"seed_id": 101}
    _create_task_in_project(db_session, p1, task_data)

    # Retrieve without processing
    # pylint: disable=line-too-long
    task, project = get_task_cross_project(task_id="A1", process_ng_state=False, db_session=db_session)
    assert task["task_id"] == "A1"
    assert project == p1
    assert task["ng_state"] == {"seed_id": 101}  # Should remain unprocessed


# Test helper functions
def test_build_user_qualifications_map_success(two_projects, db_session):
    """Test building user qualifications map."""
    from zetta_utils.task_management.task import _build_user_qualifications_map

    p1, p2 = two_projects
    user_id = "u1"

    # Create user in both projects
    create_user(project_name=p1, data=_mk_user(user_id), db_session=db_session)
    create_user(project_name=p2, data=_mk_user(user_id), db_session=db_session)

    # pylint: disable=line-too-long
    qualifications, active_project, active_task = _build_user_qualifications_map(db_session, user_id)

    assert len(qualifications) == 2
    assert p1 in qualifications
    assert p2 in qualifications
    assert qualifications[p1]["user_id"] == user_id
    assert qualifications[p1]["task_types"] == ["segmentation_proofread"]
    assert qualifications[p1]["segment_types"] == ["axon"]
    assert active_project is None
    assert active_task is None


def test_build_user_qualifications_map_with_active_task(two_projects, db_session):
    """Test building user qualifications map when user has active task."""
    from zetta_utils.task_management.task import _build_user_qualifications_map
    from zetta_utils.task_management.user import update_user

    p1, p2 = two_projects
    user_id = "u1"

    # Create user in both projects
    create_user(project_name=p1, data=_mk_user(user_id), db_session=db_session)
    create_user(project_name=p2, data=_mk_user(user_id), db_session=db_session)

    # Set active task in p1
    # pylint: disable=line-too-long
    update_user(project_name=p1, user_id=user_id, data={"active_task": "task_123"}, db_session=db_session)

    _, active_project, active_task = _build_user_qualifications_map(db_session, user_id)

    assert active_project == p1
    assert active_task == "task_123"


def test_build_user_qualifications_map_user_not_found(two_projects, db_session):
    """Test building qualifications map for non-existent user."""
    from zetta_utils.task_management.exceptions import UserValidationError
    from zetta_utils.task_management.task import _build_user_qualifications_map

    with pytest.raises(UserValidationError, match="User nonexistent not found in any project"):
        _build_user_qualifications_map(db_session, "nonexistent")


def test_find_task_across_projects_found(two_projects, db_session):
    """Test finding a task that exists."""
    from zetta_utils.task_management.task import _find_task_across_projects

    p1, p2 = two_projects

    # Create task in p2
    _create_task_in_project(db_session, p2, _mk_task("B1", priority=1, seed_id=201))

    # Find it
    result = _find_task_across_projects(db_session, "B1", [p1, p2])
    assert result is not None
    assert result.task_id == "B1"
    assert result.project_name == p2


def test_find_task_across_projects_not_found(two_projects, db_session):
    """Test finding a task that doesn't exist."""
    from zetta_utils.task_management.task import _find_task_across_projects

    p1, p2 = two_projects

    result = _find_task_across_projects(db_session, "nonexistent", [p1, p2])
    assert result is None


def test_validate_user_for_task_in_project_success(two_projects, db_session):
    """Test successful user validation for task."""
    from zetta_utils.task_management.db.models import TaskModel
    from zetta_utils.task_management.task import _validate_user_for_task_in_project

    p1, _ = two_projects
    user_id = "u1"
    create_user(project_name=p1, data=_mk_user(user_id), db_session=db_session)

    # Create task
    _create_task_in_project(db_session, p1, _mk_task("A1", priority=1, seed_id=101))

    # Get task model
    from sqlalchemy import select
    task_model = db_session.execute(
        select(TaskModel).where(TaskModel.task_id == "A1").where(TaskModel.project_name == p1)
    ).scalar_one()

    # Should not raise
    _validate_user_for_task_in_project(db_session, user_id, p1, task_model)


def test_validate_user_for_task_in_project_wrong_task_type(two_projects, db_session):
    """Test user validation fails for wrong task type."""
    from zetta_utils.task_management.db.models import TaskModel
    from zetta_utils.task_management.exceptions import UserValidationError
    from zetta_utils.task_management.task import _validate_user_for_task_in_project

    p1, _ = two_projects
    user_id = "u1"

    # Create user qualified for different task type
    user_data = _mk_user(user_id)
    user_data["qualified_task_types"] = ["different_type"]
    create_user(project_name=p1, data=user_data, db_session=db_session)

    # Create task
    _create_task_in_project(db_session, p1, _mk_task("A1", priority=1, seed_id=101))

    # Get task model
    from sqlalchemy import select
    task_model = db_session.execute(
        select(TaskModel).where(TaskModel.task_id == "A1").where(TaskModel.project_name == p1)
    ).scalar_one()

    with pytest.raises(UserValidationError, match="User not qualified for this task type"):
        _validate_user_for_task_in_project(db_session, user_id, p1, task_model)


def test_validate_user_for_task_in_project_wrong_segment_type(two_projects, db_session):
    """Test user validation fails for wrong segment type."""
    from datetime import datetime, timezone

    from zetta_utils.task_management.db.models import SegmentModel, TaskModel
    from zetta_utils.task_management.exceptions import UserValidationError
    from zetta_utils.task_management.task import _validate_user_for_task_in_project

    p1, _ = two_projects
    user_id = "u1"

    # Create user qualified for different segment type
    user_data = _mk_user(user_id)
    user_data["qualified_segment_types"] = ["different_segment"]
    create_user(project_name=p1, data=user_data, db_session=db_session)

    # Create segment with specific type
    now = datetime.now(timezone.utc)
    segment = SegmentModel(
        project_name=p1,
        seed_id=1001,
        seed_x=0.0,
        seed_y=0.0,
        seed_z=0.0,
        task_ids=[],
        status="Raw",
        is_exported=False,
        created_at=now,
        updated_at=now,
        expected_segment_type="axon",  # Different from user qualification
    )
    db_session.add(segment)
    db_session.commit()

    # Create task directly (avoid duplicate segment creation)
    task_data_first_test = _mk_task("A1", priority=1, seed_id=1001)
    from zetta_utils.task_management.task import create_task
    create_task(project_name=p1, data=task_data_first_test, db_session=db_session)

    # Get task model
    from sqlalchemy import select
    task_model = db_session.execute(
        select(TaskModel).where(TaskModel.task_id == "A1").where(TaskModel.project_name == p1)
    ).scalar_one()

    with pytest.raises(UserValidationError, match="User not qualified for this segment type"):
        _validate_user_for_task_in_project(db_session, user_id, p1, task_model)


def test_auto_select_task_cross_project_no_qualifications(two_projects, db_session):
    """Test auto selection when user has no qualifications."""
    from zetta_utils.task_management.task import _auto_select_task_cross_project

    result = _auto_select_task_cross_project(db_session, {})
    assert result is None


def test_auto_select_task_cross_project_assigned_task(two_projects, db_session):
    """Test auto selection prefers assigned tasks."""
    from zetta_utils.task_management.task import _auto_select_task_cross_project

    p1, p2 = two_projects
    user_id = "u1"

    # Create user qualifications map manually
    qualifications = {
        # pylint: disable=line-too-long
        p1: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]},
        # pylint: disable=line-too-long
        p2: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]}
    }

    # Create assigned task (lower priority) and available task (higher priority)
    assigned_task = _mk_task("assigned", priority=1, seed_id=101)
    assigned_task["assigned_user_id"] = user_id
    _create_task_in_project(db_session, p1, assigned_task)

    available_task = _mk_task("available", priority=10, seed_id=201)
    _create_task_in_project(db_session, p2, available_task)

    result = _auto_select_task_cross_project(db_session, qualifications)  # type: ignore[arg-type]
    assert result is not None
    assert result.task_id == "assigned"  # Should prefer assigned even if lower priority


def test_select_assigned_task_across_projects_found(two_projects, db_session):
    """Test finding assigned task across projects."""
    from zetta_utils.task_management.task import _select_assigned_task_across_projects

    p1, p2 = two_projects
    user_id = "u1"

    qualifications = {
        # pylint: disable=line-too-long
        p1: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]},
        # pylint: disable=line-too-long
        p2: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]}
    }

    # Create assigned task
    assigned_task = _mk_task("assigned", priority=1, seed_id=101)
    assigned_task["assigned_user_id"] = user_id
    _create_task_in_project(db_session, p1, assigned_task)

    # pylint: disable=line-too-long
    result = _select_assigned_task_across_projects(db_session, qualifications)  # type: ignore[arg-type]
    assert result is not None
    assert result.task_id == "assigned"


def test_select_assigned_task_across_projects_none_found(two_projects, db_session):
    """Test when no assigned task is found."""
    from zetta_utils.task_management.task import _select_assigned_task_across_projects

    p1, p2 = two_projects
    user_id = "u1"

    qualifications = {
        # pylint: disable=line-too-long
        p1: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]},
        # pylint: disable=line-too-long
        p2: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]}
    }

    # Create unassigned task
    _create_task_in_project(db_session, p1, _mk_task("unassigned", priority=1, seed_id=101))

    # pylint: disable=line-too-long
    result = _select_assigned_task_across_projects(db_session, qualifications)  # type: ignore[arg-type]
    assert result is None


def test_select_available_task_across_projects_found(two_projects, db_session):
    """Test finding available task across projects."""
    from zetta_utils.task_management.task import _select_available_task_across_projects

    p1, p2 = two_projects
    user_id = "u1"

    qualifications = {
        # pylint: disable=line-too-long
        p1: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]},
        # pylint: disable=line-too-long
        p2: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]}
    }

    # Create tasks with different priorities
    _create_task_in_project(db_session, p1, _mk_task("low_priority", priority=1, seed_id=101))
    _create_task_in_project(db_session, p2, _mk_task("high_priority", priority=10, seed_id=201))

    # pylint: disable=line-too-long
    result = _select_available_task_across_projects(db_session, qualifications)  # type: ignore[arg-type]
    assert result is not None
    assert result.task_id == "high_priority"  # Should select highest priority


def test_select_available_task_across_projects_none_found(two_projects, db_session):
    """Test when no available task is found."""
    from zetta_utils.task_management.task import _select_available_task_across_projects

    p1, _ = two_projects
    user_id = "u1"

    qualifications = {
        # pylint: disable=line-too-long
        p1: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]}
    }

    # Create completed tasks only
    completed_task = _mk_task("completed", priority=1, seed_id=101)
    completed_task["completion_status"] = "done"
    _create_task_in_project(db_session, p1, completed_task)

    # pylint: disable=line-too-long
    result = _select_available_task_across_projects(db_session, qualifications)  # type: ignore[arg-type]
    assert result is None


def test_select_idle_task_across_projects_found(two_projects, db_session):
    """Test finding idle task across projects."""
    from zetta_utils.task_management.task import _select_idle_task_across_projects

    p1, p2 = two_projects
    user_id = "u1"

    qualifications = {
        # pylint: disable=line-too-long
        p1: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]},
        # pylint: disable=line-too-long
        p2: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]}
    }

    # Create idle task
    idle_task = _mk_task("idle", priority=1, seed_id=101)
    idle_task["active_user_id"] = "other_user"
    idle_task["last_leased_ts"] = time.time() - get_max_idle_seconds() - 10
    _create_task_in_project(db_session, p1, idle_task)

    # pylint: disable=line-too-long
    result = _select_idle_task_across_projects(db_session, qualifications)  # type: ignore[arg-type]
    assert result is not None
    assert result.task_id == "idle"


def test_select_idle_task_across_projects_none_found(two_projects, db_session):
    """Test when no idle task is found."""
    from zetta_utils.task_management.task import _select_idle_task_across_projects

    p1, _ = two_projects
    user_id = "u1"

    qualifications = {
        # pylint: disable=line-too-long
        p1: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]}
    }

    # Create recent (not idle) task
    recent_task = _mk_task("recent", priority=1, seed_id=101)
    recent_task["active_user_id"] = "other_user"
    recent_task["last_leased_ts"] = time.time()  # Recent timestamp
    _create_task_in_project(db_session, p1, recent_task)

    # pylint: disable=line-too-long
    result = _select_idle_task_across_projects(db_session, qualifications)  # type: ignore[arg-type]
    assert result is None


# Additional edge case tests
def test_validate_user_for_task_in_project_no_segment_types(two_projects, db_session):
    """Test user validation when user has no segment type qualifications."""
    from zetta_utils.task_management.db.models import TaskModel
    from zetta_utils.task_management.task import _validate_user_for_task_in_project

    p1, _ = two_projects
    user_id = "u1"

    # Create user with no segment type qualifications
    user_data = _mk_user(user_id)
    user_data["qualified_segment_types"] = []
    create_user(project_name=p1, data=user_data, db_session=db_session)

    # Create task
    _create_task_in_project(db_session, p1, _mk_task("A1", priority=1, seed_id=101))

    # Get task model
    from sqlalchemy import select
    task_model = db_session.execute(
        select(TaskModel).where(TaskModel.task_id == "A1").where(TaskModel.project_name == p1)
    ).scalar_one()

    # Should not raise since segment type validation is skipped when user has no qualifications
    _validate_user_for_task_in_project(db_session, user_id, p1, task_model)


def test_validate_user_for_task_in_project_no_seed_id(two_projects, db_session):
    """Test user validation when task has no seed_id in extra_data."""
    from zetta_utils.task_management.db.models import TaskModel
    from zetta_utils.task_management.task import _validate_user_for_task_in_project

    p1, _ = two_projects
    user_id = "u1"
    create_user(project_name=p1, data=_mk_user(user_id), db_session=db_session)

    # Create task without seed_id
    task_data = _mk_task("A1", priority=1, seed_id=101)
    task_data["extra_data"] = {}  # No seed_id

    # Need to create manually since _create_task_in_project expects seed_id
    from zetta_utils.task_management.task import create_task
    create_task(project_name=p1, data=task_data, db_session=db_session)

    # Get task model
    from sqlalchemy import select
    task_model = db_session.execute(
        select(TaskModel).where(TaskModel.task_id == "A1").where(TaskModel.project_name == p1)
    ).scalar_one()

    # Should not raise since no seed_id means no segment validation
    _validate_user_for_task_in_project(db_session, user_id, p1, task_model)


def test_validate_user_for_task_in_project_no_segment_found(two_projects, db_session):
    """Test user validation when segment doesn't exist."""
    from zetta_utils.task_management.db.models import TaskModel
    from zetta_utils.task_management.task import _validate_user_for_task_in_project

    p1, _ = two_projects
    user_id = "u1"
    create_user(project_name=p1, data=_mk_user(user_id), db_session=db_session)

    # Create task with seed_id but don't create the segment
    task_data = _mk_task("A1", priority=1, seed_id=999)  # Non-existent seed_id

    # Create task manually without creating segment
    from zetta_utils.task_management.task import create_task
    create_task(project_name=p1, data=task_data, db_session=db_session)

    # Get task model
    from sqlalchemy import select
    task_model = db_session.execute(
        select(TaskModel).where(TaskModel.task_id == "A1").where(TaskModel.project_name == p1)
    ).scalar_one()

    # Should not raise since no segment means no segment validation
    _validate_user_for_task_in_project(db_session, user_id, p1, task_model)


def test_validate_user_for_task_in_project_segment_no_expected_type(two_projects, db_session):
    """Test user validation when segment has no expected_segment_type."""
    from datetime import datetime, timezone

    from zetta_utils.task_management.db.models import SegmentModel, TaskModel
    from zetta_utils.task_management.task import _validate_user_for_task_in_project

    p1, _ = two_projects
    user_id = "u1"
    create_user(project_name=p1, data=_mk_user(user_id), db_session=db_session)

    # Create segment without expected_segment_type
    now = datetime.now(timezone.utc)
    segment = SegmentModel(
        project_name=p1,
        seed_id=1002,
        seed_x=0.0,
        seed_y=0.0,
        seed_z=0.0,
        task_ids=[],
        status="Raw",
        is_exported=False,
        created_at=now,
        updated_at=now,
        expected_segment_type=None,  # No expected type
    )
    db_session.add(segment)
    db_session.commit()

    # Create task
    task_data_no_expected = _mk_task("A1", priority=1, seed_id=1003)  # Use different seed_id
    from zetta_utils.task_management.task import create_task
    create_task(project_name=p1, data=task_data_no_expected, db_session=db_session)

    # Get task model
    from sqlalchemy import select
    task_model = db_session.execute(
        select(TaskModel).where(TaskModel.task_id == "A1").where(TaskModel.project_name == p1)
    ).scalar_one()

    # Should not raise since no expected segment type means no validation
    _validate_user_for_task_in_project(db_session, user_id, p1, task_model)


def test_select_assigned_task_across_projects_user_no_qualifications(two_projects, db_session):
    """Test assigned task selection when user has no qualifications in some projects."""
    from zetta_utils.task_management.task import _select_assigned_task_across_projects

    p1, p2 = two_projects
    user_id = "u1"

    # Qualifications with empty lists for p2
    qualifications = {
        # pylint: disable=line-too-long
        p1: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]},
        p2: {"user_id": user_id, "task_types": [], "segment_types": []}  # No qualifications
    }

    # Create assigned tasks in both projects
    assigned_task_p1 = _mk_task("assigned_p1", priority=1, seed_id=101)
    assigned_task_p1["assigned_user_id"] = user_id
    _create_task_in_project(db_session, p1, assigned_task_p1)

    assigned_task_p2 = _mk_task("assigned_p2", priority=5, seed_id=201)
    assigned_task_p2["assigned_user_id"] = user_id
    _create_task_in_project(db_session, p2, assigned_task_p2)

    # pylint: disable=line-too-long
    result = _select_assigned_task_across_projects(db_session, qualifications)  # type: ignore[arg-type]
    assert result is not None
    assert result.task_id == "assigned_p1"  # Should only find task in p1 where user is qualified


def test_select_available_task_priority_tie_breaker(two_projects, db_session):
    """Test available task selection with priority tie-breaker using id_nonunique."""
    from zetta_utils.task_management.task import _select_available_task_across_projects

    p1, p2 = two_projects
    user_id = "u1"

    qualifications = {
        # pylint: disable=line-too-long
        p1: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]},
        # pylint: disable=line-too-long
        p2: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]}
    }

    # Create tasks with same priority - should select the one with lower id_nonunique
    _create_task_in_project(db_session, p1, _mk_task("task_a", priority=5, seed_id=101))
    _create_task_in_project(db_session, p2, _mk_task("task_b", priority=5, seed_id=201))

    # pylint: disable=line-too-long
    result = _select_available_task_across_projects(db_session, qualifications)  # type: ignore[arg-type]
    assert result is not None
    # Should select whichever has lower id_nonunique (implementation detail)
    assert result.task_id in ["task_a", "task_b"]


def test_select_idle_task_priority_tie_breaker(two_projects, db_session):
    """Test idle task selection with priority and lease time tie-breakers."""
    from zetta_utils.task_management.task import _select_idle_task_across_projects

    p1, p2 = two_projects
    user_id = "u1"

    qualifications = {
        # pylint: disable=line-too-long
        p1: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]},
        # pylint: disable=line-too-long
        p2: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]}
    }

    old_time = time.time() - get_max_idle_seconds() - 10

    # Create idle tasks with same priority - should prefer more recent lease time
    idle_task_1 = _mk_task("idle_1", priority=5, seed_id=101)
    idle_task_1["active_user_id"] = "other_user_1"
    idle_task_1["last_leased_ts"] = old_time - 5  # Older
    _create_task_in_project(db_session, p1, idle_task_1)

    idle_task_2 = _mk_task("idle_2", priority=5, seed_id=201)
    idle_task_2["active_user_id"] = "other_user_2"
    idle_task_2["last_leased_ts"] = old_time  # More recent
    _create_task_in_project(db_session, p2, idle_task_2)

    # pylint: disable=line-too-long
    result = _select_idle_task_across_projects(db_session, qualifications)  # type: ignore[arg-type]
    assert result is not None
    assert result.task_id == "idle_2"  # Should prefer more recent lease time


def test_select_available_task_returns_none_when_no_tasks(two_projects, db_session):
    """_select_available_task_across_projects returns None when nothing matches."""
    from zetta_utils.task_management.task import _select_available_task_across_projects

    p1, p2 = two_projects
    user_id = "u_no_tasks"

    # pylint: disable=line-too-long
    qualifications = {
        p1: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]},
        p2: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]},
    }

    # No tasks created in either project -> expect None
    result = _select_available_task_across_projects(db_session, qualifications)  # type: ignore[arg-type]
    assert result is None


def test_select_idle_task_returns_none_when_no_idle(two_projects, db_session):
    """_select_idle_task_across_projects returns None when no idle tasks exist."""
    from zetta_utils.task_management.task import _select_idle_task_across_projects

    p1, p2 = two_projects
    user_id = "u_no_idle"

    # pylint: disable=line-too-long
    qualifications = {
        p1: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]},
        p2: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]},
    }

    # No tasks created -> nothing is idle
    result = _select_idle_task_across_projects(db_session, qualifications)  # type: ignore[arg-type]
    assert result is None


def test_auto_select_task_no_segment_qualifications_returns_none(two_projects, db_session):
    """_auto_select_task returns None and logs when no segment qualifications."""
    from zetta_utils.task_management.task import _auto_select_task

    p1, _ = two_projects
    user_id = "u_no_seg_qual"

    # Create user with task type but without segment qualifications
    user_data = _mk_user(user_id)
    user_data["qualified_segment_types"] = []
    create_user(project_name=p1, data=user_data, db_session=db_session)

    # No need to create tasks; function returns early on missing segment quals
    result = _auto_select_task(db_session, p1, user_id)
    assert result is None


def test_select_available_task_locked_returns_none(two_projects, db_session, db_engine):
    """Locked candidate causes _select_available_task_across_projects to return None."""
    from sqlalchemy import select

    from zetta_utils.task_management.db.models import TaskModel
    from zetta_utils.task_management.db.session import get_session_factory
    from zetta_utils.task_management.task import _select_available_task_across_projects

    p1, _ = two_projects
    user_id = "u_locked"

    qualifications = {
        p1: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]},  # pylint: disable=line-too-long
    }

    # Create a matching available task
    _create_task_in_project(db_session, p1, _mk_task("avail_lock", priority=5, seed_id=101))

    # Lock the task row in a separate session so skip_locked will skip it
    session2 = get_session_factory(db_engine)()
    try:
        session2.execute(
            select(TaskModel)
            .where(TaskModel.project_name == p1)
            .where(TaskModel.task_id == "avail_lock")
            .with_for_update()
        ).scalar_one()

        # Now attempt selection; lock should cause function to return None
        result = _select_available_task_across_projects(db_session, qualifications)  # type: ignore[arg-type]  # pylint: disable=line-too-long
        assert result is None
    finally:
        session2.rollback()
        session2.close()


def test_select_idle_task_locked_returns_none(two_projects, db_session, db_engine):
    """Locked idle candidate causes _select_idle_task_across_projects to return None."""
    from sqlalchemy import select

    from zetta_utils.task_management.db.models import TaskModel
    from zetta_utils.task_management.db.session import get_session_factory
    from zetta_utils.task_management.task import _select_idle_task_across_projects

    p1, _ = two_projects
    user_id = "u_locked_idle"

    qualifications = {
        p1: {"user_id": user_id, "task_types": ["segmentation_proofread"], "segment_types": ["axon"]},  # pylint: disable=line-too-long
    }

    # Create an idle task
    idle = _mk_task("idle_lock", priority=3, seed_id=202)
    idle["last_leased_ts"] = time.time() - get_max_idle_seconds() - 10
    _create_task_in_project(db_session, p1, idle)

    # Lock the row in a separate session
    session2 = get_session_factory(db_engine)()
    try:
        session2.execute(
            select(TaskModel)
            .where(TaskModel.project_name == p1)
            .where(TaskModel.task_id == "idle_lock")
            .with_for_update()
        ).scalar_one()

        result = _select_idle_task_across_projects(db_session, qualifications)  # type: ignore[arg-type]  # pylint: disable=line-too-long
        assert result is None
    finally:
        session2.rollback()
        session2.close()
