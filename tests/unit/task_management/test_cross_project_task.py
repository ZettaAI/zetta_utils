"""Tests for cross-project task selection and API endpoint."""

# pylint: disable=unused-argument,redefined-outer-name,import-outside-toplevel,import-error

import time

import pytest

from zetta_utils.task_management.project import create_project
from zetta_utils.task_management.task import (get_max_idle_seconds,
                                              start_task_cross_project)
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

    seed_id_val = int(cast(dict, data["extra_data"]) ["seed_id"])  # mypy: data["extra_data"] is dict
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


@pytest.mark.asyncio
async def test_start_task_cross_project_api_success(mocker):
    from web_api.app import tasks as tasks_module
    from web_api.app.tasks import start_task_cross_project_api

    mocker.patch.object(tasks_module, "start_task_cross_project", return_value="T123")
    result = await start_task_cross_project_api(user_id="userX", task_id=None)
    assert result == "T123"


@pytest.mark.asyncio
async def test_start_task_cross_project_api_exception(mocker):
    from fastapi import HTTPException
    from web_api.app import tasks as tasks_module
    from web_api.app.tasks import start_task_cross_project_api

    mocker.patch.object(tasks_module, "start_task_cross_project", side_effect=Exception("boom"))
    with pytest.raises(HTTPException) as exc:
        await start_task_cross_project_api(user_id="userY", task_id=None)
    assert exc.value.status_code == 409
    assert "Failed to start task" in str(exc.value.detail)
