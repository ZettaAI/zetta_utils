"""Unit tests for get_task_feedback_cross_project with 100% line coverage."""

# pylint: disable=redefined-outer-name,too-many-locals,unused-argument

from datetime import datetime, timedelta, timezone

import pytest

from zetta_utils.task_management import task as task_mod
from zetta_utils.task_management.db.models import TaskFeedbackModel
from zetta_utils.task_management.project import create_project
from zetta_utils.task_management.task import (
    create_task,
    get_task_feedback_cross_project,
)
from zetta_utils.task_management.task_type import create_task_type
from zetta_utils.task_management.types import Task
from zetta_utils.task_management.user import create_user


@pytest.fixture
def two_projects(clean_db, db_session):
    """Create two projects available for tests."""
    create_project(
        project_name="projA",
        segmentation_path="gs://test/segmentation/projA",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )
    create_project(
        project_name="projB",
        segmentation_path="gs://test/segmentation/projB",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )
    return ("projA", "projB")


@pytest.fixture
def task_types(db_session, two_projects):
    """Create trace and feedback task types in both projects."""
    for project in two_projects:
        create_task_type(
            project_name=project,
            data={
                "task_type": "trace_v0",
                "completion_statuses": ["done", "need_help"],
            },
            db_session=db_session,
        )
        create_task_type(
            project_name=project,
            data={
                "task_type": "trace_feedback_v0",
                "completion_statuses": [
                    "Accurate",
                    "Fair",
                    "Inaccurate",
                    "Faulty Task",
                ],
            },
            db_session=db_session,
        )


@pytest.fixture
def qualified_user(db_session, two_projects):
    """Create a user with the same user_id in both projects."""
    for project in two_projects:
        create_user(
            project_name=project,
            data={
                "user_id": "u1@zetta.ai",
                "hourly_rate": 25.0,
                "active_task": "",
                "qualified_task_types": ["trace_v0", "trace_feedback_v0"],
                "qualified_segment_types": [],
            },
            db_session=db_session,
        )
    return "u1@zetta.ai"


def _mk_trace_task(task_id: str) -> Task:
    return {
        "task_id": task_id,
        "completion_status": "done",
        "assigned_user_id": "",
        "active_user_id": "",
        "completed_user_id": "u1@zetta.ai",
        "ng_state": {"url": f"http://example.com/{task_id}"},
        "ng_state_initial": {"url": f"http://example.com/{task_id}"},
        "priority": 1,
        "batch_id": "batch_1",
        "task_type": "trace_v0",
        "is_active": True,
        "last_leased_ts": 0.0,
    }


def _mk_feedback_task(task_id: str, status: str, note: str | None = None) -> Task:
    return {
        "task_id": task_id,
        "completion_status": status,
        "assigned_user_id": "",
        "active_user_id": "",
        "completed_user_id": "u1@zetta.ai",
        "ng_state": {"url": f"http://example.com/{task_id}"},
        "ng_state_initial": {"url": f"http://example.com/{task_id}"},
        "priority": 1,
        "batch_id": "batch_1",
        "task_type": "trace_feedback_v0",
        "is_active": True,
        "last_leased_ts": 0.0,
        "note": note,
    }


def test_get_task_feedback_cross_project_success(
    clean_db, db_session, two_projects, task_types, qualified_user
):
    projA, projB = two_projects
    # task_types fixture ensures required task types exist in both projects

    # Create original and feedback tasks in both projects
    # Project A
    create_task(project_name=projA, data=_mk_trace_task("orig1"), db_session=db_session)
    create_task(
        project_name=projA,
        data=_mk_feedback_task("fb1", status="Accurate", note="n1"),
        db_session=db_session,
    )

    create_task(project_name=projA, data=_mk_trace_task("orig2"), db_session=db_session)
    create_task(
        project_name=projA,
        data=_mk_feedback_task("fb2", status="Fair", note="n2"),
        db_session=db_session,
    )

    # Project B
    create_task(project_name=projB, data=_mk_trace_task("orig3"), db_session=db_session)
    create_task(
        project_name=projB,
        data=_mk_feedback_task("fb3", status="Inaccurate", note="n3"),
        db_session=db_session,
    )

    # Insert feedback links (including one with missing tasks to trigger "Unknown")
    now = datetime.now(timezone.utc)
    fb_items = [
        # Newest
        (projA, "orig1", "fb1", qualified_user, now),
        # Middle
        (projB, "orig3", "fb3", qualified_user, now - timedelta(minutes=1)),
        # Oldest
        (projA, "orig2", "fb2", qualified_user, now - timedelta(minutes=2)),
        # Missing tasks -> Unknown
        (projB, "missing_orig", "missing_fb", qualified_user, now - timedelta(minutes=3)),
    ]

    for i, (project_name, t_id, fb_id, user_id, ts) in enumerate(fb_items, start=1):
        db_session.add(
            TaskFeedbackModel(
                project_name=project_name,
                feedback_id=i,  # explicit for deterministic assertions
                task_id=t_id,
                feedback_task_id=fb_id,
                user_id=user_id,
                created_at=ts,
            )
        )
    db_session.commit()

    # Call function under test
    result = get_task_feedback_cross_project(
        user_id=qualified_user, limit=10, db_session=db_session
    )

    # Expect 4 items sorted by created_at desc
    assert [r["feedback_id"] for r in result] == [1, 2, 3, 4]

    # Item 0: Accurate -> green
    r0 = result[0]
    assert r0["project_name"] == projA
    assert r0["task_id"] == "orig1"
    assert r0["feedback_task_id"] == "fb1"
    assert r0["feedback"] == "Accurate"
    assert r0["feedback_color"] == "green"
    assert isinstance(r0["task_link"], dict)
    assert isinstance(r0["feedback_link"], dict)
    assert r0["note"] == "n1"
    assert isinstance(r0["created_at"], str) and r0["created_at"].endswith("+00:00")

    # Item 1: Inaccurate -> red
    r1 = result[1]
    assert r1["project_name"] == projB
    assert r1["feedback"] == "Inaccurate"
    assert r1["feedback_color"] == "red"

    # Item 2: Fair -> yellow
    r2 = result[2]
    assert r2["project_name"] == projA
    assert r2["feedback"] == "Fair"
    assert r2["feedback_color"] == "yellow"

    # Item 3: Unknown (missing tasks) -> default red and None links
    r3 = result[3]
    assert r3["project_name"] == projB
    assert r3["feedback"] == "Unknown"
    assert r3["feedback_color"] == "red"
    assert r3["task_link"] is None
    assert r3["feedback_link"] is None
    assert r3["note"] is None


def test_get_task_feedback_cross_project_with_skip(
    clean_db, db_session, two_projects, task_types, qualified_user
):
    projA, projB = two_projects

    # Create original and feedback tasks
    create_task(project_name=projA, data=_mk_trace_task("orig1"), db_session=db_session)
    create_task(project_name=projA, data=_mk_feedback_task("fb1", status="Accurate"), db_session=db_session) # pylint: disable=line-too-long

    create_task(project_name=projB, data=_mk_trace_task("orig3"), db_session=db_session)
    create_task(project_name=projB, data=_mk_feedback_task("fb3", status="Inaccurate"), db_session=db_session) # pylint: disable=line-too-long

    create_task(project_name=projA, data=_mk_trace_task("orig2"), db_session=db_session)
    create_task(project_name=projA, data=_mk_feedback_task("fb2", status="Fair"), db_session=db_session) # pylint: disable=line-too-long

    # Feedback links with timestamps to define order
    now = datetime.now(timezone.utc)
    fb_rows = [
        (projA, "orig1", "fb1", qualified_user, now),
        (projB, "orig3", "fb3", qualified_user, now - timedelta(minutes=1)),
        (projA, "orig2", "fb2", qualified_user, now - timedelta(minutes=2)),
        (projB, "missing_orig", "missing_fb", qualified_user, now - timedelta(minutes=3)),
    ]
    for i, (pn, t_id, fb_id, uid, ts) in enumerate(fb_rows, start=1):
        db_session.add(
            TaskFeedbackModel(
                project_name=pn,
                feedback_id=i,
                task_id=t_id,
                feedback_task_id=fb_id,
                user_id=uid,
                created_at=ts,
            )
        )
    db_session.commit()

    # Skip the first (newest) item
    result = get_task_feedback_cross_project(
        user_id=qualified_user, limit=10, skip=1, db_session=db_session
    )
    assert [r["feedback_id"] for r in result] == [2, 3, 4]


def test_get_task_feedback_cross_project_user_not_found(monkeypatch, db_session):
    """Ensure the explicit raise inside the function is covered."""

    # Patch qualifications builder to return empty mapping so the function's
    # own UserValidationError branch executes (distinct from builder's raise).

    def fake_build_user_qualifications_map(_session, _user_id):  # noqa: ARG001
        return {}, None, None

    monkeypatch.setattr(
        task_mod, "_build_user_qualifications_map", fake_build_user_qualifications_map
    )

    with pytest.raises(task_mod.UserValidationError, match="User test_user not found"):
        get_task_feedback_cross_project(
            user_id="test_user", limit=5, db_session=db_session
        )
