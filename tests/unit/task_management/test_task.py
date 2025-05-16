# pylint: disable=redefined-outer-name,unused-argument
import pytest

from zetta_utils.task_management.subtask import (
    create_subtask,
    release_subtask,
    start_subtask,
    update_subtask,
)
from zetta_utils.task_management.task import create_task, get_task, update_task
from zetta_utils.task_management.types import Subtask, Task, TaskUpdate


@pytest.fixture
def sample_subtasks() -> list[Subtask]:
    return [
        {
            "task_id": "task_1",
            "subtask_id": f"subtask_{i}",
            "completion_status": "",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": f"http://example.com/{i}",
            "ng_state_initial": f"http://example.com/{i}",
            "priority": i,
            "batch_id": "batch_1",
            "last_leased_ts": 0.0,
            "is_active": True,
            "subtask_type": "segmentation_proofread",
        }
        for i in range(1, 4)
    ]


@pytest.fixture
def existing_subtasks(firestore_emulator, project_name, sample_subtasks, existing_subtask_type):
    for subtask in sample_subtasks:
        create_subtask(project_name, subtask)
    yield sample_subtasks


def test_create_task_success(project_name, sample_task):
    result = create_task(project_name, sample_task)
    assert result == sample_task["task_id"]

    task = get_task(project_name, "task_1")
    assert task == sample_task


def test_get_task_success(existing_task, project_name):
    result = get_task(project_name, "task_1")
    assert result == existing_task


def test_get_task_not_found(project_name):
    with pytest.raises(KeyError, match="Task task_1 not found"):
        get_task(project_name, "task_1")


def test_update_task_success(existing_task, project_name):
    update_data = TaskUpdate(**{"status": "ingested"})

    result = update_task(project_name, "task_1", update_data)
    assert result is True

    task = get_task(project_name, "task_1")
    assert task["status"] == "ingested"


def test_task_completion_when_all_subtasks_done(
    existing_task, existing_subtasks, project_name, existing_user
):
    update_task(project_name, "task_1", {"status": "ingested"})

    for subtask in existing_subtasks:
        start_subtask(project_name, "test_user", subtask["subtask_id"])
        release_subtask(project_name, "test_user", subtask["subtask_id"], "done")

    task = get_task(project_name, "task_1")
    assert task["status"] == "fully_processed"


def test_task_not_complete_with_pending_subtasks(
    existing_task, existing_subtasks, project_name, existing_user
):
    update_task(project_name, "task_1", {"status": "ingested"})

    for i, subtask in enumerate(existing_subtasks):
        if i < len(existing_subtasks) - 1:
            update_subtask(
                project_name,
                subtask["subtask_id"],
                {"completion_status": "done", "completed_user_id": "test_user"},
            )

    task = get_task(project_name, "task_1")
    assert task["status"] == "ingested"


def test_create_task_validation(project_name):
    invalid_task = {"task_id": "task_1"}
    with pytest.raises(Exception):
        create_task(project_name, invalid_task)  # type: ignore


def test_update_task_invalid_status(existing_task, project_name):
    with pytest.raises(ValueError, match="Invalid status value"):
        update_task(project_name, "task_1", TaskUpdate(**{"status": "invalid_status"}))


def test_create_task_duplicate(project_name):
    task_data = Task(
        **{
            "task_id": "duplicate_task",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "task_type": "segmentation",
            "ng_state": "http://example.com/duplicate_task",
        }
    )

    result = create_task(project_name, task_data)
    assert result == "duplicate_task"

    with pytest.raises(ValueError, match="Task duplicate_task already exists"):
        create_task(project_name, task_data)

    task = get_task(project_name, "duplicate_task")
    assert task["batch_id"] == "batch_1"
    assert task["status"] == "pending_ingestion"
    assert task["ng_state"] == "http://example.com/duplicate_task"


def test_update_task_not_found(project_name):
    update_data = TaskUpdate(**{"status": "ingested"})

    with pytest.raises(KeyError, match="Task non_existent_task not found"):
        update_task(project_name, "non_existent_task", update_data)

    with pytest.raises(KeyError, match="Task non_existent_task not found"):
        get_task(project_name, "non_existent_task")


def test_create_task(project_name):
    task_data = Task(
        **{
            "task_id": "task_1",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "task_type": "segmentation",
            "ng_state": "http://example.com/task_1",
        }
    )

    result = create_task(project_name, task_data)
    assert result == "task_1"

    task = get_task(project_name, "task_1")
    assert task["batch_id"] == "batch_1"
    assert task["status"] == "pending_ingestion"
    assert task["ng_state"] == "http://example.com/task_1"


def test_update_task(project_name):
    task_data = Task(
        **{
            "task_id": "task_2",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "task_type": "segmentation",
            "ng_state": "http://example.com/task_2",
        }
    )
    create_task(project_name, task_data)

    update_data = TaskUpdate(
        **{
            "status": "ingested",
            "ng_state": "http://example.com/task_2_updated",
        }
    )
    update_task(project_name, "task_2", update_data)

    task = get_task(project_name, "task_2")
    assert task["status"] == "ingested"
    assert task["ng_state"] == "http://example.com/task_2_updated"
