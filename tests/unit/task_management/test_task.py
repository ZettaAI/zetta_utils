# pylint: disable=redefined-outer-name,unused-argument
import pytest

from zetta_utils.task_management.task import (
    create_task,
    create_tasks_batch,
    get_task,
    update_task,
)
from zetta_utils.task_management.types import Task, TaskUpdate


def test_create_task_success(project_name, sample_task, clean_db, db_session):
    result = create_task(project_name=project_name, data=sample_task, db_session=db_session)
    assert result == "task_1"

    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["batch_id"] == "batch_1"


def test_get_task_success(existing_task, project_name, db_session):
    result = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert result["task_id"] == "task_1"


def test_get_task_not_found(project_name, clean_db, db_session):
    with pytest.raises(KeyError, match="Task task_1 not found"):
        get_task(project_name=project_name, task_id="task_1", db_session=db_session)


def test_update_task_success(clean_db, existing_task, project_name, db_session):
    update_data = TaskUpdate(**{"status": "ingested"})

    result = update_task(
        project_name=project_name, task_id="task_1", data=update_data, db_session=db_session
    )
    assert result is True

    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["status"] == "ingested"


def test_update_task_invalid_status(existing_task, project_name, db_session):
    """Test update_task with invalid status - covers the status validation line"""
    # Create an update with an invalid status value
    update_data = {"status": "invalid_status_value"}

    with pytest.raises(ValueError, match="Invalid status value"):
        update_task(
            project_name=project_name,
            task_id="task_1",
            data=update_data,  # type: ignore
            db_session=db_session,
        )


def test_create_task_validation(project_name, clean_db, db_session):
    # Test with invalid task data that doesn't match the Task type
    invalid_task = {"invalid_field": "value"}
    with pytest.raises(Exception):
        create_task(
            project_name=project_name, data=invalid_task, db_session=db_session  # type: ignore
        )


def test_get_task_not_found_different_project(existing_task, clean_db, db_session):
    with pytest.raises(KeyError, match="Task task_1 not found"):
        get_task(project_name="different_project", task_id="task_1", db_session=db_session)


def test_create_task_duplicate(project_name, clean_db, db_session):
    task_data = Task(
        **{
            "task_id": "duplicate_task",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "task_type": "segmentation",
            "ng_state": "http://example.com/duplicate_task",
        }
    )

    result = create_task(project_name=project_name, data=task_data, db_session=db_session)
    assert result == "duplicate_task"

    # Creating the same task again should succeed (return the task_id)
    result2 = create_task(project_name=project_name, data=task_data, db_session=db_session)
    assert result2 == "duplicate_task"

    task = get_task(project_name=project_name, task_id="duplicate_task", db_session=db_session)
    assert task["batch_id"] == "batch_1"
    assert task["status"] == "pending_ingestion"
    assert task["ng_state"] == "http://example.com/duplicate_task"


def test_get_task_error_case(project_name, clean_db, db_session):
    """Test error case for get_task"""
    with pytest.raises(KeyError, match="Task non_existent_task not found"):
        get_task(project_name=project_name, task_id="non_existent_task", db_session=db_session)


def test_create_task(project_name, clean_db, db_session):
    task_data = Task(
        **{
            "task_id": "task_1",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "task_type": "segmentation",
            "ng_state": "http://example.com/task_1",
        }
    )

    result = create_task(project_name=project_name, data=task_data, db_session=db_session)
    assert result == "task_1"

    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["batch_id"] == "batch_1"
    assert task["status"] == "pending_ingestion"
    assert task["ng_state"] == "http://example.com/task_1"


def test_update_task(project_name, clean_db, db_session):
    task_data = Task(
        **{
            "task_id": "task_2",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "task_type": "segmentation",
            "ng_state": "http://example.com/task_2",
        }
    )
    create_task(project_name=project_name, data=task_data, db_session=db_session)

    # Update with TaskUpdate type
    update_data = TaskUpdate(status="ingested", ng_state="http://updated.com")
    update_task(
        project_name=project_name, task_id="task_2", data=update_data, db_session=db_session
    )

    updated_task = get_task(project_name=project_name, task_id="task_2", db_session=db_session)
    assert updated_task["status"] == "ingested"
    assert updated_task["ng_state"] == "http://updated.com"


def test_create_tasks_batch_success(project_name, clean_db, db_session):
    """Test creating multiple tasks in batch"""
    tasks = [
        Task(
            **{
                "task_id": "batch_task_1",
                "batch_id": "batch_1",
                "status": "pending_ingestion",
                "task_type": "segmentation",
                "ng_state": "http://example.com/batch_task_1",
            }
        ),
        Task(
            **{
                "task_id": "batch_task_2",
                "batch_id": "batch_1",
                "status": "pending_ingestion",
                "task_type": "segmentation",
                "ng_state": "http://example.com/batch_task_2",
            }
        ),
    ]

    result = create_tasks_batch(project_name=project_name, tasks=tasks, db_session=db_session)
    assert len(result) == 2
    assert "batch_task_1" in result
    assert "batch_task_2" in result

    # Verify tasks were created
    task1 = get_task(project_name=project_name, task_id="batch_task_1", db_session=db_session)
    task2 = get_task(project_name=project_name, task_id="batch_task_2", db_session=db_session)
    assert task1["batch_id"] == "batch_1"
    assert task2["batch_id"] == "batch_1"


def test_create_tasks_batch_conflicting_content(project_name, clean_db, db_session):
    existing_task = Task(
        **{
            "task_id": "task_1",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "task_type": "segmentation",
            "ng_state": "http://example.com/task_1",
        }
    )

    # Create the existing task first
    create_task(project_name=project_name, data=existing_task, db_session=db_session)

    # Try to create a task with the same ID but different content
    conflicting_task = Task(
        **{
            "task_id": "task_1",
            "batch_id": "batch_1",
            "status": "ingested",  # Different status
            "task_type": "segmentation",
            "ng_state": "http://example.com/task_1_different",  # Different ng_state
        }
    )

    with pytest.raises(ValueError, match="Task task_1 already exists with different content"):
        create_tasks_batch(
            project_name=project_name, tasks=[conflicting_task], db_session=db_session
        )


def test_update_task_with_different_data_types(project_name, clean_db, db_session):
    """Test updating a task with different field data types"""
    task_data = Task(
        **{
            "task_id": "task_1",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "task_type": "segmentation",
            "ng_state": "http://example.com/task_1",
        }
    )
    create_task(project_name=project_name, data=task_data, db_session=db_session)

    # Update with TaskUpdate type
    update_data = TaskUpdate(status="ingested", ng_state="http://updated.com")
    update_task(
        project_name=project_name, task_id="task_1", data=update_data, db_session=db_session
    )

    updated_task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert updated_task["status"] == "ingested"
    assert updated_task["ng_state"] == "http://updated.com"


def test_update_task_nonexistent_task(project_name, clean_db, db_session):
    """Test updating a task that doesn't exist"""
    update_data = TaskUpdate(status="ingested")

    with pytest.raises(KeyError, match="Task task_1 not found"):
        update_task(
            project_name=project_name, task_id="task_1", data=update_data, db_session=db_session
        )
