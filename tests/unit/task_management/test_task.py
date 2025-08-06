"""Tests for task management task module"""

# pylint: disable=unused-argument,redefined-outer-name,too-many-lines

import time

import pytest

from zetta_utils.task_management.dependency import create_dependency
from zetta_utils.task_management.exceptions import (
    TaskValidationError,
    UserValidationError,
)
from zetta_utils.task_management.project import create_project
from zetta_utils.task_management.task import (
    _validate_task,
    create_task,
    get_max_idle_seconds,
    get_task,
    list_tasks_summary,
    reactivate_task,
    release_task,
    start_task,
    update_task,
)
from zetta_utils.task_management.task_type import create_task_type
from zetta_utils.task_management.types import (
    Dependency,
    Task,
    TaskType,
    TaskUpdate,
    User,
    UserUpdate,
)
from zetta_utils.task_management.user import create_user, get_user, update_user


@pytest.fixture
def existing_priority_tasks(clean_db, project_name, existing_task_type, db_session):
    tasks = [
        Task(
            **{
                "task_id": f"task_{i}",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "ng_state": {"url": f"http://example.com/{i}"},
                "ng_state_initial": {"url": f"http://example.com/{i}"},
                "priority": i,
                "batch_id": "batch_1",
                "task_type": existing_task_type["task_type"],
                "is_active": True,
                "last_leased_ts": 0.0,
                "completion_status": "",
            }
        )
        for i in range(1, 3)
    ]
    for task in tasks:
        create_task(project_name=project_name, data=task, db_session=db_session)
    yield tasks


@pytest.fixture
def sample_dependency() -> Dependency:
    return Dependency(
        **{
            "dependency_id": "dep_1",
            "task_id": "task_2",
            "dependent_on_task_id": "task_1",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )


@pytest.fixture
def existing_dependency_chain(
    clean_db, project_name, existing_priority_tasks, sample_dependency, db_session
):
    create_dependency(project_name=project_name, data=sample_dependency, db_session=db_session)
    update_task(
        project_name=project_name,
        task_id="task_2",
        data=TaskUpdate(completion_status="waiting_for_dependencies"),
        db_session=db_session,
    )
    yield sample_dependency


def test_create_task_success(project_name, sample_task, existing_task_type, db_session):
    """Test creating a new task"""
    result = create_task(project_name=project_name, data=sample_task, db_session=db_session)
    assert result == sample_task["task_id"]

    # Verify the task was created by retrieving it
    created_task = get_task(
        project_name=project_name, task_id=sample_task["task_id"], db_session=db_session
    )
    assert created_task["task_id"] == sample_task["task_id"]


def test_create_task_nonexistent_type(db_session, project_name):
    """Test creating a task with a nonexistent task type raises TaskValidationError"""
    # Try to create a task with a nonexistent task type
    task_with_invalid_type = Task(
        **{
            "task_id": "test_task",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": {"url": "http://example.com"},
            "ng_state_initial": {"url": "http://example.com"},
            "priority": 1,
            "batch_id": "batch_1",
            "task_type": "nonexistent_task_type",  # This type doesn't exist
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )

    # This should raise a TaskValidationError with the specific message
    with pytest.raises(TaskValidationError, match="Task type nonexistent_task_type not found"):
        create_task(project_name=project_name, data=task_with_invalid_type, db_session=db_session)


def test_get_task_success(db_session, existing_task, project_name):
    """Test retrieving a task that exists"""
    result = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert result["task_id"] == existing_task["task_id"]
    assert result["completion_status"] == existing_task["completion_status"]


def test_get_task_not_found(db_session, project_name):
    """Test retrieving a task that doesn't exist"""
    with pytest.raises(KeyError, match="Task nonexistent not found"):
        get_task(project_name=project_name, task_id="nonexistent", db_session=db_session)


def test_update_task_success(db_session, project_name, existing_task, existing_task_type):
    """Test updating a task"""
    update_data = TaskUpdate(**{"priority": 5, "assigned_user_id": "user_2"})
    result = update_task(
        project_name=project_name,
        task_id="task_1",
        data=update_data,
        db_session=db_session,
    )
    assert result is True

    # Check that the task was updated
    updated_task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert updated_task["priority"] == 5
    assert updated_task["assigned_user_id"] == "user_2"


def test_update_task_not_found(db_session, project_name):
    """Test updating a task that doesn't exist"""
    update_data = TaskUpdate(completion_status="in_progress")
    with pytest.raises(KeyError, match="Task nonexistent not found"):
        update_task(
            project_name=project_name,
            task_id="nonexistent",
            data=update_data,
            db_session=db_session,
        )


def test_start_task_success(db_session, existing_task, existing_user, project_name):
    """Test starting work on a task"""
    before_time = time.time()
    result = start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )
    after_time = time.time()

    assert result == "task_1"

    # Check that the task was updated
    updated_task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert updated_task["completion_status"] == ""
    assert updated_task["active_user_id"] == "test_user"
    assert before_time <= updated_task["last_leased_ts"] <= after_time

    # Check that the user was updated
    updated_user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert updated_user["active_task"] == "task_1"


def test_start_task_auto_select(db_session, existing_tasks, existing_user, project_name):
    """Test auto-selecting a task to work on"""
    result = start_task(
        project_name=project_name, user_id="test_user", task_id=None, db_session=db_session
    )

    # Should select the highest priority task (task_3)
    assert result == "task_3"

    # Check that the task was updated
    updated_task = get_task(project_name=project_name, task_id="task_3", db_session=db_session)
    assert updated_task["completion_status"] == ""
    assert updated_task["active_user_id"] == "test_user"

    # Check that the user was updated
    updated_user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert updated_user["active_task"] == "task_3"


def test_release_task_success(db_session, project_name, existing_task, existing_user):
    """Test releasing a task"""
    # First start work on the task
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )

    # Now release it with an empty completion status
    result = release_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )
    assert result is True

    # Check that the task was updated
    updated_task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert updated_task["completion_status"] == ""  # Empty status instead of "todo"
    assert updated_task["active_user_id"] == ""

    # Check that the user was updated
    updated_user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert updated_user["active_task"] == ""


def test_release_task_with_completion(db_session, project_name, existing_task, existing_user):
    """Test releasing a task with completion"""
    # First start work on the task
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )

    # Now release it with completion
    result = release_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        completion_status="done",
        db_session=db_session,
    )
    assert result is True

    # Check that the task was updated
    updated_task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert updated_task["completion_status"] == "done"
    assert updated_task["active_user_id"] == ""
    assert updated_task["completed_user_id"] == "test_user"

    # Check that the user was updated
    updated_user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert updated_user["active_task"] == ""


def test_update_task_missing_dependency_fields(
    db_session, project_name, existing_tasks, existing_task_type
):
    dep1 = Dependency(
        **{
            "dependency_id": "dep_1",
            "task_id": "task_2",
            "dependent_on_task_id": "task_1",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )
    create_dependency(project_name=project_name, data=dep1, db_session=db_session)

    # Make task_2 inactive initially
    update_task(
        project_name=project_name,
        task_id="task_2",
        data=TaskUpdate(is_active=False),
        db_session=db_session,
    )

    # Now complete task_1
    update_data = TaskUpdate(completion_status="done", completed_user_id="test_user")
    update_task(
        project_name=project_name,
        task_id="task_1",
        data=update_data,
        db_session=db_session,
    )

    # Check if dependent task was activated
    task2 = get_task(project_name=project_name, task_id="task_2", db_session=db_session)
    assert task2["is_active"] is True
    assert task2["completion_status"] == ""


@pytest.mark.parametrize(
    "invalid_data, expected_error",
    [
        ({"task_type": "nonexistent_type"}, "Task type not found"),
        (
            {"completion_status": "invalid"},
            "Completion status 'invalid' not allowed for this task type",
        ),
    ],
)
def test_update_task_validation(
    db_session, invalid_data, expected_error, existing_task, project_name
):
    with pytest.raises(TaskValidationError, match=expected_error):
        update_task(
            project_name=project_name,
            task_id="task_1",
            data=invalid_data,
            db_session=db_session,
        )


def test_inactive_task_rejects_completion_status(db_session, project_name, existing_task):
    """Test that inactive tasks reject completion status updates"""
    # First make the task inactive
    update_task(
        project_name=project_name,
        task_id="task_1",
        data=TaskUpdate(is_active=False),
        db_session=db_session,
    )

    # Try to complete it
    update_data = TaskUpdate(completion_status="done", completed_user_id="test_user")
    with pytest.raises(TaskValidationError, match="Inactive task cannot have completion status"):
        update_task(
            project_name=project_name,
            task_id="task_1",
            data=update_data,
            db_session=db_session,
        )

    # Check that completion_status didn't change and completed_user_id is empty
    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["completion_status"] == ""  # Should remain empty
    assert task["completed_user_id"] == ""


def test_setting_task_inactive_clears_completion_status(db_session, project_name, existing_task):
    """Test that setting a task to inactive clears its completion status"""
    # First set it to active and complete it
    update_task(
        project_name=project_name,
        task_id="task_1",
        data=TaskUpdate(is_active=True),
        db_session=db_session,
    )
    update_task(
        project_name=project_name,
        task_id="task_1",
        data=TaskUpdate(completion_status="done", completed_user_id="test_user"),
        db_session=db_session,
    )

    # Verify it was completed
    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["completion_status"] == "done"
    assert task["completed_user_id"] == "test_user"


def test_inactive_task_rejects_completed_user(db_session, project_name, existing_task):
    """Test that inactive tasks reject completed_user_id updates"""
    # First make the task inactive
    update_task(
        project_name=project_name,
        task_id="task_1",
        data=TaskUpdate(is_active=False),
        db_session=db_session,
    )

    # Try to set a completed_user_id without completion status
    update_data = TaskUpdate(completed_user_id="test_user")
    with pytest.raises(TaskValidationError, match="Inactive task cannot have completed user"):
        update_task(
            project_name=project_name,
            task_id="task_1",
            data=update_data,
            db_session=db_session,
        )

    # Check that completed_user_id is still empty
    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["completed_user_id"] == ""


def test_completed_task_requires_completed_user(db_session, project_name, existing_task):
    """Test that completed tasks must have a completed_user_id"""
    # Try to set a completion status without a completed_user_id
    update_data = TaskUpdate(completion_status="done")
    with pytest.raises(TaskValidationError, match="Completed task must have completed_user_id"):
        update_task(
            project_name=project_name,
            task_id="task_1",
            data=update_data,
            db_session=db_session,
        )

    # Check that completion_status didn't change
    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["completion_status"] == ""  # Should remain empty

    # Now try with both fields set - this should work
    update_data = TaskUpdate(completion_status="done", completed_user_id="test_user")
    update_task(
        project_name=project_name,
        task_id="task_1",
        data=update_data,
        db_session=db_session,
    )

    # Verify both fields were updated
    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["completion_status"] == "done"
    assert task["completed_user_id"] == "test_user"


def test_create_task_duplicate_id(db_session, project_name, existing_task):
    """Test that creating a task with an existing ID raises an error"""
    # Try to create a task with the same ID as an existing one
    duplicate_task = Task(
        **{
            "task_id": "task_1",  # This ID already exists
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": {"url": "http://example.com/new"},
            "ng_state_initial": {"url": "http://example.com/"},
            "priority": 2,
            "batch_id": "batch_2",
            "task_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )

    # This should raise a TaskValidationError with a specific message
    with pytest.raises(TaskValidationError, match="Task task_1 already exists"):
        create_task(project_name=project_name, data=duplicate_task, db_session=db_session)

    # Verify the original task wasn't modified
    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["ng_state"] == {"url": "http://example.com"}  # Original ng_state, not the new one
    assert task["priority"] == 1  # Original priority, not the new one


def test_task_type_without_completion_statuses(db_session, project_name, existing_task):
    """Test that a task with a type that has no completion statuses cannot be completed"""
    # Create a task type without completion_statuses
    invalid_type = TaskType(task_type="no_completion_type", completion_statuses=[])
    create_task_type(project_name=project_name, data=invalid_type, db_session=db_session)

    # First update the task to use this type
    update_task(
        project_name=project_name,
        task_id="task_1",
        data=TaskUpdate(task_type="no_completion_type"),
        db_session=db_session,
    )

    # Now try to set a completion status
    update_data = TaskUpdate(completion_status="done", completed_user_id="test_user")
    with pytest.raises(
        TaskValidationError,
        match="Completion status 'done' not allowed for this task type",
    ):
        update_task(
            project_name=project_name,
            task_id="task_1",
            data=update_data,
            db_session=db_session,
        )

    # Check that completion_status didn't change
    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["completion_status"] == ""  # Should remain empty
    assert task["completed_user_id"] == ""


def test_task_type_missing_completion_statuses_key(
    db_session, project_name, existing_task, mocker
):
    """Test validation when task type dict is missing completion_statuses key entirely"""
    # Create a mock task type that doesn't have completion_statuses in its to_dict result
    mock_task_type = mocker.Mock()
    mock_task_type.to_dict.return_value = {
        "task_type": "test_type",
        "description": "Test type",
        # Missing "completion_statuses" key
    }

    # Mock the database query to return our mock task type
    mock_execute = mocker.patch.object(db_session, "execute")
    mock_result = mocker.Mock()
    mock_result.scalar_one.return_value = mock_task_type
    mock_execute.return_value = mock_result

    # Test task data that would trigger the validation
    test_task = {
        "task_type": "test_type",
        "completion_status": "done",
        "completed_user_id": "test_user",
        "is_active": True,
    }

    # This should raise the specific validation error we're testing
    with pytest.raises(
        TaskValidationError,
        match="Task type test_type has no valid completion statuses",
    ):
        _validate_task(db_session, project_name, test_task)


def test_update_nonexistent_task(db_session, project_name):
    """Test that updating a nonexistent task raises an error"""
    # Try to update a task that doesn't exist
    update_data = TaskUpdate(priority=5, assigned_user_id="user_2")

    # This should raise a KeyError with a specific message
    with pytest.raises(KeyError, match="Task nonexistent_task not found"):
        update_task(
            project_name=project_name,
            task_id="nonexistent_task",
            data=update_data,
            db_session=db_session,
        )


def test_start_task_nonexistent_user(db_session, project_name, existing_task):
    """Test that starting a task with a nonexistent user raises an error"""
    # Try to start a task with a user that doesn't exist
    nonexistent_user_id = "nonexistent_user"

    # This should raise a UserValidationError with a specific message
    with pytest.raises(UserValidationError, match=f"User {nonexistent_user_id} not found"):
        start_task(
            project_name=project_name,
            user_id=nonexistent_user_id,
            task_id="task_1",
            db_session=db_session,
        )

    # Verify that the task wasn't modified
    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["active_user_id"] == ""  # Should remain empty

    # Verify no user was created by trying to get the user and expecting an error
    with pytest.raises(KeyError):
        get_user(project_name=project_name, user_id=nonexistent_user_id, db_session=db_session)


def test_start_task_user_already_has_active(
    db_session, project_name, existing_task, existing_user
):
    """Test that a user with an active task cannot start another one"""
    # Create a second task
    second_task = Task(
        **{
            "task_id": "task_2",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": {"url": "http://example.com/second"},
            "ng_state_initial": {"url": "http://example.com/"},
            "priority": 2,
            "batch_id": "batch_1",
            "task_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    create_task(project_name=project_name, data=second_task, db_session=db_session)

    # First, start one task
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )

    # Now try to start a second task with the same user
    with pytest.raises(
        UserValidationError,
        match="User already has an active task task_1 which is different "
        "from requested task task_2",
    ):
        start_task(
            project_name=project_name,
            user_id="test_user",
            task_id="task_2",
            db_session=db_session,
        )

    # Verify the first task is still active for the user
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_task"] == "task_1"

    # Verify the second task wasn't started
    task2 = get_task(project_name=project_name, task_id="task_2", db_session=db_session)
    assert task2["active_user_id"] == ""  # Should remain empty


def test_start_nonexistent_task(db_session, project_name, existing_user):
    """Test that starting a nonexistent task raises an error"""
    # Try to start a task that doesn't exist
    nonexistent_task_id = "nonexistent_task"

    # This should raise a TaskValidationError with a specific message
    with pytest.raises(TaskValidationError, match=f"Task {nonexistent_task_id} not found"):
        start_task(
            project_name=project_name,
            user_id="test_user",
            task_id=nonexistent_task_id,
            db_session=db_session,
        )

    # Verify the user's active_task wasn't modified
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_task"] == ""

    # Verify that no task was created with this ID by trying to get it and expecting an error
    with pytest.raises(KeyError):
        get_task(
            project_name=project_name,
            task_id=nonexistent_task_id,
            db_session=db_session,
        )


def test_start_task_takeover_idle(db_session, project_name, existing_task, existing_user):
    """Test taking over an idle task from another user"""
    second_user = User(
        **{
            "user_id": "user_2",
            "hourly_rate": 45.0,
            "active_task": "",
            "qualified_task_types": ["segmentation_proofread"],
        }
    )
    create_user(project_name=project_name, data=second_user, db_session=db_session)

    # First, have user_1 start the task
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )

    # Verify user_1 has the task
    user1 = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user1["active_task"] == "task_1"

    # Manually update the last_leased_ts to be older than max idle seconds
    old_time = time.time() - (get_max_idle_seconds() + 10)
    update_task(
        project_name=project_name,
        task_id="task_1",
        data=TaskUpdate(last_leased_ts=old_time),
        db_session=db_session,
    )

    # Now have user_2 take over the task
    start_task(
        project_name=project_name,
        user_id="user_2",
        task_id="task_1",
        db_session=db_session,
    )

    # Verify user_2 now has the task
    user2 = get_user(project_name=project_name, user_id="user_2", db_session=db_session)
    assert user2["active_task"] == "task_1"

    # Verify user_1 no longer has the task
    user1 = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user1["active_task"] == ""

    # Verify the task is now assigned to user_2
    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["active_user_id"] == "user_2"
    assert task["last_leased_ts"] > old_time


def test_start_task_already_active(db_session, project_name, existing_task, existing_user):
    """Test that starting an already active task raises an error"""
    # Create a second user
    second_user = User(
        **{
            "user_id": "user_2",
            "hourly_rate": 45.0,
            "active_task": "",
            "qualified_task_types": ["segmentation_proofread"],
        }
    )
    create_user(project_name=project_name, data=second_user, db_session=db_session)

    # First, have user_1 start the task
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )

    # Verify user_1 has the task
    user1 = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user1["active_task"] == "task_1"

    # Now have user_2 try to take over the task (which is still active)
    with pytest.raises(TaskValidationError, match="Task is no longer available for takeover"):
        start_task(
            project_name=project_name,
            user_id="user_2",
            task_id="task_1",
            db_session=db_session,
        )

    # Verify user_1 still has the task
    user1 = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user1["active_task"] == "task_1"

    # Verify user_2 still has no active task
    user2 = get_user(project_name=project_name, user_id="user_2", db_session=db_session)
    assert user2["active_task"] == ""

    # Verify the task is still assigned to user_1
    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["active_user_id"] == "test_user"


def test_start_task_requires_qualification(
    db_session, project_name, existing_user, existing_task, existing_task_type
):
    """Test that a user can't start a task if they're not qualified for it"""
    # Update user to have empty qualifications list
    update_user(
        project_name=project_name,
        user_id="test_user",
        data=UserUpdate(
            hourly_rate=50.0,
            active_task="",
            qualified_task_types=[],  # Empty list means no qualifications
        ),
        db_session=db_session,
    )

    # Try to start task - this should raise an error
    with pytest.raises(UserValidationError, match="User not qualified for this task type"):
        start_task(
            project_name=project_name,
            user_id="test_user",
            task_id="task_1",
            db_session=db_session,
        )

    # Verify the user's active_task wasn't modified
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_task"] == ""

    # Verify the task wasn't modified
    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["active_user_id"] == ""

    # Now update the user to have the qualification and try again
    update_user(
        project_name=project_name,
        user_id="test_user",
        data=UserUpdate(qualified_task_types=["segmentation_proofread"]),
        db_session=db_session,
    )

    # This should now succeed
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )

    # Verify the user now has the task
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_task"] == "task_1"

    # Verify the task is now assigned to the user
    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["active_user_id"] == "test_user"


def test_start_task_user_not_qualified_for_specific_type(
    db_session, project_name, existing_task_type
):
    """Test that a user with qualifications can't start a task they're not qualified for"""
    # Create a user qualified for one type but not another
    qualified_user = User(
        user_id="partially_qualified_user",
        hourly_rate=50.0,
        active_task="",
        qualified_task_types=["different_type"],  # Qualified for different type only
    )
    create_user(project_name=project_name, data=qualified_user, db_session=db_session)

    # Create a task with the existing task type (which user is not qualified for)
    test_task = Task(
        task_id="unqualified_task",
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state={"url": "http://example.com/unqualified"},
        ng_state_initial={"url": "http://example.com/unqualified"},
        priority=1,
        batch_id="batch_1",
        task_type=existing_task_type["task_type"],  # User not qualified for this type
        is_active=True,
        last_leased_ts=0.0,
        completion_status="",
    )
    create_task(project_name=project_name, data=test_task, db_session=db_session)

    # Try to start the task - should fail because user not qualified for this specific type
    with pytest.raises(UserValidationError, match="User not qualified for this task type"):
        start_task(
            project_name=project_name,
            user_id="partially_qualified_user",
            task_id="unqualified_task",
            db_session=db_session,
        )

    # Verify the user's active_task wasn't modified
    user = get_user(
        project_name=project_name,
        user_id="partially_qualified_user",
        db_session=db_session,
    )
    assert user["active_task"] == ""

    # Verify the task wasn't modified
    task = get_task(
        project_name=project_name,
        task_id="unqualified_task",
        db_session=db_session,
    )
    assert task["active_user_id"] == ""


def test_release_task_nonexistent_task(db_session, project_name, existing_user):
    """Test that releasing a nonexistent task raises an error"""
    # Update user to have nonexistent active task
    update_user(
        project_name=project_name,
        user_id="test_user",
        data=UserUpdate(active_task="nonexistent_task"),
        db_session=db_session,
    )

    # Verify the user has been updated
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_task"] == "nonexistent_task"

    # Now try to release the task
    with pytest.raises(TaskValidationError, match="Task nonexistent_task not found"):
        release_task(
            project_name=project_name,
            user_id="test_user",
            task_id="nonexistent_task",
            db_session=db_session,
        )

    # Verify the user's active_task wasn't modified
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_task"] == "nonexistent_task"


def test_auto_select_task_no_qualified_types(db_session, project_name, existing_task):
    """Test that auto-selecting a task returns None when user has no qualified types"""
    # Create a user with no qualified task types
    unqualified_user = User(
        **{
            "user_id": "unqualified_user",
            "hourly_rate": 45.0,
            "active_task": "",
            "qualified_task_types": [],  # Empty list means no qualifications
        }
    )
    create_user(project_name=project_name, data=unqualified_user, db_session=db_session)

    # Try to auto-select a task (by not specifying a task_id)
    result = start_task(
        project_name=project_name,
        user_id="unqualified_user",
        task_id=None,
        db_session=db_session,
    )

    # Verify that no task was selected
    assert result is None

    # Verify the user's active_task wasn't modified
    user = get_user(project_name=project_name, user_id="unqualified_user", db_session=db_session)
    assert user["active_task"] == ""

    # Now update the user to have a qualification and try again
    update_user(
        project_name=project_name,
        user_id="unqualified_user",
        data=UserUpdate(qualified_task_types=["segmentation_proofread"]),
        db_session=db_session,
    )

    # This should now succeed
    result = start_task(
        project_name=project_name,
        user_id="unqualified_user",
        task_id=None,
        db_session=db_session,
    )

    # Verify a task was selected
    assert result is not None

    # Verify the user now has the task
    user = get_user(project_name=project_name, user_id="unqualified_user", db_session=db_session)
    assert user["active_task"] == result

    # Verify the task is now assigned to the user
    task = get_task(project_name=project_name, task_id=result, db_session=db_session)
    assert task["active_user_id"] == "unqualified_user"


def test_auto_select_task_prioritizes_assigned_to_user(
    db_session, project_name, existing_user, existing_task_type, task_factory
):
    # Create the project first
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    task_factory("assigned_task", assigned_user_id="test_user", priority=1)

    task_factory("unassigned_task", assigned_user_id="", priority=10)  # Higher priority

    # Auto-select a task (by not specifying a task_id)
    result = start_task(
        project_name=project_name,
        user_id="test_user",
        task_id=None,
        db_session=db_session,
    )

    # Verify that the assigned task was selected, even though it has lower priority
    assert result == "assigned_task"

    # Verify the user now has the assigned task
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_task"] == "assigned_task"

    # Verify the task is now assigned to the user
    task = get_task(project_name=project_name, task_id="assigned_task", db_session=db_session)
    assert task["active_user_id"] == "test_user"

    # Verify the unassigned task wasn't modified
    unassigned = get_task(
        project_name=project_name, task_id="unassigned_task", db_session=db_session
    )
    assert unassigned["active_user_id"] == ""


def test_release_task_no_active_task(db_session, project_name, existing_user):
    """Test that releasing a task fails when the user doesn't have an active task"""
    # Verify the user doesn't have an active task
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_task"] == ""

    # Try to release a task
    with pytest.raises(UserValidationError, match="User does not have an active task"):
        release_task(
            project_name=project_name,
            user_id="test_user",
            task_id="task_1",
            db_session=db_session,
        )

    # Try to release with a completion status
    with pytest.raises(UserValidationError, match="User does not have an active task"):
        release_task(
            project_name=project_name,
            user_id="test_user",
            task_id="task_1",
            completion_status="done",
            db_session=db_session,
        )

    # Verify the user's active_task is still empty
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_task"] == ""


def test_release_task_with_dependencies(
    db_session, project_name, existing_task_type, task_factory
):
    """Test releasing a task with dependencies updates the dependent tasks"""
    # Create the project first
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    # Create a user
    user_data = User(
        user_id="dep_test_user",
        hourly_rate=50.0,
        active_task="",
        qualified_task_types=[existing_task_type["task_type"]],
    )
    create_user(project_name=project_name, data=user_data, db_session=db_session)

    # Create tasks using factory fixtures
    task_factory("task_dep_1", assigned_user_id="dep_test_user", is_active=True)
    task_factory(
        "task_dep_2",
        is_active=False,  # Initially inactive until dependency is satisfied
    )

    # Create a dependency between them
    dependency_data = Dependency(
        dependency_id="dep_test_1",
        task_id="task_dep_2",  # This task depends on task_dep_1
        dependent_on_task_id="task_dep_1",
        required_completion_status="done",
        is_satisfied=False,
    )
    create_dependency(project_name=project_name, data=dependency_data, db_session=db_session)

    # Start work on task1
    start_task(
        project_name=project_name,
        user_id="dep_test_user",
        task_id="task_dep_1",
        db_session=db_session,
    )

    # Verify task2 is still inactive
    task2_before = get_task(project_name=project_name, task_id="task_dep_2", db_session=db_session)
    assert task2_before["is_active"] is False

    # Release task1 with completion status "done"
    result = release_task(
        project_name=project_name,
        user_id="dep_test_user",
        task_id="task_dep_1",
        completion_status="done",
        db_session=db_session,
    )
    assert result is True

    # Verify task1 was updated correctly
    task1_after = get_task(project_name=project_name, task_id="task_dep_1", db_session=db_session)
    assert task1_after["completion_status"] == "done"
    assert task1_after["active_user_id"] == ""
    assert task1_after["completed_user_id"] == "dep_test_user"


def test_start_task_returns_current_active_when_auto_selecting(
    db_session, project_name, existing_task, existing_user
):
    # First, start a task for the user
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )

    # Verify the user has an active task
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_task"] == "task_1"

    # Now call start_task with task_id=None (auto-select mode)
    # This should return the user's current active task
    result = start_task(
        project_name=project_name,
        user_id="test_user",
        task_id=None,  # Auto-select mode
        db_session=db_session,
    )

    # Should return the same task the user already has active
    assert result == "task_1"

    # User should still have the same active task
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_task"] == "task_1"


def test_release_task_mismatched_id(db_session, project_name, existing_task, existing_user):
    """Test that releasing a task with wrong ID raises validation error"""
    # Create a second task
    task2 = Task(
        **{
            "task_id": "task_2",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": {"url": "http://example.com/2"},
            "ng_state_initial": {"url": "http://example.com/2"},
            "priority": 2,
            "batch_id": "batch_1",
            "task_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    create_task(project_name=project_name, data=task2, db_session=db_session)

    # Start task_1 for the user
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )

    # Try to release task_2 (which is not the user's active task)
    with pytest.raises(
        UserValidationError,
        match="Task ID does not match user's active task",
    ):
        release_task(
            project_name=project_name,
            user_id="test_user",
            task_id="task_2",  # Wrong task ID
            db_session=db_session,
        )


def test_reactivate_task_success(db_session, project_name, existing_task, existing_user):
    """Test successfully reactivating a completed task"""
    # First start and complete the task
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )

    release_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        completion_status="pass",
        db_session=db_session,
    )

    # Verify task is completed
    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["completion_status"] == "pass"
    assert task["completed_user_id"] == "test_user"

    # Now reactivate it
    result = reactivate_task(
        project_name=project_name,
        task_id="task_1",
        db_session=db_session,
    )

    assert result is True

    # Verify task is now active again
    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["completion_status"] == ""
    assert task["completed_user_id"] == ""


def test_reactivate_task_already_active(db_session, project_name, existing_task, existing_user):
    """Test that reactivating an already active task raises an error"""
    # Task is already active (not completed)
    task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert task["completion_status"] == ""

    # Try to reactivate - should fail
    with pytest.raises(TaskValidationError, match="Task task_1 is already active"):
        reactivate_task(
            project_name=project_name,
            task_id="task_1",
            db_session=db_session,
        )


def test_reactivate_task_nonexistent(db_session, project_name):
    """Test that reactivating a nonexistent task raises KeyError"""
    with pytest.raises(KeyError, match="Task nonexistent_task not found"):
        reactivate_task(
            project_name=project_name,
            task_id="nonexistent_task",
            db_session=db_session,
        )


def test_list_tasks_summary(db_session, project_name, existing_task_type, existing_user):
    """Test getting task summary for a project"""
    # Create some tasks with different states
    # Active unpaused tasks
    for i in range(7):
        create_task(
            project_name=project_name,
            data={
                "task_id": f"active_{i}",
                "completion_status": "",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "ng_state": {"url": "http://example.com"},
                "ng_state_initial": {"url": "http://example.com"},
                "priority": i,
                "batch_id": "batch_1",
                "task_type": "segmentation_proofread",
                "is_active": True,
                "is_paused": False,
                "last_leased_ts": 0.0,
            },
            db_session=db_session,
        )

    # Active paused tasks
    for i in range(3):
        create_task(
            project_name=project_name,
            data={
                "task_id": f"paused_{i}",
                "completion_status": "",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "ng_state": {"url": "http://example.com"},
                "ng_state_initial": {"url": "http://example.com"},
                "priority": i,
                "batch_id": "batch_1",
                "task_type": "segmentation_proofread",
                "is_active": True,
                "is_paused": True,
                "last_leased_ts": 0.0,
            },
            db_session=db_session,
        )

    # Completed tasks
    for i in range(4):
        create_task(
            project_name=project_name,
            data={
                "task_id": f"completed_{i}",
                "completion_status": "done",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "test_user",
                "ng_state": {"url": "http://example.com"},
                "ng_state_initial": {"url": "http://example.com"},
                "priority": i,
                "batch_id": "batch_1",
                "task_type": "segmentation_proofread",
                "is_active": True,
                "is_paused": False,
                "last_leased_ts": 0.0,
            },
            db_session=db_session,
        )

    # Get summary
    summary = list_tasks_summary(project_name=project_name, db_session=db_session)

    # Verify counts
    assert summary["active_count"] == 10  # 7 unpaused + 3 paused
    assert summary["completed_count"] == 4
    assert summary["paused_count"] == 3

    # Verify active unpaused IDs (should be top 5 by priority)
    assert len(summary["active_unpaused_ids"]) == 5
    assert summary["active_unpaused_ids"] == [
        "active_6",
        "active_5",
        "active_4",
        "active_3",
        "active_2",
    ]

    # Verify active paused IDs
    assert len(summary["active_paused_ids"]) == 3
    assert summary["active_paused_ids"] == ["paused_2", "paused_1", "paused_0"]


def test_list_tasks_summary_empty_project(db_session, project_name):
    """Test getting task summary for empty project"""
    summary = list_tasks_summary(project_name=project_name, db_session=db_session)

    assert summary["active_count"] == 0
    assert summary["completed_count"] == 0
    assert summary["paused_count"] == 0
    assert not summary["active_unpaused_ids"]
    assert not summary["active_paused_ids"]


def test_first_start_ts_set_on_initial_start(
    db_session, existing_task, existing_user, project_name
):
    """Test that first_start_ts is set when a task is started for the first time"""
    # Verify the task initially has no first_start_ts
    initial_task = get_task(
        project_name=project_name, task_id="task_1", db_session=db_session
    )
    assert initial_task["first_start_ts"] is None

    # Start the task for the first time
    before_time = time.time()
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )
    after_time = time.time()

    # Verify first_start_ts was set
    updated_task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert updated_task["first_start_ts"] is not None
    assert before_time <= updated_task["first_start_ts"] <= after_time


def test_first_start_ts_unchanged_on_restart(
    db_session, existing_task, existing_user, project_name
):
    """Test that first_start_ts remains unchanged when a task is restarted"""
    # Start the task for the first time
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )

    # Get the first_start_ts after initial start
    first_started_task = get_task(
        project_name=project_name, task_id="task_1", db_session=db_session
    )
    original_first_start_ts = first_started_task["first_start_ts"]
    assert original_first_start_ts is not None

    # Release the task
    release_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )

    # Wait a bit to ensure timestamp would be different if it were reset
    time.sleep(0.1)

    # Start the task again
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )

    # Verify first_start_ts remained unchanged
    restarted_task = get_task(
        project_name=project_name, task_id="task_1", db_session=db_session
    )
    assert restarted_task["first_start_ts"] == original_first_start_ts


def test_first_start_ts_set_on_takeover_when_null(
    db_session, existing_task, existing_user, project_name
):
    """Test that first_start_ts is set during task takeover only if it's null"""
    # Create a second user
    second_user = User(
        **{
            "user_id": "user_2",
            "hourly_rate": 45.0,
            "active_task": "",
            "qualified_task_types": ["segmentation_proofread"],
        }
    )
    create_user(project_name=project_name, data=second_user, db_session=db_session)

    # Verify the task initially has no first_start_ts
    initial_task = get_task(
        project_name=project_name, task_id="task_1", db_session=db_session
    )
    assert initial_task["first_start_ts"] is None

    # Have user_1 start the task
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )

    # Manually update the last_leased_ts to be older than max idle seconds
    old_time = time.time() - (get_max_idle_seconds() + 10)
    update_task(
        project_name=project_name,
        task_id="task_1",
        data=TaskUpdate(last_leased_ts=old_time),
        db_session=db_session,
    )

    # Get the first_start_ts after initial start
    first_started_task = get_task(
        project_name=project_name, task_id="task_1", db_session=db_session
    )
    original_first_start_ts = first_started_task["first_start_ts"]
    assert original_first_start_ts is not None

    # Wait a bit to ensure timestamp would be different if it were reset
    time.sleep(0.1)

    # Now have user_2 take over the task
    start_task(
        project_name=project_name,
        user_id="user_2",
        task_id="task_1",
        db_session=db_session,
    )

    # Verify first_start_ts remained unchanged (was not reset during takeover)
    taken_over_task = get_task(
        project_name=project_name, task_id="task_1", db_session=db_session
    )
    assert taken_over_task["first_start_ts"] == original_first_start_ts
    assert taken_over_task["active_user_id"] == "user_2"


def test_first_start_ts_set_on_takeover_when_task_never_started(
    db_session, project_name, existing_task_type
):
    """Test that first_start_ts is set during task takeover when task never started before"""
    # Create two users
    user1 = User(
        **{
            "user_id": "user_1",
            "hourly_rate": 45.0,
            "active_task": "",
            "qualified_task_types": ["segmentation_proofread"],
        }
    )
    user2 = User(
        **{
            "user_id": "user_2",
            "hourly_rate": 45.0,
            "active_task": "",
            "qualified_task_types": ["segmentation_proofread"],
        }
    )
    create_user(project_name=project_name, data=user1, db_session=db_session)
    create_user(project_name=project_name, data=user2, db_session=db_session)

    # Create a task and manually assign it to user_1 without calling start_task
    test_task = Task(
        task_id="takeover_test_task",
        assigned_user_id="",
        active_user_id="user_1",  # Manually assigned
        completed_user_id="",
        ng_state={"url": "http://example.com/takeover"},
        ng_state_initial={"url": "http://example.com/takeover"},
        priority=1,
        batch_id="batch_1",
        task_type="segmentation_proofread",
        is_active=True,
        last_leased_ts=time.time() - (get_max_idle_seconds() + 10),
        completion_status="",
    )
    create_task(project_name=project_name, data=test_task, db_session=db_session)

    # Update user_1 to have this as active task
    update_user(
        project_name=project_name,
        user_id="user_1",
        data=UserUpdate(active_task="takeover_test_task"),
        db_session=db_session,
    )

    # Verify the task has no first_start_ts
    initial_task = get_task(
        project_name=project_name,
        task_id="takeover_test_task",
        db_session=db_session,
    )
    assert initial_task["first_start_ts"] is None

    # Now have user_2 take over the task
    before_takeover = time.time()
    start_task(
        project_name=project_name,
        user_id="user_2",
        task_id="takeover_test_task",
        db_session=db_session,
    )
    after_takeover = time.time()

    # Verify first_start_ts was set during takeover
    taken_over_task = get_task(
        project_name=project_name,
        task_id="takeover_test_task",
        db_session=db_session,
    )
    assert taken_over_task["first_start_ts"] is not None
    assert before_takeover <= taken_over_task["first_start_ts"] <= after_takeover
    assert taken_over_task["active_user_id"] == "user_2"


def test_first_start_ts_null_on_task_creation(project_name, existing_task_type, db_session):
    """Test that first_start_ts is None when a task is created"""
    # Create a new task
    new_task = Task(
        task_id="creation_test_task",
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state={"url": "http://example.com/creation"},
        ng_state_initial={"url": "http://example.com/creation"},
        priority=1,
        batch_id="batch_1",
        task_type="segmentation_proofread",
        is_active=True,
        last_leased_ts=0.0,
        completion_status="",
    )

    # Create the task
    result = create_task(
        project_name=project_name, data=new_task, db_session=db_session
    )
    assert result == "creation_test_task"

    # Verify the task was created with first_start_ts as None
    created_task = get_task(
        project_name=project_name,
        task_id="creation_test_task",
        db_session=db_session,
    )
    assert created_task["first_start_ts"] is None
    assert created_task["task_id"] == "creation_test_task"


def test_first_start_ts_can_be_set_explicitly_on_creation(
    project_name, existing_task_type, db_session
):
    """Test that first_start_ts can be explicitly set when creating a task"""
    explicit_start_time = time.time() - 3600

    # Create a task with explicit first_start_ts
    new_task = Task(
        task_id="explicit_first_start_task",
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state={"url": "http://example.com/explicit"},
        ng_state_initial={"url": "http://example.com/explicit"},
        priority=1,
        batch_id="batch_1",
        task_type="segmentation_proofread",
        is_active=True,
        last_leased_ts=0.0,
        completion_status="",
        first_start_ts=explicit_start_time,
    )

    # Create the task
    result = create_task(
        project_name=project_name, data=new_task, db_session=db_session
    )
    assert result == "explicit_first_start_task"

    # Verify the task was created with the explicit first_start_ts
    created_task = get_task(
        project_name=project_name,
        task_id="explicit_first_start_task",
        db_session=db_session,
    )
    assert created_task["first_start_ts"] == explicit_start_time
    assert created_task["task_id"] == "explicit_first_start_task"


def test_release_task_with_note(db_session, existing_task, existing_user, project_name):
    """Test that release_task can save a note to the task"""
    # First start work on the task
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )

    # Release it with a note
    test_note = "Task completed successfully with some observations"
    result = release_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        completion_status="done",
        note=test_note,
        db_session=db_session,
    )
    assert result is True

    # Check that the task was updated with the note
    updated_task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert updated_task["note"] == test_note
    assert updated_task["completion_status"] == "done"
    assert updated_task["active_user_id"] == ""
    assert updated_task["completed_user_id"] == "test_user"


def test_release_task_without_note(db_session, existing_task, existing_user, project_name):
    """Test that release_task works without providing a note"""
    # First start work on the task
    start_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        db_session=db_session,
    )

    # Release it without a note
    result = release_task(
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        completion_status="done",
        db_session=db_session,
    )
    assert result is True

    # Check that the task was updated but note remains None
    updated_task = get_task(project_name=project_name, task_id="task_1", db_session=db_session)
    assert updated_task["note"] is None
    assert updated_task["completion_status"] == "done"
    assert updated_task["active_user_id"] == ""
    assert updated_task["completed_user_id"] == "test_user"


def test_get_task_seed_id_integration(
        db_session, project_name, existing_task_type, mocker
):
    """Integration test: Create task with seed_id format, verify get_task processes it correctly"""
    # Mock the get_segment_ng_state function to return a predictable result
    mock_generated_state = {
        "dimensions": {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [40e-9, "m"]},
        "position": [100, 200, 300],
        "layers": [
            {
                "type": "segmentation",
                "source": "gs://test-bucket/segmentation",
                "segments": ["12345"],
                "name": "Segmentation"
            },
            {
                "type": "annotation",
                "name": "Seed Location",
                "annotationColor": "#ff00ff",
                "annotations": [{"point": [100, 200, 300], "type": "point", "id": "abc123"}]
            }
        ]
    }

    mock_get_segment_ng_state = mocker.patch(
        "zetta_utils.task_management.task.get_segment_ng_state",
        return_value=mock_generated_state
    )

    # Create a task with seed_id format in ng_state
    seed_id = 74732294451380972
    integration_task = Task(
        **{
            "task_id": "integration_seed_task",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": {"seed_id": seed_id},
            "ng_state_initial": {"seed_id": seed_id},
            "priority": 1,
            "batch_id": "batch_1",
            "task_type": existing_task_type["task_type"],
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    create_task(project_name=project_name, data=integration_task, db_session=db_session)

    # Verify initial state in database has seed_id format
    initial_result = get_task(
        project_name=project_name,
        task_id="integration_seed_task",
        process_ng_state=False,  # Don't process yet
        db_session=db_session,
    )
    assert initial_result["ng_state"] == {"seed_id": seed_id}
    assert initial_result["ng_state_initial"] == {"seed_id": seed_id}

    # Now call get_task with processing enabled (default behavior)
    processed_result = get_task(
        project_name=project_name,
        task_id="integration_seed_task",
        db_session=db_session,
    )

    # Verify that get_segment_ng_state was called with correct parameters
    mock_get_segment_ng_state.assert_called_once_with(
        project_name=project_name,
        seed_id=seed_id,
        include_certain_ends=True,
        include_uncertain_ends=True,
        include_breadcrumbs=True,
        include_segment_type_layers=True,
        db_session=mocker.ANY
    )

    # Verify that the result now contains the generated ng_state
    assert processed_result["task_id"] == "integration_seed_task"
    assert processed_result["ng_state"] == mock_generated_state
    assert processed_result["ng_state_initial"] == mock_generated_state

    # Verify that calling get_task again doesn't re-process (since it's already processed)
    mock_get_segment_ng_state.reset_mock()
    second_result = get_task(
        project_name=project_name,
        task_id="integration_seed_task",
        db_session=db_session,
    )

    # get_segment_ng_state should NOT be called again since ng_state is no longer in seed_id format
    mock_get_segment_ng_state.assert_not_called()

    # Result should be the same as before
    assert second_result["ng_state"] == mock_generated_state
    assert second_result["ng_state_initial"] == mock_generated_state


def test_get_task_process_ng_state_flag_default_true(
    db_session, project_name, existing_task_type, mocker
):
    """Test that get_task processes ng_state by default (process_ng_state=True)"""
    # Create a task with seed_id format in ng_state
    seed_task = Task(
        **{
            "task_id": "seed_task_1",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": {"seed_id": 74732294451380972},
            "ng_state_initial": {"seed_id": 74732294451380972},
            "priority": 1,
            "batch_id": "batch_1",
            "task_type": existing_task_type["task_type"],
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    create_task(project_name=project_name, data=seed_task, db_session=db_session)

    # Mock the _process_ng_state_seed_id function to track if it's called
    mock_process = mocker.patch(
        "zetta_utils.task_management.task._process_ng_state_seed_id",
        autospec=True
    )

    # Call get_task without specifying process_ng_state (should default to True)
    result = get_task(
        project_name=project_name,
        task_id="seed_task_1",
        db_session=db_session,
    )

    # Verify that _process_ng_state_seed_id was called
    mock_process.assert_called_once()
    call_args = mock_process.call_args
    assert call_args[0][1] == project_name  # project_name argument
    assert call_args[0][2].task_id == "seed_task_1"  # task argument

    # Verify task was returned
    assert result["task_id"] == "seed_task_1"


def test_get_task_process_ng_state_flag_explicit_true(
    db_session, project_name, existing_task_type, mocker
):
    """Test that get_task processes ng_state when process_ng_state=True"""
    # Create a task with seed_id format in ng_state
    seed_task = Task(
        **{
            "task_id": "seed_task_2",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": {"seed_id": 74732294451380972},
            "ng_state_initial": {"seed_id": 74732294451380972},
            "priority": 1,
            "batch_id": "batch_1",
            "task_type": existing_task_type["task_type"],
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    create_task(project_name=project_name, data=seed_task, db_session=db_session)

    # Mock the _process_ng_state_seed_id function to track if it's called
    mock_process = mocker.patch(
        "zetta_utils.task_management.task._process_ng_state_seed_id",
        autospec=True
    )

    # Call get_task with process_ng_state=True explicitly
    result = get_task(
        project_name=project_name,
        task_id="seed_task_2",
        process_ng_state=True,
        db_session=db_session,
    )

    # Verify that _process_ng_state_seed_id was called
    mock_process.assert_called_once()
    call_args = mock_process.call_args
    assert call_args[0][1] == project_name  # project_name argument
    assert call_args[0][2].task_id == "seed_task_2"  # task argument

    # Verify task was returned
    assert result["task_id"] == "seed_task_2"


def test_get_task_process_ng_state_flag_false(
    db_session, project_name, existing_task_type, mocker
):
    """Test that get_task skips ng_state processing when process_ng_state=False"""
    # Create a task with seed_id format in ng_state
    seed_task = Task(
        **{
            "task_id": "seed_task_3",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": {"seed_id": 74732294451380972},
            "ng_state_initial": {"seed_id": 74732294451380972},
            "priority": 1,
            "batch_id": "batch_1",
            "task_type": existing_task_type["task_type"],
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    create_task(project_name=project_name, data=seed_task, db_session=db_session)

    # Mock the _process_ng_state_seed_id function to track if it's called
    mock_process = mocker.patch(
        "zetta_utils.task_management.task._process_ng_state_seed_id",
        autospec=True
    )

    # Call get_task with process_ng_state=False
    result = get_task(
        project_name=project_name,
        task_id="seed_task_3",
        process_ng_state=False,
        db_session=db_session,
    )

    # Verify that _process_ng_state_seed_id was NOT called
    mock_process.assert_not_called()

    # Verify task was returned with original ng_state (not processed)
    assert result["task_id"] == "seed_task_3"
    assert result["ng_state"] == {"seed_id": 74732294451380972}
    assert result["ng_state_initial"] == {"seed_id": 74732294451380972}


def test_get_task_process_ng_state_flag_with_regular_ng_state(
    db_session, project_name, existing_task_type, mocker
):
    """Test that process_ng_state flag doesn't affect tasks with regular ng_state"""
    # Create a task with regular (non-seed_id) ng_state
    regular_task = Task(
        **{
            "task_id": "regular_task_1",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": {"url": "http://example.com/regular"},
            "ng_state_initial": {"url": "http://example.com/regular"},
            "priority": 1,
            "batch_id": "batch_1",
            "task_type": existing_task_type["task_type"],
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    create_task(project_name=project_name, data=regular_task, db_session=db_session)

    # Mock the _process_ng_state_seed_id function to track if it's called
    mock_process = mocker.patch(
        "zetta_utils.task_management.task._process_ng_state_seed_id",
        autospec=True
    )

    # Call get_task with process_ng_state=True (should still call the function but no processing)
    result_true = get_task(
        project_name=project_name,
        task_id="regular_task_1",
        process_ng_state=True,
        db_session=db_session,
    )

    # Verify that _process_ng_state_seed_id was called (even though it won't process anything)
    mock_process.assert_called_once()
    # Reset the mock
    mock_process.reset_mock()

    # Call get_task with process_ng_state=False
    result_false = get_task(
        project_name=project_name,
        task_id="regular_task_1",
        process_ng_state=False,
        db_session=db_session,
    )

    # Verify that _process_ng_state_seed_id was NOT called
    mock_process.assert_not_called()

    # Both results should be identical since ng_state doesn't need processing
    assert result_true["task_id"] == "regular_task_1"
    assert result_false["task_id"] == "regular_task_1"
    assert result_true["ng_state"] == result_false["ng_state"]
    assert result_true["ng_state_initial"] == result_false["ng_state_initial"]
