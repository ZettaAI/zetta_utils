# pylint: disable=redefined-outer-name,unused-argument,too-many-lines
import time

import pytest

from zetta_utils.task_management.dependency import create_dependency
from zetta_utils.task_management.exceptions import (
    SubtaskValidationError,
    UserValidationError,
)
from zetta_utils.task_management.subtask import (
    _validate_subtask,
    create_subtask,
    get_max_idle_seconds,
    get_subtask,
    release_subtask,
    start_subtask,
    update_subtask,
)
from zetta_utils.task_management.subtask_type import create_subtask_type
from zetta_utils.task_management.types import (
    Dependency,
    Subtask,
    SubtaskType,
    SubtaskUpdate,
    User,
    UserUpdate,
)
from zetta_utils.task_management.user import create_user, get_user, update_user


@pytest.fixture
def existing_priority_subtasks(clean_db, project_name, existing_subtask_type, db_session):
    subtasks = [
        Subtask(
            **{
                "task_id": "task_1",
                "subtask_id": f"subtask_{i}",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "ng_state": f"http://example.com/{i}",
                "ng_state_initial": f"http://example.com/{i}",
                "priority": i,
                "batch_id": "batch_1",
                "subtask_type": existing_subtask_type["subtask_type"],
                "is_active": True,
                "last_leased_ts": 0.0,
                "completion_status": "",
            }
        )
        for i in range(1, 3)
    ]
    for subtask in subtasks:
        create_subtask(project_name=project_name, data=subtask, db_session=db_session)
    yield subtasks


@pytest.fixture
def sample_dependency() -> Dependency:
    return Dependency(
        **{
            "dependency_id": "dep_1",
            "subtask_id": "subtask_2",
            "dependent_on_subtask_id": "subtask_1",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )


@pytest.fixture
def existing_dependency_chain(
    clean_db, project_name, existing_priority_subtasks, sample_dependency, db_session
):
    create_dependency(project_name=project_name, data=sample_dependency, db_session=db_session)
    update_subtask(
        project_name=project_name,
        subtask_id="subtask_2",
        data=SubtaskUpdate(completion_status="waiting_for_dependencies"),
        db_session=db_session,
    )
    yield sample_dependency


def test_create_subtask_success(project_name, sample_subtask, existing_subtask_type, db_session):
    """Test creating a new subtask"""
    result = create_subtask(project_name=project_name, data=sample_subtask, db_session=db_session)
    assert result == sample_subtask["subtask_id"]

    # Verify the subtask was created by retrieving it
    created_subtask = get_subtask(
        project_name=project_name, subtask_id=sample_subtask["subtask_id"], db_session=db_session
    )
    assert created_subtask["subtask_id"] == sample_subtask["subtask_id"]


def test_create_subtask_nonexistent_type(db_session, project_name):
    """Test creating a subtask with a nonexistent subtask type raises SubtaskValidationError"""
    # Try to create a subtask with a nonexistent subtask type
    subtask_with_invalid_type = Subtask(
        **{
            "task_id": "task_1",
            "subtask_id": "test_subtask",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com",
            "ng_state_initial": "http://example.com",
            "priority": 1,
            "batch_id": "batch_1",
            "subtask_type": "nonexistent_subtask_type",  # This type doesn't exist
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )

    # This should raise a SubtaskValidationError with the specific message
    with pytest.raises(
        SubtaskValidationError, match="Subtask type nonexistent_subtask_type not found"
    ):
        create_subtask(
            project_name=project_name, data=subtask_with_invalid_type, db_session=db_session
        )


def test_get_subtask_success(db_session, existing_subtask, project_name):
    """Test retrieving a subtask that exists"""
    result = get_subtask(project_name=project_name, subtask_id="subtask_1", db_session=db_session)
    assert result["subtask_id"] == existing_subtask["subtask_id"]
    assert result["task_id"] == existing_subtask["task_id"]
    assert result["completion_status"] == existing_subtask["completion_status"]


def test_get_subtask_not_found(db_session, project_name):
    """Test retrieving a subtask that doesn't exist"""
    with pytest.raises(KeyError, match="Subtask nonexistent not found"):
        get_subtask(project_name=project_name, subtask_id="nonexistent", db_session=db_session)


def test_update_subtask_success(db_session, project_name, existing_subtask, existing_subtask_type):
    """Test updating a subtask"""
    update_data = SubtaskUpdate(**{"priority": 5, "assigned_user_id": "user_2"})
    result = update_subtask(
        project_name=project_name,
        subtask_id="subtask_1",
        data=update_data,
        db_session=db_session,
    )
    assert result is True

    # Check that the subtask was updated
    updated_subtask = get_subtask(
        project_name=project_name, subtask_id="subtask_1", db_session=db_session
    )
    assert updated_subtask["priority"] == 5
    assert updated_subtask["assigned_user_id"] == "user_2"


def test_update_subtask_not_found(db_session, project_name):
    """Test updating a subtask that doesn't exist"""
    update_data = SubtaskUpdate(completion_status="in_progress")
    with pytest.raises(KeyError, match="Subtask nonexistent not found"):
        update_subtask(
            project_name=project_name,
            subtask_id="nonexistent",
            data=update_data,
            db_session=db_session,
        )


def test_start_subtask_success(db_session, existing_subtask, existing_user, project_name):
    """Test starting work on a subtask"""
    before_time = time.time()
    result = start_subtask(
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
        db_session=db_session,
    )
    after_time = time.time()

    assert result == "subtask_1"

    # Check that the subtask was updated
    updated_subtask = get_subtask(
        project_name=project_name, subtask_id="subtask_1", db_session=db_session
    )
    assert updated_subtask["completion_status"] == ""
    assert updated_subtask["active_user_id"] == "test_user"
    assert before_time <= updated_subtask["last_leased_ts"] <= after_time

    # Check that the user was updated
    updated_user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert updated_user["active_subtask"] == "subtask_1"


def test_start_subtask_auto_select(db_session, existing_subtasks, existing_user, project_name):
    """Test auto-selecting a subtask to work on"""
    result = start_subtask(
        project_name=project_name, user_id="test_user", subtask_id=None, db_session=db_session
    )

    # Should select the highest priority subtask (subtask_3)
    assert result == "subtask_3"

    # Check that the subtask was updated
    updated_subtask = get_subtask(
        project_name=project_name, subtask_id="subtask_3", db_session=db_session
    )
    assert updated_subtask["completion_status"] == ""
    assert updated_subtask["active_user_id"] == "test_user"

    # Check that the user was updated
    updated_user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert updated_user["active_subtask"] == "subtask_3"


def test_release_subtask_success(db_session, project_name, existing_subtask, existing_user):
    """Test releasing a subtask"""
    # First start work on the subtask
    start_subtask(
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
        db_session=db_session,
    )

    # Now release it with an empty completion status
    result = release_subtask(
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
        db_session=db_session,
    )
    assert result is True

    # Check that the subtask was updated
    updated_subtask = get_subtask(
        project_name=project_name, subtask_id="subtask_1", db_session=db_session
    )
    assert updated_subtask["completion_status"] == ""  # Empty status instead of "todo"
    assert updated_subtask["active_user_id"] == ""

    # Check that the user was updated
    updated_user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert updated_user["active_subtask"] == ""


def test_release_subtask_with_completion(
    db_session, project_name, existing_subtask, existing_user
):
    """Test releasing a subtask with completion"""
    # First start work on the subtask
    start_subtask(
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
        db_session=db_session,
    )

    # Now release it with completion
    result = release_subtask(
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
        completion_status="done",
        db_session=db_session,
    )
    assert result is True

    # Check that the subtask was updated
    updated_subtask = get_subtask(
        project_name=project_name, subtask_id="subtask_1", db_session=db_session
    )
    assert updated_subtask["completion_status"] == "done"
    assert updated_subtask["active_user_id"] == ""
    assert updated_subtask["completed_user_id"] == "test_user"

    # Check that the user was updated
    updated_user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert updated_user["active_subtask"] == ""


def test_update_subtask_missing_dependency_fields(
    db_session, project_name, existing_subtasks, existing_subtask_type
):
    dep1 = Dependency(
        **{
            "dependency_id": "dep_1",
            "subtask_id": "subtask_2",
            "dependent_on_subtask_id": "subtask_1",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )
    create_dependency(project_name=project_name, data=dep1, db_session=db_session)

    # Make subtask_2 inactive initially
    update_subtask(
        project_name=project_name,
        subtask_id="subtask_2",
        data=SubtaskUpdate(is_active=False),
        db_session=db_session,
    )

    # Now complete subtask_1
    update_data = SubtaskUpdate(completion_status="done", completed_user_id="test_user")
    update_subtask(
        project_name=project_name,
        subtask_id="subtask_1",
        data=update_data,
        db_session=db_session,
    )

    # Check if dependent subtask was activated
    subtask2 = get_subtask(
        project_name=project_name, subtask_id="subtask_2", db_session=db_session
    )
    assert subtask2["is_active"] is True
    assert subtask2["completion_status"] == ""


@pytest.mark.parametrize(
    "invalid_data, expected_error",
    [
        ({"subtask_type": "nonexistent_type"}, "Subtask type not found"),
        (
            {"completion_status": "invalid"},
            "Completion status 'invalid' not allowed for this subtask type",
        ),
    ],
)
def test_update_subtask_validation(
    db_session, invalid_data, expected_error, existing_subtask, project_name
):
    with pytest.raises(SubtaskValidationError, match=expected_error):
        update_subtask(
            project_name=project_name,
            subtask_id="subtask_1",
            data=invalid_data,
            db_session=db_session,
        )


def test_inactive_subtask_rejects_completion_status(db_session, project_name, existing_subtask):
    """Test that inactive subtasks reject completion status updates"""
    # First make the subtask inactive
    update_subtask(
        project_name=project_name,
        subtask_id="subtask_1",
        data=SubtaskUpdate(is_active=False),
        db_session=db_session,
    )

    # Try to complete it
    update_data = SubtaskUpdate(completion_status="done", completed_user_id="test_user")
    with pytest.raises(
        SubtaskValidationError, match="Inactive subtask cannot have completion status"
    ):
        update_subtask(
            project_name=project_name,
            subtask_id="subtask_1",
            data=update_data,
            db_session=db_session,
        )

    # Check that completion_status didn't change and completed_user_id is empty
    subtask = get_subtask(project_name=project_name, subtask_id="subtask_1", db_session=db_session)
    assert subtask["completion_status"] == ""  # Should remain empty
    assert subtask["completed_user_id"] == ""


def test_setting_subtask_inactive_clears_completion_status(
    db_session, project_name, existing_subtask
):
    """Test that setting a subtask to inactive clears its completion status"""
    # First set it to active and complete it
    update_subtask(
        project_name=project_name,
        subtask_id="subtask_1",
        data=SubtaskUpdate(is_active=True),
        db_session=db_session,
    )
    update_subtask(
        project_name=project_name,
        subtask_id="subtask_1",
        data=SubtaskUpdate(completion_status="done", completed_user_id="test_user"),
        db_session=db_session,
    )

    # Verify it was completed
    subtask = get_subtask(project_name=project_name, subtask_id="subtask_1", db_session=db_session)
    assert subtask["completion_status"] == "done"
    assert subtask["completed_user_id"] == "test_user"


def test_inactive_subtask_rejects_completed_user(db_session, project_name, existing_subtask):
    """Test that inactive subtasks reject completed_user_id updates"""
    # First make the subtask inactive
    update_subtask(
        project_name=project_name,
        subtask_id="subtask_1",
        data=SubtaskUpdate(is_active=False),
        db_session=db_session,
    )

    # Try to set a completed_user_id without completion status
    update_data = SubtaskUpdate(completed_user_id="test_user")
    with pytest.raises(
        SubtaskValidationError, match="Inactive subtask cannot have completed user"
    ):
        update_subtask(
            project_name=project_name,
            subtask_id="subtask_1",
            data=update_data,
            db_session=db_session,
        )

    # Check that completed_user_id is still empty
    subtask = get_subtask(project_name=project_name, subtask_id="subtask_1", db_session=db_session)
    assert subtask["completed_user_id"] == ""


def test_completed_subtask_requires_completed_user(db_session, project_name, existing_subtask):
    """Test that completed subtasks must have a completed_user_id"""
    # Try to set a completion status without a completed_user_id
    update_data = SubtaskUpdate(completion_status="done")
    with pytest.raises(
        SubtaskValidationError, match="Completed subtask must have completed_user_id"
    ):
        update_subtask(
            project_name=project_name,
            subtask_id="subtask_1",
            data=update_data,
            db_session=db_session,
        )

    # Check that completion_status didn't change
    subtask = get_subtask(project_name=project_name, subtask_id="subtask_1", db_session=db_session)
    assert subtask["completion_status"] == ""  # Should remain empty

    # Now try with both fields set - this should work
    update_data = SubtaskUpdate(completion_status="done", completed_user_id="test_user")
    update_subtask(
        project_name=project_name,
        subtask_id="subtask_1",
        data=update_data,
        db_session=db_session,
    )

    # Verify both fields were updated
    subtask = get_subtask(project_name=project_name, subtask_id="subtask_1", db_session=db_session)
    assert subtask["completion_status"] == "done"
    assert subtask["completed_user_id"] == "test_user"


def test_create_subtask_duplicate_id(db_session, project_name, existing_subtask):
    """Test that creating a subtask with an existing ID raises an error"""
    # Try to create a subtask with the same ID as an existing one
    duplicate_subtask = Subtask(
        **{
            "task_id": "task_1",
            "subtask_id": "subtask_1",  # This ID already exists
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com/new",
            "ng_state_initial": "http://example.com/",
            "priority": 2,
            "batch_id": "batch_2",
            "subtask_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )

    # This should raise a SubtaskValidationError with a specific message
    with pytest.raises(SubtaskValidationError, match="Subtask subtask_1 already exists"):
        create_subtask(project_name=project_name, data=duplicate_subtask, db_session=db_session)

    # Verify the original subtask wasn't modified
    subtask = get_subtask(project_name=project_name, subtask_id="subtask_1", db_session=db_session)
    assert subtask["ng_state"] == "http://example.com"  # Original ng_state, not the new one
    assert subtask["priority"] == 1  # Original priority, not the new one


def test_subtask_type_without_completion_statuses(db_session, project_name, existing_subtask):
    """Test that a subtask with a type that has no completion statuses cannot be completed"""
    # Create a subtask type without completion_statuses
    invalid_type = SubtaskType(subtask_type="no_completion_type", completion_statuses=[])
    create_subtask_type(project_name=project_name, data=invalid_type, db_session=db_session)

    # First update the subtask to use this type
    update_subtask(
        project_name=project_name,
        subtask_id="subtask_1",
        data=SubtaskUpdate(subtask_type="no_completion_type"),
        db_session=db_session,
    )

    # Now try to set a completion status
    update_data = SubtaskUpdate(completion_status="done", completed_user_id="test_user")
    with pytest.raises(
        SubtaskValidationError,
        match="Completion status 'done' not allowed for this subtask type",
    ):
        update_subtask(
            project_name=project_name,
            subtask_id="subtask_1",
            data=update_data,
            db_session=db_session,
        )

    # Check that completion_status didn't change
    subtask = get_subtask(project_name=project_name, subtask_id="subtask_1", db_session=db_session)
    assert subtask["completion_status"] == ""  # Should remain empty
    assert subtask["completed_user_id"] == ""


def test_subtask_type_missing_completion_statuses_key(
    db_session, project_name, existing_subtask, mocker
):
    """Test validation when subtask type dict is missing completion_statuses key entirely"""
    # Create a mock subtask type that doesn't have completion_statuses in its to_dict result
    mock_subtask_type = mocker.Mock()
    mock_subtask_type.to_dict.return_value = {
        "subtask_type": "test_type",
        "description": "Test type"
        # Missing "completion_statuses" key
    }

    # Mock the database query to return our mock subtask type
    mock_execute = mocker.patch.object(db_session, "execute")
    mock_result = mocker.Mock()
    mock_result.scalar_one.return_value = mock_subtask_type
    mock_execute.return_value = mock_result

    # Test subtask data that would trigger the validation
    test_subtask = {
        "subtask_type": "test_type",
        "completion_status": "done",
        "completed_user_id": "test_user",
        "is_active": True,
    }

    # This should raise the specific validation error we're testing
    with pytest.raises(
        SubtaskValidationError,
        match="Subtask type test_type has no valid completion statuses",
    ):
        _validate_subtask(db_session, project_name, test_subtask)


def test_update_nonexistent_subtask(db_session, project_name):
    """Test that updating a nonexistent subtask raises an error"""
    # Try to update a subtask that doesn't exist
    update_data = SubtaskUpdate(priority=5, assigned_user_id="user_2")

    # This should raise a KeyError with a specific message
    with pytest.raises(KeyError, match="Subtask nonexistent_subtask not found"):
        update_subtask(
            project_name=project_name,
            subtask_id="nonexistent_subtask",
            data=update_data,
            db_session=db_session,
        )


def test_start_subtask_nonexistent_user(db_session, project_name, existing_subtask):
    """Test that starting a subtask with a nonexistent user raises an error"""
    # Try to start a subtask with a user that doesn't exist
    nonexistent_user_id = "nonexistent_user"

    # This should raise a UserValidationError with a specific message
    with pytest.raises(UserValidationError, match=f"User {nonexistent_user_id} not found"):
        start_subtask(
            project_name=project_name,
            user_id=nonexistent_user_id,
            subtask_id="subtask_1",
            db_session=db_session,
        )

    # Verify that the subtask wasn't modified
    subtask = get_subtask(project_name=project_name, subtask_id="subtask_1", db_session=db_session)
    assert subtask["active_user_id"] == ""  # Should remain empty

    # Verify no user was created by trying to get the user and expecting an error
    with pytest.raises(KeyError):
        get_user(project_name=project_name, user_id=nonexistent_user_id, db_session=db_session)


def test_start_subtask_user_already_has_active(
    db_session, project_name, existing_subtask, existing_user
):
    """Test that a user with an active subtask cannot start another one"""
    # Create a second subtask
    second_subtask = Subtask(
        **{
            "task_id": "task_1",
            "subtask_id": "subtask_2",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com/second",
            "ng_state_initial": "http://example.com/",
            "priority": 2,
            "batch_id": "batch_1",
            "subtask_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    create_subtask(project_name=project_name, data=second_subtask, db_session=db_session)

    # First, start one subtask
    start_subtask(
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
        db_session=db_session,
    )

    # Now try to start a second subtask with the same user
    with pytest.raises(
        UserValidationError,
        match="User already has an active subtask subtask_1 which is different "
        "from requested subtask subtask_2",
    ):
        start_subtask(
            project_name=project_name,
            user_id="test_user",
            subtask_id="subtask_2",
            db_session=db_session,
        )

    # Verify the first subtask is still active for the user
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_subtask"] == "subtask_1"

    # Verify the second subtask wasn't started
    subtask2 = get_subtask(
        project_name=project_name, subtask_id="subtask_2", db_session=db_session
    )
    assert subtask2["active_user_id"] == ""  # Should remain empty


def test_start_nonexistent_subtask(db_session, project_name, existing_user):
    """Test that starting a nonexistent subtask raises an error"""
    # Try to start a subtask that doesn't exist
    nonexistent_subtask_id = "nonexistent_subtask"

    # This should raise a SubtaskValidationError with a specific message
    with pytest.raises(
        SubtaskValidationError, match=f"Subtask {nonexistent_subtask_id} not found"
    ):
        start_subtask(
            project_name=project_name,
            user_id="test_user",
            subtask_id=nonexistent_subtask_id,
            db_session=db_session,
        )

    # Verify the user's active_subtask wasn't modified
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_subtask"] == ""

    # Verify that no subtask was created with this ID by trying to get it and expecting an error
    with pytest.raises(KeyError):
        get_subtask(
            project_name=project_name,
            subtask_id=nonexistent_subtask_id,
            db_session=db_session,
        )


def test_start_subtask_takeover_idle(db_session, project_name, existing_subtask, existing_user):
    """Test taking over an idle subtask from another user"""
    second_user = User(
        **{
            "user_id": "user_2",
            "hourly_rate": 45.0,
            "active_subtask": "",
            "qualified_subtask_types": ["segmentation_proofread"],
        }
    )
    create_user(project_name=project_name, data=second_user, db_session=db_session)

    # First, have user_1 start the subtask
    start_subtask(
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
        db_session=db_session,
    )

    # Verify user_1 has the subtask
    user1 = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user1["active_subtask"] == "subtask_1"

    # Manually update the last_leased_ts to be older than max idle seconds
    old_time = time.time() - (get_max_idle_seconds() + 10)
    update_subtask(
        project_name=project_name,
        subtask_id="subtask_1",
        data=SubtaskUpdate(last_leased_ts=old_time),
        db_session=db_session,
    )

    # Now have user_2 take over the subtask
    start_subtask(
        project_name=project_name,
        user_id="user_2",
        subtask_id="subtask_1",
        db_session=db_session,
    )

    # Verify user_2 now has the subtask
    user2 = get_user(project_name=project_name, user_id="user_2", db_session=db_session)
    assert user2["active_subtask"] == "subtask_1"

    # Verify user_1 no longer has the subtask
    user1 = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user1["active_subtask"] == ""

    # Verify the subtask is now assigned to user_2
    subtask = get_subtask(project_name=project_name, subtask_id="subtask_1", db_session=db_session)
    assert subtask["active_user_id"] == "user_2"
    assert subtask["last_leased_ts"] > old_time


def test_start_subtask_already_active(db_session, project_name, existing_subtask, existing_user):
    """Test that starting an already active subtask raises an error"""
    # Create a second user
    second_user = User(
        **{
            "user_id": "user_2",
            "hourly_rate": 45.0,
            "active_subtask": "",
            "qualified_subtask_types": ["segmentation_proofread"],
        }
    )
    create_user(project_name=project_name, data=second_user, db_session=db_session)

    # First, have user_1 start the subtask
    start_subtask(
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
        db_session=db_session,
    )

    # Verify user_1 has the subtask
    user1 = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user1["active_subtask"] == "subtask_1"

    # Now have user_2 try to take over the subtask (which is still active)
    with pytest.raises(
        SubtaskValidationError, match="Subtask is no longer available for takeover"
    ):
        start_subtask(
            project_name=project_name,
            user_id="user_2",
            subtask_id="subtask_1",
            db_session=db_session,
        )

    # Verify user_1 still has the subtask
    user1 = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user1["active_subtask"] == "subtask_1"

    # Verify user_2 still has no active subtask
    user2 = get_user(project_name=project_name, user_id="user_2", db_session=db_session)
    assert user2["active_subtask"] == ""

    # Verify the subtask is still assigned to user_1
    subtask = get_subtask(project_name=project_name, subtask_id="subtask_1", db_session=db_session)
    assert subtask["active_user_id"] == "test_user"


def test_start_subtask_requires_qualification(
    db_session, project_name, existing_user, existing_subtask, existing_subtask_type
):
    """Test that a user can't start a subtask if they're not qualified for it"""
    # Update user to have empty qualifications list
    update_user(
        project_name=project_name,
        user_id="test_user",
        data=UserUpdate(
            hourly_rate=50.0,
            active_subtask="",
            qualified_subtask_types=[],  # Empty list means no qualifications
        ),
        db_session=db_session,
    )

    # Try to start subtask - this should raise an error
    with pytest.raises(UserValidationError, match="User not qualified for this subtask type"):
        start_subtask(
            project_name=project_name,
            user_id="test_user",
            subtask_id="subtask_1",
            db_session=db_session,
        )

    # Verify the user's active_subtask wasn't modified
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_subtask"] == ""

    # Verify the subtask wasn't modified
    subtask = get_subtask(project_name=project_name, subtask_id="subtask_1", db_session=db_session)
    assert subtask["active_user_id"] == ""

    # Now update the user to have the qualification and try again
    update_user(
        project_name=project_name,
        user_id="test_user",
        data=UserUpdate(qualified_subtask_types=["segmentation_proofread"]),
        db_session=db_session,
    )

    # This should now succeed
    start_subtask(
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
        db_session=db_session,
    )

    # Verify the user now has the subtask
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_subtask"] == "subtask_1"

    # Verify the subtask is now assigned to the user
    subtask = get_subtask(project_name=project_name, subtask_id="subtask_1", db_session=db_session)
    assert subtask["active_user_id"] == "test_user"


def test_start_subtask_user_not_qualified_for_specific_type(
    db_session, project_name, existing_subtask_type
):
    """Test that a user with qualifications can't start a subtask they're not qualified for"""
    # Create a user qualified for one type but not another
    qualified_user = User(
        user_id="partially_qualified_user",
        hourly_rate=50.0,
        active_subtask="",
        qualified_subtask_types=["different_type"],  # Qualified for different type only
    )
    create_user(project_name=project_name, data=qualified_user, db_session=db_session)

    # Create a subtask with the existing subtask type (which user is not qualified for)
    test_subtask = Subtask(
        task_id="task_1",
        subtask_id="unqualified_subtask",
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state="http://example.com/unqualified",
        ng_state_initial="http://example.com/unqualified",
        priority=1,
        batch_id="batch_1",
        subtask_type=existing_subtask_type["subtask_type"],  # User not qualified for this type
        is_active=True,
        last_leased_ts=0.0,
        completion_status="",
    )
    create_subtask(project_name=project_name, data=test_subtask, db_session=db_session)

    # Try to start the subtask - should fail because user not qualified for this specific type
    with pytest.raises(UserValidationError, match="User not qualified for this subtask type"):
        start_subtask(
            project_name=project_name,
            user_id="partially_qualified_user",
            subtask_id="unqualified_subtask",
            db_session=db_session,
        )

    # Verify the user's active_subtask wasn't modified
    user = get_user(
        project_name=project_name,
        user_id="partially_qualified_user",
        db_session=db_session,
    )
    assert user["active_subtask"] == ""

    # Verify the subtask wasn't modified
    subtask = get_subtask(
        project_name=project_name,
        subtask_id="unqualified_subtask",
        db_session=db_session,
    )
    assert subtask["active_user_id"] == ""


def test_release_subtask_nonexistent_subtask(db_session, project_name, existing_user):
    """Test that releasing a nonexistent subtask raises an error"""
    # Update user to have nonexistent active subtask
    update_user(
        project_name=project_name,
        user_id="test_user",
        data=UserUpdate(active_subtask="nonexistent_subtask"),
        db_session=db_session,
    )

    # Verify the user has been updated
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_subtask"] == "nonexistent_subtask"

    # Now try to release the subtask
    with pytest.raises(SubtaskValidationError, match="Subtask nonexistent_subtask not found"):
        release_subtask(
            project_name=project_name,
            user_id="test_user",
            subtask_id="nonexistent_subtask",
            db_session=db_session,
        )

    # Verify the user's active_subtask wasn't modified
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_subtask"] == "nonexistent_subtask"


def test_auto_select_subtask_no_qualified_types(db_session, project_name, existing_subtask):
    """Test that auto-selecting a subtask returns None when user has no qualified types"""
    # Create a user with no qualified subtask types
    unqualified_user = User(
        **{
            "user_id": "unqualified_user",
            "hourly_rate": 45.0,
            "active_subtask": "",
            "qualified_subtask_types": [],  # Empty list means no qualifications
        }
    )
    create_user(project_name=project_name, data=unqualified_user, db_session=db_session)

    # Try to auto-select a subtask (by not specifying a subtask_id)
    result = start_subtask(
        project_name=project_name,
        user_id="unqualified_user",
        subtask_id=None,
        db_session=db_session,
    )

    # Verify that no subtask was selected
    assert result is None

    # Verify the user's active_subtask wasn't modified
    user = get_user(project_name=project_name, user_id="unqualified_user", db_session=db_session)
    assert user["active_subtask"] == ""

    # Now update the user to have a qualification and try again
    update_user(
        project_name=project_name,
        user_id="unqualified_user",
        data=UserUpdate(qualified_subtask_types=["segmentation_proofread"]),
        db_session=db_session,
    )

    # This should now succeed
    result = start_subtask(
        project_name=project_name,
        user_id="unqualified_user",
        subtask_id=None,
        db_session=db_session,
    )

    # Verify a subtask was selected
    assert result is not None

    # Verify the user now has the subtask
    user = get_user(project_name=project_name, user_id="unqualified_user", db_session=db_session)
    assert user["active_subtask"] == result

    # Verify the subtask is now assigned to the user
    subtask = get_subtask(project_name=project_name, subtask_id=result, db_session=db_session)
    assert subtask["active_user_id"] == "unqualified_user"


def test_auto_select_subtask_prioritizes_assigned_to_user(
    db_session, project_name, existing_user, existing_subtask_type, task_factory, subtask_factory
):
    task_factory("task_1")

    subtask_factory("task_1", "assigned_subtask", assigned_user_id="test_user", priority=1)

    subtask_factory(
        "task_1", "unassigned_subtask", assigned_user_id="", priority=10  # Higher priority
    )

    # Auto-select a subtask (by not specifying a subtask_id)
    result = start_subtask(
        project_name=project_name,
        user_id="test_user",
        subtask_id=None,
        db_session=db_session,
    )

    # Verify that the assigned subtask was selected, even though it has lower priority
    assert result == "assigned_subtask"

    # Verify the user now has the assigned subtask
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_subtask"] == "assigned_subtask"

    # Verify the subtask is now assigned to the user
    subtask = get_subtask(
        project_name=project_name, subtask_id="assigned_subtask", db_session=db_session
    )
    assert subtask["active_user_id"] == "test_user"

    # Verify the unassigned subtask wasn't modified
    unassigned = get_subtask(
        project_name=project_name, subtask_id="unassigned_subtask", db_session=db_session
    )
    assert unassigned["active_user_id"] == ""


def test_release_subtask_no_active_subtask(db_session, project_name, existing_user):
    """Test that releasing a subtask fails when the user doesn't have an active subtask"""
    # Verify the user doesn't have an active subtask
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_subtask"] == ""

    # Try to release a subtask
    with pytest.raises(UserValidationError, match="User does not have an active subtask"):
        release_subtask(
            project_name=project_name,
            user_id="test_user",
            subtask_id="subtask_1",
            db_session=db_session,
        )

    # Try to release with a completion status
    with pytest.raises(UserValidationError, match="User does not have an active subtask"):
        release_subtask(
            project_name=project_name,
            user_id="test_user",
            subtask_id="subtask_1",
            completion_status="done",
            db_session=db_session,
        )

    # Verify the user's active_subtask is still empty
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_subtask"] == ""


def test_release_subtask_with_dependencies(
    db_session, project_name, existing_subtask_type, task_factory, subtask_factory
):
    """Test releasing a subtask with dependencies updates the dependent subtasks"""
    # Create a user
    user_data = User(
        user_id="dep_test_user",
        hourly_rate=50.0,
        active_subtask="",
        qualified_subtask_types=[existing_subtask_type["subtask_type"]],
    )
    create_user(project_name=project_name, data=user_data, db_session=db_session)

    # Create task and subtasks using factory fixtures
    task_factory("task_dep")
    subtask_factory("task_dep", "subtask_dep_1", assigned_user_id="dep_test_user", is_active=True)
    subtask_factory(
        "task_dep",
        "subtask_dep_2",
        is_active=False,  # Initially inactive until dependency is satisfied
    )

    # Create a dependency between them
    dependency_data = Dependency(
        dependency_id="dep_test_1",
        subtask_id="subtask_dep_2",  # This subtask depends on subtask_dep_1
        dependent_on_subtask_id="subtask_dep_1",
        required_completion_status="done",
        is_satisfied=False,
    )
    create_dependency(project_name=project_name, data=dependency_data, db_session=db_session)

    # Start work on subtask1
    start_subtask(
        project_name=project_name,
        user_id="dep_test_user",
        subtask_id="subtask_dep_1",
        db_session=db_session,
    )

    # Verify subtask2 is still inactive
    subtask2_before = get_subtask(
        project_name=project_name, subtask_id="subtask_dep_2", db_session=db_session
    )
    assert subtask2_before["is_active"] is False

    # Release subtask1 with completion status "done"
    result = release_subtask(
        project_name=project_name,
        user_id="dep_test_user",
        subtask_id="subtask_dep_1",
        completion_status="done",
        db_session=db_session,
    )
    assert result is True

    # Verify subtask1 was updated correctly
    subtask1_after = get_subtask(
        project_name=project_name, subtask_id="subtask_dep_1", db_session=db_session
    )
    assert subtask1_after["completion_status"] == "done"
    assert subtask1_after["active_user_id"] == ""
    assert subtask1_after["completed_user_id"] == "dep_test_user"


def test_start_subtask_returns_current_active_when_auto_selecting(
    db_session, project_name, existing_subtask, existing_user
):
    # First, start a subtask for the user
    start_subtask(
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
        db_session=db_session,
    )

    # Verify the user has an active subtask
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_subtask"] == "subtask_1"

    # Now call start_subtask with subtask_id=None (auto-select mode)
    # This should return the user's current active subtask
    result = start_subtask(
        project_name=project_name,
        user_id="test_user",
        subtask_id=None,  # Auto-select mode
        db_session=db_session,
    )

    # Should return the same subtask the user already has active
    assert result == "subtask_1"

    # User should still have the same active subtask
    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_subtask"] == "subtask_1"


def test_release_subtask_mismatched_id(db_session, project_name, existing_subtask, existing_user):
    """Test that releasing a subtask with wrong ID raises validation error"""
    # Create a second subtask
    subtask2 = Subtask(
        **{
            "task_id": "task_1",
            "subtask_id": "subtask_2",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com/2",
            "ng_state_initial": "http://example.com/2",
            "priority": 2,
            "batch_id": "batch_1",
            "subtask_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    create_subtask(project_name=project_name, data=subtask2, db_session=db_session)

    # Start subtask_1 for the user
    start_subtask(
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
        db_session=db_session,
    )

    # Try to release subtask_2 (which is not the user's active subtask)
    with pytest.raises(
        UserValidationError,
        match="Subtask ID does not match user's active subtask",
    ):
        release_subtask(
            project_name=project_name,
            user_id="test_user",
            subtask_id="subtask_2",  # Wrong subtask ID
            db_session=db_session,
        )
