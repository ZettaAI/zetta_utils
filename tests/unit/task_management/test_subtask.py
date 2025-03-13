# pylint: disable=redefined-outer-name,unused-argument,too-many-lines
import time
from unittest.mock import patch

import pytest
from google.cloud import firestore

from zetta_utils.task_management.dependency import create_dependency
from zetta_utils.task_management.exceptions import (
    SubtaskValidationError,
    UserValidationError,
)
from zetta_utils.task_management.subtask import (
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
    Task,
    User,
    UserUpdate,
)
from zetta_utils.task_management.user import create_user, get_user, update_user


@pytest.fixture
def project_name_subtask() -> str:
    return "test_project_subtask"


@pytest.fixture(autouse=True, name="clean_collections")
def clean_collections_fixture(firestore_emulator, project_name_subtask):
    client = firestore.Client()
    collections = [
        f"{project_name_subtask}_users",
        f"{project_name_subtask}_subtasks",
        f"{project_name_subtask}_dependencies",
        f"{project_name_subtask}_tasks",
        "subtask_types",
    ]
    for coll in collections:
        for doc in client.collection(coll).list_documents():
            doc.delete()
    yield
    for coll in collections:
        for doc in client.collection(coll).list_documents():
            doc.delete()


@pytest.fixture
def sample_user() -> User:
    return User(
        **{
            "user_id": "user_1",
            "hourly_rate": 50.0,
            "active_subtask": "",
            "qualified_subtask_types": ["segmentation_proofread", "segmentation_verify"],
        }
    )


@pytest.fixture
def existing_user(firestore_emulator, sample_user, project_name_subtask):
    create_user(project_name_subtask, sample_user)
    yield sample_user


@pytest.fixture
def sample_subtask_type() -> SubtaskType:
    return SubtaskType(
        **{"subtask_type": "segmentation_proofread", "completion_statuses": ["done", "need_help"]}
    )


@pytest.fixture
def existing_subtask_type(sample_subtask_type):
    client = firestore.Client()
    doc_ref = client.collection("subtask_types").document(sample_subtask_type["subtask_type"])

    if doc_ref.get().exists:
        doc_ref.delete()

    create_subtask_type(sample_subtask_type)

    yield sample_subtask_type


@pytest.fixture
def sample_subtask() -> Subtask:
    return Subtask(
        **{
            "task_id": "task_1",
            "subtask_id": "subtask_1",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com",
            "priority": 1,
            "batch_id": "batch_1",
            "subtask_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )


@pytest.fixture
def existing_subtask(
    firestore_emulator, project_name_subtask, sample_subtask, existing_subtask_type, existing_task
):
    create_subtask(project_name_subtask, sample_subtask)
    yield sample_subtask


@pytest.fixture
def sample_subtasks(sample_subtask_type) -> list[Subtask]:
    return [
        Subtask(
            **{
                "task_id": "task_1",
                "subtask_id": f"subtask_{i}",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "ng_state": f"http://example.com/{i}",
                "priority": i,
                "batch_id": "batch_1",
                "subtask_type": sample_subtask_type["subtask_type"],
                "is_active": True,
                "last_leased_ts": 0.0,
                "completion_status": "",
            }
        )
        for i in range(1, 4)
    ]


@pytest.fixture
def existing_subtasks(
    firestore_emulator, project_name_subtask, sample_subtasks, existing_subtask_type
):
    for subtask in sample_subtasks:
        create_subtask(project_name_subtask, subtask)
    yield sample_subtasks


@pytest.fixture
def existing_priority_subtasks(firestore_emulator, project_name_subtask, existing_subtask_type):
    subtasks = [
        Subtask(
            **{
                "task_id": "task_1",
                "subtask_id": f"subtask_{i}",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "ng_state": f"http://example.com/{i}",
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
        create_subtask(project_name_subtask, subtask)
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
    firestore_emulator, project_name_subtask, existing_priority_subtasks, sample_dependency
):
    create_dependency(project_name_subtask, sample_dependency)
    update_subtask(
        project_name_subtask,
        "subtask_2",
        {
            "completion_status": "waiting_for_dependencies",
        },
    )
    yield sample_dependency


@pytest.fixture
def sample_task() -> Task:
    return Task(
        **{
            "task_id": "task_1",
            "batch_id": "batch_1",
            "status": "ingested",
            "task_type": "segmentation",
            "ng_state": "http://example.com/task_1",
        }
    )


@pytest.fixture
def existing_task(firestore_emulator, project_name_subtask, sample_task):
    client = firestore.Client()
    tasks_collection = client.collection(f"{project_name_subtask}_tasks")
    tasks_collection.document(sample_task["task_id"]).set(sample_task)
    yield sample_task


def test_create_subtask_success(project_name_subtask, sample_subtask, existing_subtask_type):
    """Test creating a new subtask"""
    result = create_subtask(project_name_subtask, sample_subtask)
    assert result == sample_subtask["subtask_id"]

    client = firestore.Client()
    doc = (
        client.collection(f"{project_name_subtask}_subtasks")
        .document(sample_subtask["subtask_id"])
        .get()
    )
    assert doc.exists


def test_get_subtask_success(existing_subtask, project_name_subtask):
    """Test retrieving a subtask that exists"""
    result = get_subtask(project_name_subtask, "subtask_1")
    assert result["subtask_id"] == existing_subtask["subtask_id"]
    assert result["task_id"] == existing_subtask["task_id"]
    assert result["completion_status"] == existing_subtask["completion_status"]


def test_get_subtask_not_found(project_name_subtask):
    """Test retrieving a subtask that doesn't exist"""
    with pytest.raises(KeyError, match="Subtask nonexistent not found"):
        get_subtask(project_name_subtask, "nonexistent")


def test_update_subtask_success(project_name_subtask, existing_subtask, existing_subtask_type):
    """Test updating a subtask"""
    update_data = SubtaskUpdate(**{"priority": 5, "assigned_user_id": "user_2"})
    result = update_subtask(project_name_subtask, "subtask_1", update_data)
    assert result is True

    # Check that the subtask was updated in Firestore
    updated_subtask = get_subtask(project_name_subtask, "subtask_1")
    assert updated_subtask["priority"] == 5
    assert updated_subtask["assigned_user_id"] == "user_2"


def test_update_subtask_not_found(project_name_subtask):
    """Test updating a subtask that doesn't exist"""
    update_data = SubtaskUpdate(completion_status="in_progress")
    with pytest.raises(KeyError, match="Subtask nonexistent not found"):
        update_subtask(project_name_subtask, "nonexistent", update_data)


def test_start_subtask_success(existing_subtask, existing_user, project_name_subtask):
    """Test starting work on a subtask"""
    before_time = time.time()
    result = start_subtask(project_name_subtask, "user_1", "subtask_1")
    after_time = time.time()

    assert result == "subtask_1"

    # Check that the subtask was updated in Firestore
    updated_subtask = get_subtask(project_name_subtask, "subtask_1")
    assert updated_subtask["completion_status"] == ""
    assert updated_subtask["active_user_id"] == "user_1"
    assert before_time <= updated_subtask["last_leased_ts"] <= after_time

    # Check that the user was updated in Firestore
    updated_user = get_user(project_name_subtask, "user_1")
    assert updated_user["active_subtask"] == "subtask_1"


def test_start_subtask_auto_select(existing_subtasks, existing_user, project_name_subtask):
    """Test auto-selecting a subtask to work on"""
    result = start_subtask(project_name_subtask, "user_1", None)

    # Should select the highest priority subtask (subtask_3)
    assert result == "subtask_3"

    # Check that the subtask was updated in Firestore
    updated_subtask = get_subtask(project_name_subtask, "subtask_3")
    assert updated_subtask["completion_status"] == ""
    assert updated_subtask["active_user_id"] == "user_1"

    # Check that the user was updated in Firestore
    updated_user = get_user(project_name_subtask, "user_1")
    assert updated_user["active_subtask"] == "subtask_3"


def test_release_subtask_success(project_name_subtask, existing_subtask, existing_user):
    """Test releasing a subtask"""
    # First start work on the subtask
    start_subtask(project_name_subtask, "user_1", "subtask_1")

    # Now release it with an empty completion status
    result = release_subtask(project_name_subtask, "user_1", "subtask_1")
    assert result is True

    # Check that the subtask was updated in Firestore
    updated_subtask = get_subtask(project_name_subtask, "subtask_1")
    assert updated_subtask["completion_status"] == ""  # Empty status instead of "todo"
    assert updated_subtask["active_user_id"] == ""

    # Check that the user was updated in Firestore
    updated_user = get_user(project_name_subtask, "user_1")
    assert updated_user["active_subtask"] == ""


def test_release_subtask_with_completion(project_name_subtask, existing_subtask, existing_user):
    """Test releasing a subtask with completion"""
    # First start work on the subtask
    start_subtask(project_name_subtask, "user_1", "subtask_1")

    # Now release it with completion
    result = release_subtask(project_name_subtask, "user_1", "subtask_1", "done")
    assert result is True

    # Check that the subtask was updated in Firestore
    updated_subtask = get_subtask(project_name_subtask, "subtask_1")
    assert updated_subtask["completion_status"] == "done"
    assert updated_subtask["active_user_id"] == ""
    assert updated_subtask["completed_user_id"] == "user_1"

    # Check that the user was updated in Firestore
    updated_user = get_user(project_name_subtask, "user_1")
    assert updated_user["active_subtask"] == ""


def test_update_subtask_missing_dependency_fields(
    project_name_subtask, existing_subtasks, existing_subtask_type
):
    client = firestore.Client()

    dep1 = Dependency(
        **{
            "dependency_id": "dep_1",
            "subtask_id": "subtask_2",
            "dependent_on_subtask_id": "subtask_1",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )
    deps_coll = client.collection(f"{project_name_subtask}_dependencies")
    deps_coll.document("dep_1").set(dep1)

    # Make subtask_2 inactive initially
    update_subtask(project_name_subtask, "subtask_2", SubtaskUpdate(is_active=False))

    # Now complete subtask_1
    update_data = SubtaskUpdate(completion_status="done", completed_user_id="test_user")
    update_subtask(project_name_subtask, "subtask_1", update_data)

    # Check if dependent subtask was activated
    subtask2 = get_subtask(project_name_subtask, "subtask_2")
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
    invalid_data, expected_error, existing_subtask, project_name_subtask
):
    with pytest.raises(SubtaskValidationError, match=expected_error):
        update_subtask(project_name_subtask, "subtask_1", invalid_data)


def test_inactive_subtask_rejects_completion_status(project_name_subtask, existing_subtask):
    """Test that inactive subtasks reject completion status updates"""
    # First make the subtask inactive
    update_subtask(project_name_subtask, "subtask_1", SubtaskUpdate(is_active=False))

    # Try to complete it
    update_data = SubtaskUpdate(completion_status="done", completed_user_id="test_user")
    with pytest.raises(
        SubtaskValidationError, match="Inactive subtask cannot have completion status"
    ):
        update_subtask(project_name_subtask, "subtask_1", update_data)

    # Check that completion_status didn't change and completed_user_id is empty
    subtask = get_subtask(project_name_subtask, "subtask_1")
    assert subtask["completion_status"] == ""  # Should remain empty
    assert subtask["completed_user_id"] == ""


def test_setting_subtask_inactive_clears_completion_status(project_name_subtask, existing_subtask):
    """Test that setting a subtask to inactive clears its completion status"""
    # First set it to active and complete it
    update_subtask(project_name_subtask, "subtask_1", SubtaskUpdate(is_active=True))
    update_subtask(
        project_name_subtask,
        "subtask_1",
        SubtaskUpdate(completion_status="done", completed_user_id="test_user"),
    )

    # Verify it was completed
    subtask = get_subtask(project_name_subtask, "subtask_1")
    assert subtask["completion_status"] == "done"
    assert subtask["completed_user_id"] == "test_user"


def test_inactive_subtask_rejects_completed_user(project_name_subtask, existing_subtask):
    """Test that inactive subtasks reject completed_user_id updates"""
    # First make the subtask inactive
    update_subtask(project_name_subtask, "subtask_1", SubtaskUpdate(is_active=False))

    # Try to set a completed_user_id without completion status
    update_data = SubtaskUpdate(completed_user_id="test_user")
    with pytest.raises(
        SubtaskValidationError, match="Inactive subtask cannot have completed user"
    ):
        update_subtask(project_name_subtask, "subtask_1", update_data)

    # Check that completed_user_id is still empty
    subtask = get_subtask(project_name_subtask, "subtask_1")
    assert subtask["completed_user_id"] == ""


def test_completed_subtask_requires_completed_user(project_name_subtask, existing_subtask):
    """Test that completed subtasks must have a completed_user_id"""
    # Try to set a completion status without a completed_user_id
    update_data = SubtaskUpdate(completion_status="done")
    with pytest.raises(
        SubtaskValidationError, match="Completed subtask must have completed_user_id"
    ):
        update_subtask(project_name_subtask, "subtask_1", update_data)

    # Check that completion_status didn't change
    subtask = get_subtask(project_name_subtask, "subtask_1")
    assert subtask["completion_status"] == ""  # Should remain empty

    # Now try with both fields set - this should work
    update_data = SubtaskUpdate(completion_status="done", completed_user_id="test_user")
    update_subtask(project_name_subtask, "subtask_1", update_data)

    # Verify both fields were updated
    subtask = get_subtask(project_name_subtask, "subtask_1")
    assert subtask["completion_status"] == "done"
    assert subtask["completed_user_id"] == "test_user"


def test_create_subtask_duplicate_id(project_name_subtask, existing_subtask):
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
        create_subtask(project_name_subtask, duplicate_subtask)

    # Verify the original subtask wasn't modified
    subtask = get_subtask(project_name_subtask, "subtask_1")
    assert subtask["ng_state"] == "http://example.com"  # Original ng_state, not the new one
    assert subtask["priority"] == 1  # Original priority, not the new one


def test_subtask_type_without_completion_statuses(project_name_subtask, existing_subtask):
    """Test that a subtask with a type that has no completion statuses cannot be completed"""
    # Create a subtask type without completion_statuses
    client = firestore.Client()
    invalid_type = {"subtask_type": "no_completion_type"}
    client.collection("subtask_types").document("no_completion_type").set(invalid_type)

    # First update the subtask to use this type
    update_subtask(
        project_name_subtask, "subtask_1", SubtaskUpdate(subtask_type="no_completion_type")
    )

    # Now try to set a completion status
    update_data = SubtaskUpdate(completion_status="done", completed_user_id="test_user")
    with pytest.raises(
        SubtaskValidationError,
        match="Subtask type no_completion_type has no valid completion statuses",
    ):
        update_subtask(project_name_subtask, "subtask_1", update_data)

    # Check that completion_status didn't change
    subtask = get_subtask(project_name_subtask, "subtask_1")
    assert subtask["completion_status"] == ""  # Should remain empty
    assert subtask["completed_user_id"] == ""

    # Clean up the test subtask type
    client.collection("subtask_types").document("no_completion_type").delete()


def test_update_nonexistent_subtask(project_name_subtask):
    """Test that updating a nonexistent subtask raises an error"""
    # Try to update a subtask that doesn't exist
    update_data = SubtaskUpdate(priority=5, assigned_user_id="user_2")

    # This should raise a KeyError with a specific message
    with pytest.raises(KeyError, match="Subtask nonexistent_subtask not found"):
        update_subtask(project_name_subtask, "nonexistent_subtask", update_data)

    # Verify that no subtask was created with this ID
    client = firestore.Client()
    doc_ref = client.collection(f"{project_name_subtask}_subtasks").document("nonexistent_subtask")
    assert not doc_ref.get().exists


def test_start_subtask_nonexistent_user(project_name_subtask, existing_subtask):
    """Test that starting a subtask with a nonexistent user raises an error"""
    # Try to start a subtask with a user that doesn't exist
    nonexistent_user_id = "nonexistent_user"

    # This should raise a UserValidationError with a specific message
    with pytest.raises(UserValidationError, match=f"User {nonexistent_user_id} not found"):
        start_subtask(project_name_subtask, nonexistent_user_id, "subtask_1")

    # Verify that the subtask wasn't modified
    subtask = get_subtask(project_name_subtask, "subtask_1")
    assert subtask["active_user_id"] == ""  # Should remain empty

    # Verify no user was created
    client = firestore.Client()
    user_ref = client.collection(f"{project_name_subtask}_users").document(nonexistent_user_id)
    assert not user_ref.get().exists


def test_start_subtask_user_already_has_active(
    project_name_subtask, existing_subtask, existing_user
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
            "priority": 2,
            "batch_id": "batch_1",
            "subtask_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    create_subtask(project_name_subtask, second_subtask)

    # First, start one subtask
    start_subtask(project_name_subtask, "user_1", "subtask_1")

    # Verify the user has an active subtask
    user = get_user(project_name_subtask, "user_1")
    assert user["active_subtask"] == "subtask_1"

    # Now try to start another subtask
    with pytest.raises(UserValidationError, match="User already has an active subtask"):
        start_subtask(project_name_subtask, "user_1", "subtask_2")

    # Verify the user's active subtask hasn't changed
    user = get_user(project_name_subtask, "user_1")
    assert user["active_subtask"] == "subtask_1"

    # Verify the second subtask wasn't modified
    subtask2 = get_subtask(project_name_subtask, "subtask_2")
    assert subtask2["active_user_id"] == ""

    start_subtask(project_name_subtask, "user_1", "subtask_1")
    start_subtask(project_name_subtask, "user_1")
    user = get_user(project_name_subtask, "user_1")
    assert user["active_subtask"] == "subtask_1"


def test_start_nonexistent_subtask(project_name_subtask, existing_user):
    """Test that starting a nonexistent subtask raises an error"""
    # Try to start a subtask that doesn't exist
    nonexistent_subtask_id = "nonexistent_subtask"

    # This should raise a SubtaskValidationError with a specific message
    with pytest.raises(
        SubtaskValidationError, match=f"Subtask {nonexistent_subtask_id} not found"
    ):
        start_subtask(project_name_subtask, "user_1", nonexistent_subtask_id)

    # Verify the user's active_subtask wasn't modified
    user = get_user(project_name_subtask, "user_1")
    assert user["active_subtask"] == ""

    # Verify that no subtask was created with this ID
    client = firestore.Client()
    doc_ref = client.collection(f"{project_name_subtask}_subtasks").document(
        nonexistent_subtask_id
    )
    assert not doc_ref.get().exists

    assert not doc_ref.get().exists


def test_start_subtask_takeover_idle(project_name_subtask, existing_subtask, existing_user):
    """Test taking over an idle subtask from another user"""
    client = firestore.Client()

    # Create a second user
    second_user = User(
        **{
            "user_id": "user_2",
            "hourly_rate": 45.0,
            "active_subtask": "",
            "qualified_subtask_types": ["segmentation_proofread"],
        }
    )
    create_user(project_name_subtask, second_user)

    # First, have user_1 start the subtask
    start_subtask(project_name_subtask, "user_1", "subtask_1")

    # Verify user_1 has the subtask
    user1: User = get_user(project_name_subtask, "user_1")
    assert user1["active_subtask"] == "subtask_1"

    # Manually update the last_leased_ts to be older than max idle seconds
    old_time = time.time() - (get_max_idle_seconds() + 10)
    subtask_ref = client.collection(f"{project_name_subtask}_subtasks").document("subtask_1")
    subtask_ref.update({"last_leased_ts": old_time})

    # Now have user_2 take over the subtask
    start_subtask(project_name_subtask, "user_2", "subtask_1")

    # Verify user_2 now has the subtask
    user2: User = get_user(project_name_subtask, "user_2")
    assert user2["active_subtask"] == "subtask_1"

    # Verify user_1 no longer has the subtask
    user1 = get_user(project_name_subtask, "user_1")
    assert user1["active_subtask"] == ""

    # Verify the subtask is now assigned to user_2
    subtask: Subtask = get_subtask(project_name_subtask, "subtask_1")
    assert subtask["active_user_id"] == "user_2"
    assert subtask["last_leased_ts"] > old_time


def test_start_subtask_already_active(project_name_subtask, existing_subtask, existing_user):
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
    create_user(project_name_subtask, second_user)

    # First, have user_1 start the subtask
    start_subtask(project_name_subtask, "user_1", "subtask_1")

    # Verify user_1 has the subtask
    user1: User = get_user(project_name_subtask, "user_1")
    assert user1["active_subtask"] == "subtask_1"

    # Now have user_2 try to take over the subtask (which is still active)
    with pytest.raises(SubtaskValidationError, match="Subtask is already active"):
        start_subtask(project_name_subtask, "user_2", "subtask_1")

    # Verify user_1 still has the subtask
    user1 = get_user(project_name_subtask, "user_1")
    assert user1["active_subtask"] == "subtask_1"

    # Verify user_2 still has no active subtask
    user2: User = get_user(project_name_subtask, "user_2")
    assert user2["active_subtask"] == ""

    # Verify the subtask is still assigned to user_1
    subtask: Subtask = get_subtask(project_name_subtask, "subtask_1")
    assert subtask["active_user_id"] == "user_1"


def test_release_subtask_nonexistent_user(project_name_subtask, existing_subtask):
    """Test that releasing a subtask with a nonexistent user raises an error"""
    # Try to release a subtask with a user that doesn't exist
    nonexistent_user_id = "nonexistent_user"

    # This should raise a UserValidationError with a specific message
    with pytest.raises(UserValidationError, match=f"User {nonexistent_user_id} not found"):
        release_subtask(project_name_subtask, nonexistent_user_id, "subtask_1")

    # Verify that no user was created with this ID
    client = firestore.Client()
    user_ref = client.collection(f"{project_name_subtask}_users").document(nonexistent_user_id)
    assert not user_ref.get().exists

    # Verify the subtask wasn't modified
    subtask: Subtask = get_subtask(project_name_subtask, "subtask_1")
    assert subtask["active_user_id"] == ""  # Should remain empty
    assert subtask["completion_status"] == ""  # Should remain empty


def test_start_subtask_requires_qualification(
    project_name_subtask, existing_user, existing_subtask, existing_subtask_type
):
    """Test that a user can't start a subtask if they're not qualified for it"""
    # Update user to have empty qualifications list
    update_user(
        project_name_subtask,
        "user_1",
        UserUpdate(
            hourly_rate=50.0,
            active_subtask="",
            qualified_subtask_types=[],  # Empty list means no qualifications
        ),
    )

    # Try to start subtask - this should raise an error
    with pytest.raises(UserValidationError, match="User not qualified for this subtask type"):
        start_subtask(project_name_subtask, "user_1", "subtask_1")

    # Verify the user's active_subtask wasn't modified
    user: User = get_user(project_name_subtask, "user_1")
    assert user["active_subtask"] == ""

    # Verify the subtask wasn't modified
    subtask: Subtask = get_subtask(project_name_subtask, "subtask_1")
    assert subtask["active_user_id"] == ""

    # Now update the user to have the qualification and try again
    update_user(
        project_name_subtask,
        "user_1",
        UserUpdate(qualified_subtask_types=["segmentation_proofread"]),
    )

    # This should now succeed
    start_subtask(project_name_subtask, "user_1", "subtask_1")

    # Verify the user now has the subtask
    user = get_user(project_name_subtask, "user_1")
    assert user["active_subtask"] == "subtask_1"

    # Verify the subtask is now assigned to the user


def test_release_subtask_nonexistent_subtask(project_name_subtask, existing_user):
    """Test that releasing a nonexistent subtask raises an error"""
    client = firestore.Client()

    # First, manually update the user to have an active subtask that doesn't exist
    user_ref = client.collection(f"{project_name_subtask}_users").document("user_1")
    user_ref.update({"active_subtask": "nonexistent_subtask"})

    # Verify the user has been updated
    user = get_user(project_name_subtask, "user_1")
    assert user["active_subtask"] == "nonexistent_subtask"

    # Now try to release the subtask
    with pytest.raises(SubtaskValidationError, match="Subtask nonexistent_subtask not found"):
        release_subtask(project_name_subtask, "user_1", "nonexistent_subtask")

    # Verify the user's active_subtask wasn't modified
    user = get_user(project_name_subtask, "user_1")
    assert user["active_subtask"] == "nonexistent_subtask"
    # Verify that no subtask was created with this ID
    subtask_ref = client.collection(f"{project_name_subtask}_subtasks").document(
        "nonexistent_subtask"
    )
    assert not subtask_ref.get().exists


def test_auto_select_subtask_no_qualified_types(project_name_subtask, existing_subtask):
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
    create_user(project_name_subtask, unqualified_user)

    # Try to auto-select a subtask (by not specifying a subtask_id)
    result = start_subtask(project_name_subtask, "unqualified_user")

    # Verify that no subtask was selected
    assert result is None

    # Verify the user's active_subtask wasn't modified
    user = get_user(project_name_subtask, "unqualified_user")
    assert user["active_subtask"] == ""

    # Now update the user to have a qualification and try again
    update_user(
        project_name_subtask,
        "unqualified_user",
        UserUpdate(qualified_subtask_types=["segmentation_proofread"]),
    )

    # This should now succeed
    result = start_subtask(project_name_subtask, "unqualified_user")

    # Verify a subtask was selected
    assert result is not None

    # Verify the user now has the subtask
    user = get_user(project_name_subtask, "unqualified_user")
    assert user["active_subtask"] == result

    # Verify the subtask is now assigned to the user
    subtask = get_subtask(project_name_subtask, result)
    assert subtask["active_user_id"] == "unqualified_user"


def test_auto_select_subtask_prioritizes_assigned_to_user(
    project_name_subtask, existing_user, existing_subtask_type
):
    """Test that auto-selecting a subtask prioritizes tasks assigned to the user"""
    assigned_subtask = Subtask(
        **{
            "task_id": "task_1",
            "subtask_id": "assigned_subtask",
            "assigned_user_id": "user_1",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com/assigned",
            "priority": 1,
            "batch_id": "batch_1",
            "subtask_type": existing_subtask_type["subtask_type"],
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )

    unassigned_subtask = Subtask(
        **{
            "task_id": "task_1",
            "subtask_id": "unassigned_subtask",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com/unassigned",
            "priority": 10,  # Higher priority
            "batch_id": "batch_1",
            "subtask_type": existing_subtask_type["subtask_type"],
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )

    # Create both subtasks
    create_subtask(project_name_subtask, assigned_subtask)
    create_subtask(project_name_subtask, unassigned_subtask)

    # Auto-select a subtask (by not specifying a subtask_id)
    result = start_subtask(project_name_subtask, "user_1")

    # Verify that the assigned subtask was selected, even though it has lower priority
    assert result == "assigned_subtask"

    # Verify the user now has the assigned subtask
    user = get_user(project_name_subtask, "user_1")
    assert user["active_subtask"] == "assigned_subtask"

    # Verify the subtask is now assigned to the user
    subtask = get_subtask(project_name_subtask, "assigned_subtask")
    assert subtask["active_user_id"] == "user_1"

    # Verify the unassigned subtask wasn't modified
    unassigned = get_subtask(project_name_subtask, "unassigned_subtask")
    assert unassigned["active_user_id"] == ""

    assert unassigned["active_user_id"] == ""


def test_release_subtask_no_active_subtask(project_name_subtask, existing_user):
    """Test that releasing a subtask fails when the user doesn't have an active subtask"""
    # Verify the user doesn't have an active subtask
    user = get_user(project_name_subtask, "user_1")
    assert user["active_subtask"] == ""

    # Try to release a subtask
    with pytest.raises(UserValidationError, match="User does not have an active subtask"):
        release_subtask(project_name_subtask, "user_1", "subtask_1")

    # Try to release with a completion status
    with pytest.raises(UserValidationError, match="User does not have an active subtask"):
        release_subtask(project_name_subtask, "user_1", "subtask_1", "done")

    # Verify the user's active_subtask is still empty
    user = get_user(project_name_subtask, "user_1")
    assert user["active_subtask"] == ""


def test_release_subtask_with_dependencies(project_name_subtask, existing_subtask_type):
    """Test releasing a subtask with dependencies updates the dependent subtasks"""
    client = firestore.Client()

    # Create a user
    user_data = User(
        user_id="dep_test_user",
        hourly_rate=50.0,
        active_subtask="",
        qualified_subtask_types=[existing_subtask_type["subtask_type"]],
    )
    create_user(project_name_subtask, user_data)

    # Create two subtasks
    subtask1 = Subtask(
        task_id="task_dep",
        subtask_id="subtask_dep_1",
        assigned_user_id="dep_test_user",
        active_user_id="",
        completed_user_id="",
        ng_state="http://example.com/dep1",
        priority=1,
        batch_id="batch_dep",
        subtask_type=existing_subtask_type["subtask_type"],
        is_active=True,
        last_leased_ts=0.0,
        completion_status="",
    )

    subtask2 = Subtask(
        task_id="task_dep",
        subtask_id="subtask_dep_2",
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state="http://example.com/dep2",
        priority=2,
        batch_id="batch_dep",
        subtask_type=existing_subtask_type["subtask_type"],
        is_active=False,  # Initially inactive until dependency is satisfied
        last_leased_ts=0.0,
        completion_status="",
    )

    # Create both subtasks
    create_subtask(project_name_subtask, subtask1)
    create_subtask(project_name_subtask, subtask2)

    # Create a dependency between them
    dependency_data = Dependency(
        dependency_id="dep_test_1",
        subtask_id="subtask_dep_2",  # This subtask depends on subtask_dep_1
        dependent_on_subtask_id="subtask_dep_1",
        required_completion_status="done",
        is_satisfied=False,
    )
    create_dependency(project_name_subtask, dependency_data)

    # Start work on subtask1
    start_subtask(project_name_subtask, "dep_test_user", "subtask_dep_1")

    # Verify subtask2 is still inactive
    subtask2_before = get_subtask(project_name_subtask, "subtask_dep_2")
    assert subtask2_before["is_active"] is False

    # Release subtask1 with completion status "done"
    result = release_subtask(project_name_subtask, "dep_test_user", "subtask_dep_1", "done")
    assert result is True

    # Verify subtask1 was updated correctly
    subtask1_after = get_subtask(project_name_subtask, "subtask_dep_1")
    assert subtask1_after["completion_status"] == "done"
    assert subtask1_after["active_user_id"] == ""
    assert subtask1_after["completed_user_id"] == "dep_test_user"

    # Verify the dependency was satisfied
    deps_collection = client.collection(f"{project_name_subtask}_dependencies")
    dep_doc = deps_collection.document("dep_test_1").get()
    assert dep_doc.exists
    assert dep_doc.get("is_satisfied") is True

    # Verify subtask2 was activated
    subtask2_after = get_subtask(project_name_subtask, "subtask_dep_2")
    assert subtask2_after["is_active"] is True


def test_auto_select_subtask_finds_idle_task(project_name_subtask, existing_subtask_type):
    """Test that auto-selecting a subtask can find and take over an idle task"""
    # Create a user
    user_data = User(
        user_id="idle_test_user",
        hourly_rate=50.0,
        active_subtask="",
        qualified_subtask_types=[existing_subtask_type["subtask_type"]],
    )
    create_user(project_name_subtask, user_data)

    # Create another user who will have an idle task
    idle_user_data = User(
        user_id="user_with_idle_task",
        hourly_rate=50.0,
        active_subtask="idle_subtask",
        qualified_subtask_types=[existing_subtask_type["subtask_type"]],
    )
    create_user(project_name_subtask, idle_user_data)

    # Create an idle subtask (active but not worked on for a long time)
    current_time = time.time()
    idle_time = current_time - (
        get_max_idle_seconds() + 60
    )  # 60 seconds beyond the idle threshold

    idle_subtask = Subtask(
        task_id="task_idle",
        subtask_id="idle_subtask",
        assigned_user_id="",
        active_user_id="user_with_idle_task",
        completed_user_id="",
        ng_state="http://example.com/idle",
        priority=5,
        batch_id="batch_idle",
        subtask_type=existing_subtask_type["subtask_type"],
        is_active=True,
        last_leased_ts=idle_time,  # This makes it idle
        completion_status="",
    )
    create_subtask(project_name_subtask, idle_subtask)

    # Auto-select a subtask for the first user
    with patch("time.time", return_value=current_time):  # Freeze time for consistent testing
        result = start_subtask(project_name_subtask, "idle_test_user")

    # Verify that the idle subtask was selected
    assert result == "idle_subtask"

    # Verify the user now has the idle subtask
    user = get_user(project_name_subtask, "idle_test_user")
    assert user["active_subtask"] == "idle_subtask"

    # Verify the subtask is now assigned to the new user
    subtask = get_subtask(project_name_subtask, "idle_subtask")
    assert subtask["active_user_id"] == "idle_test_user"
    assert subtask["last_leased_ts"] == current_time

    # Verify the previous user's active_subtask was cleared
    previous_user = get_user(project_name_subtask, "user_with_idle_task")
    assert previous_user["active_subtask"] == ""


def test_auto_select_subtask_no_available_tasks(project_name_subtask, existing_subtask_type):
    """Test that auto-selecting a subtask returns None when no tasks are available"""
    # Create a user
    user_data = User(
        user_id="no_tasks_user",
        hourly_rate=50.0,
        active_subtask="",
        qualified_subtask_types=[existing_subtask_type["subtask_type"]],
    )
    create_user(project_name_subtask, user_data)

    # Create a subtask that is already completed
    completed_subtask = Subtask(
        task_id="task_1",
        subtask_id="subtask_completed",
        assigned_user_id="",
        active_user_id="",
        completed_user_id="test_user",
        ng_state="http://example.com",
        priority=1,
        batch_id="batch_1",
        subtask_type=existing_subtask_type["subtask_type"],
        is_active=True,
        last_leased_ts=0.0,
        completion_status="done",
    )
    create_subtask(project_name_subtask, completed_subtask)

    # Create a different subtask type
    different_type = SubtaskType(
        subtask_type="different_type", completion_statuses=["done", "rejected"]
    )
    create_subtask_type(different_type)

    # Create a subtask with a different type that the user isn't qualified for
    other_type_subtask = Subtask(
        task_id="task_1",
        subtask_id="subtask_other",
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state="http://example.com/other",
        priority=1,
        batch_id="batch_other",
        subtask_type="different_type",
        is_active=True,
        last_leased_ts=0.0,
        completion_status="",
    )
    create_subtask(project_name_subtask, other_type_subtask)

    # Create a subtask that is active but not idle
    active_subtask = Subtask(
        task_id="task_1",
        subtask_id="active_subtask",
        assigned_user_id="",
        active_user_id="other_user",
        completed_user_id="",
        ng_state="http://example.com/active",
        priority=5,
        batch_id="batch_active",
        subtask_type=existing_subtask_type["subtask_type"],
        is_active=True,
        last_leased_ts=time.time(),  # Recently leased, not idle
        completion_status="",
    )
    create_subtask(project_name_subtask, active_subtask)

    # Try to auto-select a subtask
    result = start_subtask(project_name_subtask, "no_tasks_user")

    # Verify that no subtask was selected
    assert result is None


def test_release_wrong_subtask_id(project_name_subtask, existing_subtask, existing_user):
    """Test that releasing a different subtask than active raises error"""
    # First start work on the subtask
    start_subtask(project_name_subtask, "user_1", "subtask_1")

    # Create another subtask
    other_subtask = Subtask(
        **{
            "task_id": "task_1",
            "subtask_id": "subtask_2",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com",
            "priority": 1,
            "batch_id": "batch_1",
            "subtask_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    create_subtask(project_name_subtask, other_subtask)

    # Try to release the wrong subtask
    with pytest.raises(
        UserValidationError, match="Subtask ID does not match user's active subtask"
    ):
        release_subtask(project_name_subtask, "user_1", "subtask_2", "done")

    # Verify original subtask is still active
    active_subtask = get_subtask(project_name_subtask, "subtask_1")
    assert active_subtask["active_user_id"] == "user_1"
    assert active_subtask["completion_status"] == ""

    # Verify other subtask wasn't modified
    other_subtask = get_subtask(project_name_subtask, "subtask_2")
    assert other_subtask["active_user_id"] == ""
    assert other_subtask["completion_status"] == ""
