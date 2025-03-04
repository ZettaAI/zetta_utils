# pylint: disable=redefined-outer-name,unused-argument
from copy import deepcopy

import pytest
from google.cloud import firestore

from zetta_utils.task_management.subtask import create_subtask, start_subtask
from zetta_utils.task_management.subtask_type import create_subtask_type
from zetta_utils.task_management.types import Subtask, SubtaskType, User, UserUpdate
from zetta_utils.task_management.user import create_user, get_user, update_user


@pytest.fixture
def sample_user() -> User:
    return User(
        **{
            "user_id": "test_user_1",
            "hourly_rate": 50.0,
            "active_subtask": "",
            "qualified_subtask_types": ["segmentation_proofread", "segmentation_verify"],
        }
    )


@pytest.fixture
def project_name_user() -> str:
    return "test_project_user"


@pytest.fixture
def user_collection(firestore_emulator, project_name_user):
    client = firestore.Client()
    collection = client.collection(f"{project_name_user}_users")
    yield collection
    for doc in collection.list_documents():
        doc.delete()


@pytest.fixture
def existing_user(user_collection, sample_user):
    user_collection.document(sample_user["user_id"]).set(sample_user)
    yield sample_user


@pytest.fixture
def sample_subtask_type() -> SubtaskType:
    return {"subtask_type": "segmentation_proofread", "completion_statuses": ["done", "need_help"]}


@pytest.fixture
def existing_subtask_type(sample_subtask_type):
    # First try to clean up any existing subtask type
    client = firestore.Client()
    doc_ref = client.collection("subtask_types").document(sample_subtask_type["subtask_type"])

    # Delete if exists
    if doc_ref.get().exists:
        doc_ref.delete()

    # Now create the subtask type
    create_subtask_type(sample_subtask_type)

    # Yield the data for the test to use
    yield sample_subtask_type

    # Clean up after the test
    doc_ref.delete()


@pytest.fixture
def sample_subtask(existing_subtask_type) -> Subtask:
    return {
        "task_id": "task_1",
        "subtask_id": "subtask_1",
        "completion_status": "",
        "assigned_user_id": "",
        "active_user_id": "",
        "completed_user_id": "",
        "link": "http://example.com",
        "priority": 1,
        "batch_id": "batch_1",
        "subtask_type": existing_subtask_type["subtask_type"],
        "is_active": True,
        "last_leased_ts": 0.0,
    }


@pytest.fixture
def existing_subtask(project_name_user, existing_subtask_type, sample_subtask):
    # First try to clean up any existing subtask
    client = firestore.Client()
    doc_ref = client.collection(f"{project_name_user}_subtasks").document(
        sample_subtask["subtask_id"]
    )

    # Delete if exists
    if doc_ref.get().exists:
        doc_ref.delete()

    # Now create the subtask
    create_subtask(project_name_user, sample_subtask)

    # Yield the data for the test to use
    yield sample_subtask

    # Clean up after the test
    doc_ref.delete()


def test_get_user_success(existing_user, project_name_user):
    result = get_user(project_name_user, "test_user_1")
    assert result == existing_user


def test_get_user_not_found(user_collection, project_name_user):
    with pytest.raises(KeyError, match="User test_user_1 not found"):
        get_user(project_name_user, "test_user_1")


def test_create_user_success(user_collection, sample_user, project_name_user):
    result = create_user(project_name_user, sample_user)
    assert result == sample_user["user_id"]
    doc = user_collection.document("test_user_1").get()
    assert doc.exists
    assert doc.to_dict() == sample_user


def test_create_user_already_exists(existing_user, sample_user, project_name_user):
    with pytest.raises(ValueError, match="User test_user_1 already exists"):
        create_user(project_name_user, sample_user)


def test_update_user_success(user_collection, existing_user, project_name_user):
    updated_data = deepcopy(existing_user)
    updated_data["hourly_rate"] = 60.0
    del updated_data["user_id"]

    result = update_user(project_name_user, "test_user_1", updated_data)
    assert result is True
    doc = user_collection.document("test_user_1").get()

    # Compare with merged data instead of just update data
    expected_data = {**existing_user, **updated_data}
    assert doc.to_dict() == expected_data


def test_update_user_not_found(user_collection, sample_user, project_name_user):
    with pytest.raises(KeyError, match="User test_user_1 not found"):
        update_user(project_name_user, "test_user_1", sample_user)


def test_create_user_with_qualifications(user_collection, project_name_user):
    user_data = User(
        **{
            "user_id": "test_user_1",
            "hourly_rate": 50.0,
            "active_subtask": "",
            "qualified_subtask_types": ["segmentation_proofread"],
        }
    )
    result = create_user(project_name_user, user_data)
    assert result == user_data["user_id"]

    stored_user = get_user(project_name_user, "test_user_1")
    assert stored_user == user_data


def test_update_user_qualifications(user_collection, existing_user, project_name_user):
    updated_data = UserUpdate(qualified_subtask_types=["segmentation_verify"])

    result = update_user(project_name_user, "test_user_1", updated_data)
    assert result is True

    stored_user = get_user(project_name_user, "test_user_1")
    assert stored_user["qualified_subtask_types"] == ["segmentation_verify"]


def test_start_subtask_requires_qualification(
    project_name_user, existing_user, existing_subtask, existing_subtask_type
):
    """Test that a user can't start a subtask if they're not qualified for it"""
    # Update user to have empty qualifications list
    update_user(
        project_name_user,
        "test_user_1",
        {
            "hourly_rate": 50.0,
            "active_subtask": "",
            "qualified_subtask_types": [],  # Empty list means no qualifications
        },
    )

    # Try to start subtask - this should raise an error
    with pytest.raises(Exception):
        start_subtask(project_name_user, "test_user_1", "subtask_1")
