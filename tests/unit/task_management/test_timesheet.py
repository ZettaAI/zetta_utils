# pylint: disable=redefined-outer-name,unused-argument
import time

import pytest
from google.cloud import firestore

from zetta_utils.task_management.exceptions import UserValidationError
from zetta_utils.task_management.subtask import create_subtask, start_subtask
from zetta_utils.task_management.subtask_type import create_subtask_type
from zetta_utils.task_management.timesheet import submit_timesheet
from zetta_utils.task_management.types import Subtask, SubtaskType, User
from zetta_utils.task_management.user import create_user


@pytest.fixture
def project_name_timesheet() -> str:
    return "test_project_timesheet"


@pytest.fixture(autouse=True)
def clean_collections(firestore_emulator, project_name_timesheet):
    client = firestore.Client()
    collections = [
        f"{project_name_timesheet}_users",
        f"{project_name_timesheet}_subtasks",
        f"{project_name_timesheet}_timesheets",
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
            "user_id": "test_user",
            "hourly_rate": 50.0,
            "active_subtask": "",
            "qualified_subtask_types": ["test_type"],
        }
    )


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
            "subtask_type": "test_type",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )


@pytest.fixture
def sample_subtask_type() -> SubtaskType:
    return SubtaskType(
        **{"subtask_type": "test_type", "completion_statuses": ["done", "need_help"]}
    )


@pytest.fixture
def existing_user(project_name_timesheet, sample_user):
    create_user(project_name_timesheet, sample_user)
    yield sample_user


@pytest.fixture
def existing_subtask_type(sample_subtask_type):
    """Create the subtask type in Firestore"""
    client = firestore.Client()
    doc_ref = client.collection("subtask_types").document(sample_subtask_type["subtask_type"])

    # Delete if exists
    if doc_ref.get().exists:
        doc_ref.delete()

    create_subtask_type(sample_subtask_type)
    yield sample_subtask_type
    doc_ref.delete()


@pytest.fixture
def existing_subtask(project_name_timesheet, sample_subtask, existing_subtask_type):
    create_subtask(project_name_timesheet, sample_subtask)
    yield sample_subtask


def test_submit_timesheet_success(project_name_timesheet, existing_user, existing_subtask):
    """Test successful timesheet submission"""
    # Start the subtask
    start_subtask(project_name_timesheet, "test_user", "subtask_1")

    submit_timesheet(project_name_timesheet, "test_user", 3600, "subtask_1")

    # Verify last_leased_ts was updated
    client = firestore.Client()
    subtask_doc = (
        client.collection(f"{project_name_timesheet}_subtasks").document("subtask_1").get()
    )
    assert subtask_doc.exists
    subtask_data = subtask_doc.to_dict()
    assert subtask_data["last_leased_ts"] > time.time() - 10  # Updated within last 10 seconds


def test_submit_timesheet_update_existing(project_name_timesheet, existing_user, existing_subtask):
    """Test updating an existing timesheet entry with additional duration"""
    # Start the subtask
    start_subtask(project_name_timesheet, "test_user", "subtask_1")

    # Submit first timesheet entry
    initial_duration = 3600
    submit_timesheet(project_name_timesheet, "test_user", initial_duration, "subtask_1")

    # Submit second timesheet entry for the same subtask
    additional_duration = 1800
    submit_timesheet(project_name_timesheet, "test_user", additional_duration, "subtask_1")

    # Verify the timesheet entry was updated with the combined duration
    client = firestore.Client()
    timesheet_doc = (
        client.collection(f"{project_name_timesheet}_timesheets")
        .document("test_user_subtask_1")
        .get()
    )
    assert timesheet_doc.exists
    timesheet_data = timesheet_doc.to_dict()
    assert timesheet_data["duration_seconds"] == initial_duration + additional_duration
    assert timesheet_data["user_id"] == "test_user"
    assert timesheet_data["subtask_id"] == "subtask_1"
    assert "last_updated_ts" in timesheet_data
    assert timesheet_data["last_updated_ts"] > time.time() - 10  # Updated within last 10 seconds


def test_submit_timesheet_no_active_subtask(project_name_timesheet, existing_user):
    """Test submitting timesheet without an active subtask"""
    with pytest.raises(UserValidationError, match="User does not have an active subtask"):
        submit_timesheet(project_name_timesheet, "test_user", 3600, "subtask_1")


def test_submit_timesheet_no_user_subtask(project_name_timesheet, existing_user, existing_subtask):
    with pytest.raises(UserValidationError, match="User does not have an active subtask"):
        submit_timesheet(project_name_timesheet, "test_user", 3600, "subtask_1")


def test_submit_timesheet_negative_duration(
    project_name_timesheet, existing_user, existing_subtask
):
    """Test submitting timesheet with negative duration"""
    start_subtask(project_name_timesheet, "test_user", "subtask_1")

    with pytest.raises(ValueError, match="Duration must be positive"):
        submit_timesheet(project_name_timesheet, "test_user", -3600, "subtask_1")


def test_submit_timesheet_nonexistent_user(project_name_timesheet, existing_subtask):
    """Test submitting timesheet for a user that doesn't exist"""
    with pytest.raises(UserValidationError, match="User nonexistent_user not found"):
        submit_timesheet(project_name_timesheet, "nonexistent_user", 3600, "subtask_1")


def test_submit_timesheet_nonexistent_subtask(project_name_timesheet, existing_user):
    """Test submitting timesheet when user has nonexistent subtask"""
    # Manually set user's active_subtask to a nonexistent one
    client = firestore.Client()
    user_ref = client.collection(f"{project_name_timesheet}_users").document("test_user")
    user_ref.update({"active_subtask": "nonexistent_subtask"})

    with pytest.raises(UserValidationError, match="Subtask nonexistent_subtask not found"):
        submit_timesheet(project_name_timesheet, "test_user", 3600, "nonexistent_subtask")


def test_submit_timesheet_wrong_user(project_name_timesheet, existing_user, existing_subtask):
    """Test submitting timesheet for subtask assigned to different user"""
    # Start subtask with one user
    start_subtask(project_name_timesheet, "test_user", "subtask_1")

    # Create another user
    other_user = User(
        **{
            "user_id": "other_user",
            "hourly_rate": 50.0,
            "active_subtask": "subtask_1",  # Manually set to first user's subtask
            "qualified_subtask_types": ["test_type"],
        }
    )
    create_user(project_name_timesheet, other_user)

    with pytest.raises(UserValidationError, match="Subtask not assigned to this user"):
        submit_timesheet(project_name_timesheet, "other_user", 3600, "subtask_1")


def test_submit_timesheet_mismatched_subtask_id(
    project_name_timesheet, existing_user, existing_subtask
):
    """Test submitting timesheet with a subtask_id that doesn't match user's active subtask"""
    # Start subtask with one ID
    start_subtask(project_name_timesheet, "test_user", "subtask_1")

    # Try to submit timesheet with a different subtask_id
    with pytest.raises(
        UserValidationError,
        match="Provided subtask_id subtask_2 does not match user's active subtask subtask_1",
    ):
        submit_timesheet(project_name_timesheet, "test_user", 3600, "subtask_2")
