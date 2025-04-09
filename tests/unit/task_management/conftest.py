# pylint: disable=redefined-outer-name,unused-argument
import pytest
from google.cloud import firestore

from zetta_utils.task_management.project import get_collection
from zetta_utils.task_management.subtask import create_subtask
from zetta_utils.task_management.subtask_type import create_subtask_type
from zetta_utils.task_management.task import create_task
from zetta_utils.task_management.types import Subtask, SubtaskType, Task, User
from zetta_utils.task_management.user import create_user


@pytest.fixture
def project_name() -> str:
    return "test_project"


@pytest.fixture(autouse=True)
def clean_collections(firestore_emulator, project_name):
    client = firestore.Client()
    collections = [
        f"projects/{project_name}/tasks",
        f"projects/{project_name}/subtasks",
        f"projects/{project_name}/dependencies",
        f"projects/{project_name}/timesheets",
        f"projects/{project_name}/users",
        f"projects/{project_name}/subtask_types",
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
    return {
        "user_id": "test_user",
        "hourly_rate": 50.0,
        "active_subtask": "",
        "qualified_subtask_types": ["segmentation_proofread"],
    }


@pytest.fixture
def existing_user(firestore_emulator, project_name, sample_user):
    create_user(project_name, sample_user)
    yield sample_user


@pytest.fixture
def sample_task() -> Task:
    return Task(
        **{
            "task_id": "task_1",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "task_type": "segmentation",
            "ng_state": "http://example.com/task_1",
        }
    )


@pytest.fixture
def existing_task(firestore_emulator, project_name, sample_task):
    create_task(project_name, sample_task)
    yield sample_task


@pytest.fixture
def sample_subtask_type() -> SubtaskType:
    return {
        "subtask_type": "segmentation_proofread",
        "completion_statuses": ["done", "need_help"],
    }


@pytest.fixture
def existing_subtask_type(firestore_emulator, project_name, sample_subtask_type):
    create_subtask_type(project_name, sample_subtask_type)
    yield sample_subtask_type


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


@pytest.fixture
def sample_subtask(existing_subtask_type) -> Subtask:
    return {
        "task_id": "task_1",
        "subtask_id": "subtask_1",
        "completion_status": "",
        "assigned_user_id": "",
        "active_user_id": "",
        "completed_user_id": "",
        "ng_state": "http://example.com",
        "ng_state_initial": "http://example.com",
        "priority": 1,
        "batch_id": "batch_1",
        "subtask_type": existing_subtask_type["subtask_type"],
        "is_active": True,
        "last_leased_ts": 0.0,
    }


@pytest.fixture
def existing_subtask(project_name, existing_subtask_type, sample_subtask):
    doc_ref = get_collection(project_name, "subtasks").document(sample_subtask["subtask_id"])

    if doc_ref.get().exists:
        doc_ref.delete()

    create_subtask(project_name, sample_subtask)

    yield sample_subtask

    doc_ref.delete()
