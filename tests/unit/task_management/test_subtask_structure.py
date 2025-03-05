# pylint: disable=redefined-outer-name,unused-argument
import pytest
from google.cloud import firestore

from zetta_utils.task_management.subtask_structure import create_subtask_structure
from zetta_utils.task_management.task import create_task
from zetta_utils.task_management.types import Task


@pytest.fixture
def project_name_subtask_structure() -> str:
    return "test_project_subtask_structure"


@pytest.fixture(autouse=True)
def clean_collections(firestore_emulator, project_name_subtask_structure):
    client = firestore.Client()
    collections = [
        f"{project_name_subtask_structure}_tasks",
        f"{project_name_subtask_structure}_subtasks",
        f"{project_name_subtask_structure}_dependencies",
    ]
    for coll in collections:
        for doc in client.collection(coll).list_documents():
            doc.delete()
    yield
    for coll in collections:
        for doc in client.collection(coll).list_documents():
            doc.delete()


@pytest.fixture
def sample_task() -> Task:
    return {
        "task_id": "task_1",
        "batch_id": "batch_1",
        "status": "pending_ingestion",
        "task_type": "segmentation",
        "ng_state": "http://example.com/task_1",
    }


@pytest.fixture
def existing_task(firestore_emulator, project_name_subtask_structure, sample_task):
    create_task(project_name_subtask_structure, sample_task)
    yield sample_task


def test_create_subtask_structure_nonexistent_task(
    firestore_emulator, project_name_subtask_structure
):
    """Test that creating a subtask structure for a nonexistent task raises KeyError"""
    client = firestore.Client()
    transaction = client.transaction()

    with pytest.raises(KeyError, match="Task nonexistent_task not found"):
        create_subtask_structure(
            client=client,
            transaction=transaction,
            project_name=project_name_subtask_structure,
            task_id="nonexistent_task",
            subtask_structure="segmentation_proofread_simple",
            priority=2,
        )


def test_create_subtask_structure_success(
    firestore_emulator, project_name_subtask_structure, existing_task
):
    """Test successful creation of a subtask structure"""
    client = firestore.Client()
    transaction = client.transaction()

    @firestore.transactional
    def create_in_transaction(transaction):
        result = create_subtask_structure(
            client=client,
            transaction=transaction,
            project_name=project_name_subtask_structure,
            task_id="task_1",
            subtask_structure="segmentation_proofread_simple",
            priority=2,
        )
        return result

    result = create_in_transaction(transaction)
    assert result is True

    # Verify that subtasks were created
    subtasks = list(client.collection(f"{project_name_subtask_structure}_subtasks").stream())
    assert len(subtasks) == 3  # Should have created 3 subtasks

    # Verify that dependencies were created
    dependencies = list(
        client.collection(f"{project_name_subtask_structure}_dependencies").stream()
    )
    assert len(dependencies) == 2  # Should have created 2 dependencies

    # Verify that task status was updated
    task_ref = client.collection(f"{project_name_subtask_structure}_tasks").document("task_1")
    task = task_ref.get().to_dict()
    assert task["status"] == "ingested"
