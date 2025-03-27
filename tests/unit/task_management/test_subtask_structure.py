# pylint: disable=redefined-outer-name,unused-argument
import pytest
from google.cloud import firestore

from zetta_utils.task_management.project import get_collection
from zetta_utils.task_management.subtask_structure import create_subtask_structure


@pytest.fixture
def project_name() -> str:
    return "test_project_subtask_structure"


def test_create_subtask_structure_nonexistent_task(firestore_emulator, project_name):
    """Test that creating a subtask structure for a nonexistent task raises KeyError"""
    client = firestore.Client()
    transaction = client.transaction()

    with pytest.raises(KeyError, match="Task nonexistent_task not found"):
        create_subtask_structure(
            transaction=transaction,
            project_name=project_name,
            task_id="nonexistent_task",
            subtask_structure="segmentation_proofread_simple",
            priority=2,
            subtask_structure_kwargs={},
        )


def test_create_subtask_structure_success(firestore_emulator, project_name, existing_task):
    """Test successful creation of a subtask structure"""
    client = firestore.Client()
    transaction = client.transaction()

    @firestore.transactional
    def create_in_transaction(transaction):
        result = create_subtask_structure(
            transaction=transaction,
            project_name=project_name,
            task_id="task_1",
            subtask_structure="segmentation_proofread_simple",
            priority=2,
            subtask_structure_kwargs={},
        )
        return result

    result = create_in_transaction(transaction)
    assert result is True

    subtasks = list(get_collection(project_name, "subtasks").stream())
    assert len(subtasks) == 3

    dependencies = list(get_collection(project_name, "dependencies").stream())
    assert len(dependencies) == 2

    task_ref = get_collection(project_name, "tasks").document("task_1")
    task = task_ref.get().to_dict()
    assert task["status"] == "ingested"
