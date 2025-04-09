# pylint: disable=redefined-outer-name,unused-argument
import pytest
from google.cloud import firestore

from zetta_utils.task_management.subtask_type import (
    create_subtask_type,
    get_subtask_type,
)
from zetta_utils.task_management.types import SubtaskType


@pytest.fixture(autouse=True)
def clean_subtask_types(firestore_emulator):
    """Clean the global subtask_types collection before and after each test"""
    client = firestore.Client()
    collection =get_collection(project_name, "subtask_types")
    for doc in collection.list_documents():
        doc.delete()
    yield
    for doc in collection.list_documents():
        doc.delete()


@pytest.fixture
def sample_subtask_type() -> SubtaskType:
    return SubtaskType(
        **{"subtask_type": "segmentation_proofread", "completion_statuses": ["done", "need_help"]}
    )


def test_get_subtask_type_success(sample_subtask_type):
    create_subtask_type(sample_subtask_type)
    result = get_subtask_type("segmentation_proofread")
    assert result == sample_subtask_type


def test_get_subtask_type_not_found():
    with pytest.raises(KeyError, match="Subtask type not found"):
        get_subtask_type("nonexistent")


def test_create_subtask_type_success(sample_subtask_type):
    result = create_subtask_type(sample_subtask_type)
    assert result == sample_subtask_type["subtask_type"]

    doc = firestore.Client().collection("subtask_types").document("segmentation_proofread").get()
    assert doc.exists
    assert doc.to_dict() == sample_subtask_type


def test_create_subtask_type_already_exists(sample_subtask_type):
    create_subtask_type(sample_subtask_type)
    with pytest.raises(ValueError, match="Subtask type already exists"):
        create_subtask_type(sample_subtask_type)
