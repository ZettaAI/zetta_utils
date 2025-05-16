# pylint: disable=redefined-outer-name,unused-argument
import pytest

from zetta_utils.task_management.project import get_collection
from zetta_utils.task_management.subtask_type import (
    create_subtask_type,
    get_subtask_type,
)


def test_get_subtask_type_success(project_name, sample_subtask_type):
    create_subtask_type(project_name, sample_subtask_type)
    result = get_subtask_type(project_name, "segmentation_proofread")
    assert result == sample_subtask_type


def test_get_subtask_type_not_found(project_name):
    with pytest.raises(KeyError, match="Subtask type not found"):
        get_subtask_type(project_name, "nonexistent")


def test_create_subtask_type_success(project_name, sample_subtask_type):
    result = create_subtask_type(project_name, sample_subtask_type)
    assert result == sample_subtask_type["subtask_type"]

    doc = get_collection(project_name, "subtask_types").document("segmentation_proofread").get()
    assert doc.exists
    assert doc.to_dict() == sample_subtask_type


def test_create_subtask_type_already_exists(project_name, sample_subtask_type):
    create_subtask_type(project_name, sample_subtask_type)
    with pytest.raises(ValueError, match="Subtask type already exists"):
        create_subtask_type(project_name, sample_subtask_type)
