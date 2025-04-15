# pylint: disable=redefined-outer-name,unused-argument
import pytest
from sqlalchemy import select

from zetta_utils.task_management.subtask_type import (
    create_subtask_type,
    get_subtask_type,
)


def test_get_subtask_type_success(db_session, project_name, sample_subtask_type):
    create_subtask_type(db_session, project_name, sample_subtask_type)
    result = get_subtask_type(db_session, project_name, "segmentation_proofread")
    assert result == sample_subtask_type


def test_get_subtask_type_not_found(db_session, project_name):
    with pytest.raises(KeyError, match="Subtask type not found"):
        get_subtask_type(db_session, project_name, "nonexistent")


def test_create_subtask_type_success(db_session, project_name, sample_subtask_type):
    result = create_subtask_type(db_session, project_name, sample_subtask_type)
    assert result == sample_subtask_type["subtask_type"]

    # Verify the subtask type was created in the database
    from zetta_utils.task_management.db.models import SubtaskTypeModel

    query = (
        select(SubtaskTypeModel)
        .where(SubtaskTypeModel.subtask_type == "segmentation_proofread")
        .where(SubtaskTypeModel.project_name == project_name)
    )
    db_subtask_type = db_session.execute(query).scalar_one()

    assert db_subtask_type is not None
    assert db_subtask_type.subtask_type == sample_subtask_type["subtask_type"]
    assert db_subtask_type.completion_statuses == sample_subtask_type["completion_statuses"]


def test_create_subtask_type_already_exists(db_session, project_name, sample_subtask_type):
    create_subtask_type(db_session, project_name, sample_subtask_type)
    with pytest.raises(ValueError, match="Subtask type already exists"):
        create_subtask_type(db_session, project_name, sample_subtask_type)
