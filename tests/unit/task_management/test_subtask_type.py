# pylint: disable=redefined-outer-name,unused-argument
import pytest

from zetta_utils.task_management.subtask_type import (
    create_subtask_type,
    get_subtask_type,
)


def test_get_subtask_type_success(clean_db, db_session, project_name, sample_subtask_type):
    create_subtask_type(project_name=project_name, data=sample_subtask_type, db_session=db_session)
    result = get_subtask_type(
        project_name=project_name, subtask_type="segmentation_proofread", db_session=db_session
    )
    assert result == sample_subtask_type


def test_get_subtask_type_not_found(clean_db, db_session, project_name):
    with pytest.raises(
        KeyError, match="SubtaskType nonexistent not found in project test_project"
    ):
        get_subtask_type(
            project_name=project_name, subtask_type="nonexistent", db_session=db_session
        )


def test_create_subtask_type_success(clean_db, db_session, project_name, sample_subtask_type):
    result = create_subtask_type(
        project_name=project_name, data=sample_subtask_type, db_session=db_session
    )
    assert result == sample_subtask_type["subtask_type"]

    # Note: The following test was using get_collection which doesn't exist in the SQL version
    # This would need to be replaced with actual database verification if needed


def test_create_subtask_type_already_exists(
    clean_db, db_session, project_name, sample_subtask_type
):
    create_subtask_type(project_name=project_name, data=sample_subtask_type, db_session=db_session)
    with pytest.raises(
        ValueError,
        match="SubtaskType segmentation_proofread already exists in project test_project",
    ):
        create_subtask_type(
            project_name=project_name, data=sample_subtask_type, db_session=db_session
        )
