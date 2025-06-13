# pylint: disable=redefined-outer-name,unused-argument
import pytest

from zetta_utils.task_management.task_type import (
    create_task_type,
    get_task_type,
)


def test_get_task_type_success(clean_db, db_session, project_name, sample_task_type):
    create_task_type(project_name=project_name, data=sample_task_type, db_session=db_session)
    result = get_task_type(
        project_name=project_name, task_type="segmentation_proofread", db_session=db_session
    )
    assert result == sample_task_type


def test_get_task_type_not_found(clean_db, db_session, project_name):
    with pytest.raises(
        KeyError, match="TaskType nonexistent not found in project test_project"
    ):
        get_task_type(
            project_name=project_name, task_type="nonexistent", db_session=db_session
        )


def test_create_task_type_success(clean_db, db_session, project_name, sample_task_type):
    result = create_task_type(
        project_name=project_name, data=sample_task_type, db_session=db_session
    )
    assert result == sample_task_type["task_type"]

    # Note: The following test was using get_collection which doesn't exist in the SQL version
    # This would need to be replaced with actual database verification if needed


def test_create_task_type_already_exists(
    clean_db, db_session, project_name, sample_task_type
):
    create_task_type(project_name=project_name, data=sample_task_type, db_session=db_session)
    with pytest.raises(
        ValueError,
        match="TaskType segmentation_proofread already exists in project test_project",
    ):
        create_task_type(
            project_name=project_name, data=sample_task_type, db_session=db_session
        )
