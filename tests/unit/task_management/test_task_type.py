"""Tests for task_type module"""

# pylint: disable=unused-argument,redefined-outer-name

import pytest

from zetta_utils.task_management.task_type import (
    add_standard_task_types,
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
    with pytest.raises(KeyError, match="TaskType nonexistent not found in project test_project"):
        get_task_type(project_name=project_name, task_type="nonexistent", db_session=db_session)


def test_create_task_type_success(clean_db, db_session, project_name, sample_task_type):
    result = create_task_type(
        project_name=project_name, data=sample_task_type, db_session=db_session
    )
    assert result == sample_task_type["task_type"]

    # Note: The following test was using get_collection which doesn't exist in the SQL version
    # This would need to be replaced with actual database verification if needed


def test_create_task_type_already_exists(clean_db, db_session, project_name, sample_task_type):
    create_task_type(project_name=project_name, data=sample_task_type, db_session=db_session)
    with pytest.raises(
        ValueError,
        match="TaskType segmentation_proofread already exists in project test_project",
    ):
        create_task_type(project_name=project_name, data=sample_task_type, db_session=db_session)


def test_add_standard_task_types(clean_db, db_session, project_name):
    """Test creating standard task types for a project"""
    # Create standard task types
    result = add_standard_task_types(project_name=project_name, db_session=db_session)

    # Verify the correct task types were created
    assert "trace_v0" in result
    assert "trace_postprocess_v0" in result
    assert "trace_feedback_v0" in result

    # Verify completion statuses
    assert result["trace_v0"] == ["Done", "Can't Continue", "Merger", "Wrong Cell Type"]
    assert result["trace_postprocess_v0"] == ["Done"]
    assert result["trace_feedback_v0"] == ["Faulty Task", "Accurate", "Inaccurate", "Fair"]

    # Verify they exist in the database
    trace_type = get_task_type(
        project_name=project_name, task_type="trace_v0", db_session=db_session
    )
    assert trace_type["task_type"] == "trace_v0"
    assert trace_type["completion_statuses"] == [
        "Done",
        "Can't Continue",
        "Merger",
        "Wrong Cell Type",
    ]

    trace_postprocess_type = get_task_type(
        project_name=project_name, task_type="trace_postprocess_v0", db_session=db_session
    )
    assert trace_postprocess_type["task_type"] == "trace_postprocess_v0"
    assert trace_postprocess_type["completion_statuses"] == ["Done"]

    trace_feedback_type = get_task_type(
        project_name=project_name, task_type="trace_feedback_v0", db_session=db_session
    )
    assert trace_feedback_type["task_type"] == "trace_feedback_v0"
    assert trace_feedback_type["completion_statuses"] == [
        "Faulty Task",
        "Accurate",
        "Inaccurate",
        "Fair",
    ]


def test_add_standard_task_types_replaces_existing(
    clean_db, db_session, project_name, sample_task_type
):
    """Test that creating standard task types replaces existing ones"""
    # Create a custom task type first
    create_task_type(project_name=project_name, data=sample_task_type, db_session=db_session)

    # Verify it exists
    existing = get_task_type(
        project_name=project_name, task_type="segmentation_proofread", db_session=db_session
    )
    assert existing is not None

    # Create standard task types - should delete the existing one
    result = add_standard_task_types(project_name=project_name, db_session=db_session)

    # Verify standard types were created
    assert "trace_v0" in result
    assert "trace_postprocess_v0" in result

    # Verify the custom task type was deleted
    with pytest.raises(KeyError, match="TaskType segmentation_proofread not found"):
        get_task_type(
            project_name=project_name, task_type="segmentation_proofread", db_session=db_session
        )
