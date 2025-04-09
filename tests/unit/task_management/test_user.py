# pylint: disable=redefined-outer-name,unused-argument
from copy import deepcopy

import pytest
from google.cloud import firestore

from zetta_utils.task_management.project import get_collection
from zetta_utils.task_management.subtask import create_subtask, start_subtask
from zetta_utils.task_management.subtask_type import create_subtask_type
from zetta_utils.task_management.types import Subtask, SubtaskType, User, UserUpdate
from zetta_utils.task_management.user import create_user, get_user, update_user


def test_get_user_success(existing_user, project_name):
    result = get_user(project_name, "test_user")
    assert result == existing_user


def test_get_user_not_found(project_name):
    with pytest.raises(KeyError, match="User test_user not found"):
        get_user(project_name, "test_user")


def test_create_user_success(sample_user, project_name):
    result = create_user(project_name, sample_user)
    assert result == sample_user["user_id"]
    doc = get_collection(project_name,"users").document("test_user").get()
    assert doc.exists
    assert doc.to_dict() == sample_user


def test_create_user_already_exists(existing_user, sample_user, project_name):
    with pytest.raises(ValueError, match="User test_user already exists"):
        create_user(project_name, sample_user)


def test_update_user_success(existing_user, project_name):
    updated_data = deepcopy(existing_user)
    updated_data["hourly_rate"] = 60.0
    del updated_data["user_id"]

    result = update_user(project_name, "test_user", updated_data)
    assert result is True
    doc = get_collection(project_name,"users").document("test_user").get()

    # Compare with merged data instead of just update data
    expected_data = {**existing_user, **updated_data}
    assert doc.to_dict() == expected_data


def test_update_user_not_found(sample_user, project_name):
    with pytest.raises(KeyError, match="User test_user not found"):
        update_user(project_name, "test_user", sample_user)


def test_create_user_with_qualifications(project_name):
    user_data = User(
        **{
            "user_id": "test_user",
            "hourly_rate": 50.0,
            "active_subtask": "",
            "qualified_subtask_types": ["segmentation_proofread"],
        }
    )
    result = create_user(project_name, user_data)
    assert result == user_data["user_id"]

    stored_user = get_user(project_name, "test_user")
    assert stored_user == user_data


def test_update_user_qualifications(existing_user, project_name):
    updated_data = UserUpdate(qualified_subtask_types=["segmentation_verify"])

    result = update_user(project_name, "test_user", updated_data)
    assert result is True

    stored_user = get_user(project_name, "test_user")
    assert stored_user["qualified_subtask_types"] == ["segmentation_verify"]


def test_start_subtask_requires_qualification(
    project_name, existing_user, existing_subtask, existing_subtask_type
):
    """Test that a user can't start a subtask if they're not qualified for it"""
    # Update user to have empty qualifications list
    update_user(
        project_name,
        "test_user",
        {
            "hourly_rate": 50.0,
            "active_subtask": "",
            "qualified_subtask_types": [],  # Empty list means no qualifications
        },
    )

    # Try to start subtask - this should raise an error
    with pytest.raises(Exception):
        start_subtask(project_name, "test_user", "subtask_1")
