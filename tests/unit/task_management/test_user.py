# pylint: disable=redefined-outer-name,unused-argument
from copy import deepcopy

import pytest
from sqlalchemy import select

from zetta_utils.task_management.db.models import UserModel
from zetta_utils.task_management.project import get_collection
from zetta_utils.task_management.subtask import start_subtask
from zetta_utils.task_management.types import User, UserUpdate
from zetta_utils.task_management.user import create_user, get_user, update_user


def test_get_user_success_sql(db_session, project_name, sample_user):
    create_user(db_session, project_name, sample_user)
    result = get_user(db_session, project_name, "test_user")
    assert result == sample_user


def test_get_user_not_found_sql(db_session, project_name):
    with pytest.raises(KeyError, match="User test_user not found"):
        get_user(db_session, project_name, "test_user")


def test_create_user_success_sql(db_session, project_name, sample_user):
    result = create_user(db_session, project_name, sample_user)
    assert result == sample_user["user_id"]

    # Verify the user was created in the database
    query = (
        select(UserModel)
        .where(UserModel.user_id == "test_user")
        .where(UserModel.project_name == project_name)
    )
    db_user = db_session.execute(query).scalar_one()

    assert db_user is not None
    assert db_user.user_id == sample_user["user_id"]
    assert db_user.hourly_rate == sample_user["hourly_rate"]
    assert db_user.active_subtask == sample_user["active_subtask"]
    assert db_user.qualified_subtask_types == sample_user["qualified_subtask_types"]


def test_create_user_already_exists_sql(db_session, project_name, sample_user):
    # First create
    create_user(db_session, project_name, sample_user)

    # Try to create again
    with pytest.raises(ValueError, match="User test_user already exists"):
        create_user(db_session, project_name, sample_user)


def test_update_user_success_sql(db_session, project_name, sample_user):
    # First create the user
    create_user(db_session, project_name, sample_user)

    # Update
    updated_data = UserUpdate(
        hourly_rate=60.0,
    )

    result = update_user(db_session, project_name, "test_user", updated_data)
    assert result is True

    # Fetch the updated user
    updated_user = get_user(db_session, project_name, "test_user")

    # Check updated field
    assert updated_user["hourly_rate"] == 60.0

    # Check other fields weren't modified
    assert updated_user["user_id"] == sample_user["user_id"]
    assert updated_user["active_subtask"] == sample_user["active_subtask"]
    assert updated_user["qualified_subtask_types"] == sample_user["qualified_subtask_types"]


def test_update_user_not_found_sql(db_session, project_name):
    update_data = UserUpdate(hourly_rate=60.0)

    with pytest.raises(KeyError, match="User test_user not found"):
        update_user(db_session, project_name, "test_user", update_data)


def test_create_user_with_qualifications_sql(db_session, project_name):
    user_data = User(
        **{
            "user_id": "test_user",
            "hourly_rate": 50.0,
            "active_subtask": "",
            "qualified_subtask_types": ["segmentation_proofread"],
        }
    )
    result = create_user(db_session, project_name, user_data)
    assert result == user_data["user_id"]

    stored_user = get_user(db_session, project_name, "test_user")
    assert stored_user == user_data


def test_update_user_qualifications_sql(db_session, project_name, sample_user):
    # First create the user
    create_user(db_session, project_name, sample_user)

    # Update qualifications
    updated_data = UserUpdate(qualified_subtask_types=["segmentation_verify"])

    result = update_user(db_session, project_name, "test_user", updated_data)
    assert result is True

    stored_user = get_user(db_session, project_name, "test_user")
    assert stored_user["qualified_subtask_types"] == ["segmentation_verify"]


# Keep the original Firestore tests for now for backward compatibility


def test_get_user_success(existing_user, project_name):
    result = get_user(project_name=project_name, user_id="test_user")
    assert result == existing_user


def test_get_user_not_found(clean_collections, project_name):
    with pytest.raises(KeyError, match="User test_user not found"):
        get_user(project_name=project_name, user_id="test_user")


def test_create_user_success(project_name, clean_collections, sample_user):
    result = create_user(project_name=project_name, data=sample_user)
    assert result == sample_user["user_id"]
    doc = get_collection(project_name, "users").document("test_user").get()
    assert doc.exists
    assert doc.to_dict() == sample_user


def test_create_user_already_exists(clean_collections, existing_user, sample_user, project_name):
    with pytest.raises(ValueError, match="User test_user already exists"):
        create_user(project_name=project_name, data=sample_user)


def test_update_user_success(clean_collections, existing_user, project_name):
    updated_data = deepcopy(existing_user)
    updated_data["hourly_rate"] = 60.0
    del updated_data["user_id"]

    result = update_user(project_name=project_name, user_id="test_user", data=updated_data)
    assert result is True
    doc = get_collection(project_name, "users").document("test_user").get()

    # Compare with merged data instead of just update data
    expected_data = {**existing_user, **updated_data}
    assert doc.to_dict() == expected_data


def test_update_user_not_found(clean_collections, sample_user, project_name):
    with pytest.raises(KeyError, match="User test_user not found"):
        update_user(project_name=project_name, user_id="test_user", data=sample_user)


def test_create_user_with_qualifications(clean_collections, project_name):
    user_data = User(
        **{
            "user_id": "test_user",
            "hourly_rate": 50.0,
            "active_subtask": "",
            "qualified_subtask_types": ["segmentation_proofread"],
        }
    )
    result = create_user(project_name=project_name, data=user_data)
    assert result == user_data["user_id"]

    stored_user = get_user(project_name=project_name, user_id="test_user")
    assert stored_user == user_data


def test_update_user_qualifications(clean_collections, existing_user, project_name):
    updated_data = UserUpdate(qualified_subtask_types=["segmentation_verify"])

    result = update_user(project_name=project_name, user_id="test_user", data=updated_data)
    assert result is True

    stored_user = get_user(project_name=project_name, user_id="test_user")
    assert stored_user["qualified_subtask_types"] == ["segmentation_verify"]


def test_start_subtask_requires_qualification(
    project_name, clean_collections, existing_user, existing_subtask, existing_subtask_type
):
    """Test that a user can't start a subtask if they're not qualified for it"""
    # Update user to have empty qualifications list
    update_user(
        project_name=project_name,
        user_id="test_user",
        data={
            "hourly_rate": 50.0,
            "active_subtask": "",
            "qualified_subtask_types": [],  # Empty list means no qualifications
        },
    )

    # Try to start subtask - this should raise an error
    with pytest.raises(Exception):
        start_subtask(project_name, "test_user", "subtask_1")
