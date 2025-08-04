# pylint: disable=redefined-outer-name,unused-argument

import pytest

from zetta_utils.task_management.types import User, UserUpdate
from zetta_utils.task_management.user import create_user, get_user, update_user


def test_get_user_success(existing_user, project_name, db_session):
    result = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert result == existing_user


def test_get_user_not_found(clean_db, project_name, db_session):
    with pytest.raises(KeyError, match="User test_user not found"):
        get_user(project_name=project_name, user_id="test_user", db_session=db_session)


def test_create_user_success(project_name, clean_db, sample_user, db_session):
    result = create_user(project_name=project_name, data=sample_user, db_session=db_session)
    assert result == sample_user["user_id"]

    user = get_user(
        project_name=project_name, user_id=sample_user["user_id"], db_session=db_session
    )
    assert user == sample_user


def test_create_user_already_exists(project_name, clean_db, sample_user, db_session):
    # First creation
    create_user(project_name=project_name, data=sample_user, db_session=db_session)

    # Second creation with same data should succeed
    result = create_user(project_name=project_name, data=sample_user, db_session=db_session)
    assert result == sample_user["user_id"]

    # Creation with different data should raise ValueError
    different_user_data = dict(sample_user)
    different_user_data["hourly_rate"] = 99.0
    different_user = User(**different_user_data)  # type: ignore
    with pytest.raises(ValueError, match="already exists with different data"):
        create_user(project_name=project_name, data=different_user, db_session=db_session)


def test_update_user_success(project_name, existing_user, db_session):
    update_data = UserUpdate(hourly_rate=75.0)
    result = update_user(
        project_name=project_name, user_id="test_user", data=update_data, db_session=db_session
    )
    assert result is True

    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["hourly_rate"] == 75.0


def test_update_user_not_found(project_name, clean_db, db_session):
    update_data = UserUpdate(hourly_rate=75.0)
    with pytest.raises(KeyError, match="User unknown_user not found"):
        update_user(
            project_name=project_name,
            user_id="unknown_user",
            data=update_data,
            db_session=db_session,
        )


def test_create_user_with_qualifications(clean_db, project_name, db_session, sample_user):
    result = create_user(project_name=project_name, data=sample_user, db_session=db_session)
    assert result == sample_user["user_id"]

    user = get_user(
        project_name=project_name, user_id=sample_user["user_id"], db_session=db_session
    )
    assert user["qualified_task_types"] == ["segmentation_proofread"]


def test_update_user_qualified_task_types(project_name, existing_user, db_session):
    new_qualifications = ["segmentation_proofread", "segmentation_verify"]
    update_data = UserUpdate(qualified_task_types=new_qualifications)

    result = update_user(
        project_name=project_name, user_id="test_user", data=update_data, db_session=db_session
    )
    assert result is True

    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["qualified_task_types"] == new_qualifications


def test_update_user_active_task(project_name, existing_user, db_session):
    update_data = UserUpdate(active_task="task_123")

    result = update_user(
        project_name=project_name, user_id="test_user", data=update_data, db_session=db_session
    )
    assert result is True

    user = get_user(project_name=project_name, user_id="test_user", db_session=db_session)
    assert user["active_task"] == "task_123"
