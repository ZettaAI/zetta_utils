from typing import cast

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from .db.models import UserModel
from .types import User, UserUpdate


def get_user(
    db_session: Session,
    project_name: str,
    user_id: str,
) -> User:
    """
    Retrieve a user record from the database.

    :param db_session: SQLAlchemy session.
    :param project_name: The name of the project.
    :param user_id: The unique identifier of the user.
    :return: The user record.
    :raises KeyError: If the user does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    query = (
        select(UserModel)
        .where(UserModel.user_id == user_id)
        .where(UserModel.project_name == project_name)
    )
    try:
        result = db_session.execute(query).scalar_one()
        return cast(User, result.to_dict())
    except NoResultFound as exc:
        raise KeyError(f"User {user_id} not found") from exc


@typechecked
def create_user(
    db_session: Session,
    project_name: str,
    data: User,
) -> str:
    """
    Create a new user record in the database.

    :param db_session: SQLAlchemy session.
    :param project_name: The name of the project.
    :param data: The user data to create, must contain user_id, hourly_rate, and active_subtask.
    :return: The user_id of the created user.
    :raises ValueError: If the user data is invalid or user already exists.
    :raises RuntimeError: If the database operation fails.
    """
    # Check if user already exists
    existing_query = (
        select(UserModel)
        .where(UserModel.user_id == data["user_id"])
        .where(UserModel.project_name == project_name)
    )
    existing = db_session.execute(existing_query).scalar_one_or_none()
    if existing:
        raise ValueError(f"User {data['user_id']} already exists")

    # Create new user
    model = UserModel.from_dict(project_name, dict(data))
    db_session.add(model)
    db_session.commit()

    return data["user_id"]


@typechecked
def update_user(
    db_session: Session,
    project_name: str,
    user_id: str,
    data: UserUpdate,
) -> bool:
    """
    Update a user record in the database.

    :param db_session: SQLAlchemy session.
    :param project_name: The name of the project.
    :param user_id: The unique identifier of the user.
    :param data: The user data to update.
    :raises ValueError: If the update data is invalid.
    :raises KeyError: If the user does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    # Get current user
    query = (
        select(UserModel)
        .where(UserModel.user_id == user_id)
        .where(UserModel.project_name == project_name)
    )
    try:
        user = db_session.execute(query).scalar_one()
    except NoResultFound as exc:
        raise KeyError(f"User {user_id} not found") from exc

    # Apply updates generically
    for field, value in data.items():
        if hasattr(user, field):
            setattr(user, field, value)

    db_session.commit()
    return True
