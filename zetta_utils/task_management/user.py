from typing import cast

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from .db.models import UserModel
from .db.session import get_session_context
from .types import User, UserUpdate


def get_user(
    *,
    project_name: str,
    user_id: str,
    db_session: Session | None = None,
) -> User:
    """
    Retrieve a user record from the database.

    :param project_name: The name of the project.
    :param user_id: The unique identifier of the user.
    :param db_session: SQLAlchemy session (optional).
    :return: The user record.
    :raises KeyError: If the user does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    with get_session_context(db_session) as session:
        query = (
            select(UserModel)
            .where(UserModel.user_id == user_id)
            .where(UserModel.project_name == project_name)
        )
        try:
            result = session.execute(query).scalar_one()
            return cast(User, result.to_dict())
        except NoResultFound as exc:
            raise KeyError(f"User {user_id} not found") from exc


@typechecked
def create_user(
    *,
    project_name: str,
    data: User,
    db_session: Session | None = None,
) -> str:
    """
    Create a new user record in the database.

    :param project_name: The name of the project.
    :param data: The user data to create, must contain user_id, hourly_rate, and active_subtask.
    :param db_session: SQLAlchemy session (optional).
    :return: The user_id of the created user.
    :raises ValueError: If the user data is invalid or user already exists with different data.
    :raises RuntimeError: If the database operation fails.
    """
    with get_session_context(db_session) as session:
        # Check if user already exists
        existing_query = (
            select(UserModel)
            .where(UserModel.user_id == data["user_id"])
            .where(UserModel.project_name == project_name)
        )
        existing = session.execute(existing_query).scalar_one_or_none()
        if existing:
            # Check if the data is the same (idempotent operation)
            existing_data = existing.to_dict()
            if existing_data == dict(data):
                # Same data, return success (idempotent)
                return data["user_id"]
            else:
                # Different data, raise error
                raise ValueError(f"User {data['user_id']} already exists with different data")

        # Create new user
        model = UserModel.from_dict(project_name, dict(data))
        session.add(model)
        session.commit()

        return data["user_id"]


@typechecked
def update_user(
    *,
    project_name: str,
    user_id: str,
    data: UserUpdate,
    db_session: Session | None = None,
) -> bool:
    """
    Update a user record in the database.

    :param project_name: The name of the project.
    :param user_id: The unique identifier of the user.
    :param data: The user data to update.
    :param db_session: SQLAlchemy session (optional).
    :raises ValueError: If the update data is invalid.
    :raises KeyError: If the user does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    with get_session_context(db_session) as session:
        # Get current user
        query = (
            select(UserModel)
            .where(UserModel.user_id == user_id)
            .where(UserModel.project_name == project_name)
        )
        try:
            user = session.execute(query).scalar_one()
        except NoResultFound as exc:
            raise KeyError(f"User {user_id} not found") from exc

        # Apply updates generically
        for field, value in data.items():
            if hasattr(user, field):
                setattr(user, field, value)

        session.commit()
        return True
