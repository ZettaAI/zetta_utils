from typing import Optional

from google.cloud import firestore
from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from zetta_utils.task_management.helpers import get_transaction

from .db.models import UserModel
from .project import get_collection
from .types import User, UserUpdate


def get_user(
    db_session: Optional[Session] = None,
    project_name: str = "",
    user_id: str = "",
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
    if db_session is not None:
        try:
            query = (
                select(UserModel)
                .where(UserModel.user_id == user_id)
                .where(UserModel.project_name == project_name)
            )
            user = db_session.execute(query).scalar_one()
            return user.to_dict()
        except NoResultFound:
            raise KeyError(f"User {user_id} not found")
    else:
        # Legacy Firestore path
        collection = get_collection(project_name, "users")
        doc = collection.document(user_id).get()
        if not doc.exists:
            raise KeyError(f"User {user_id} not found")
        return doc.to_dict()


@typechecked
def create_user(
    db_session: Optional[Session] = None,
    project_name: str = "",
    data: Optional[User] = None,
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
    if data is None:
        raise ValueError("User data cannot be None")

    if db_session is not None:
        # Check if user already exists
        query = (
            select(UserModel)
            .where(UserModel.user_id == data["user_id"])
            .where(UserModel.project_name == project_name)
        )
        existing = db_session.execute(query).scalar_one_or_none()

        if existing:
            raise ValueError(f"User {data['user_id']} already exists")

        # Create new user
        model = UserModel.from_dict(project_name, data)
        db_session.add(model)
        db_session.commit()

        return data["user_id"]
    else:
        # Legacy Firestore path
        collection = get_collection(project_name, "users")
        doc_ref = collection.document(data["user_id"])

        @firestore.transactional
        def create_in_transaction(transaction):
            doc = doc_ref.get(transaction=transaction)
            if doc.exists:
                raise ValueError(f"User {data['user_id']} already exists")
            transaction.set(doc_ref, data)
            return data["user_id"]

        return create_in_transaction(get_transaction())


def update_user(
    db_session: Optional[Session] = None,
    project_name: str = "",
    user_id: str = "",
    data: Optional[UserUpdate] = None,
) -> bool:
    """
    Update a user record in the database.

    :param db_session: SQLAlchemy session.
    :param project_name: The name of the project.
    :param user_id: The unique identifier of the user.
    :param data: The user data to update.
    :return: True on success.
    :raises ValueError: If the update data is invalid.
    :raises KeyError: If the user does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    if data is None:
        raise ValueError("Update data cannot be None")

    if db_session is not None:
        # Get the user to update
        query = (
            select(UserModel)
            .where(UserModel.user_id == user_id)
            .where(UserModel.project_name == project_name)
        )

        try:
            user = db_session.execute(query).scalar_one()

            # Update fields
            if "hourly_rate" in data:
                user.hourly_rate = data["hourly_rate"]
            if "active_subtask" in data:
                user.active_subtask = data["active_subtask"]
            if "qualified_subtask_types" in data:
                user.qualified_subtask_types = data["qualified_subtask_types"]

            db_session.commit()
            return True
        except NoResultFound:
            raise KeyError(f"User {user_id} not found")
    else:
        # Legacy Firestore path
        collection = get_collection(project_name, "users")
        doc_ref = collection.document(user_id)

        @firestore.transactional
        def update_in_transaction(transaction):
            doc = doc_ref.get(transaction=transaction)
            if not doc.exists:
                raise KeyError(f"User {user_id} not found")

            transaction.update(doc_ref, data)
            return True

        return update_in_transaction(get_transaction())
