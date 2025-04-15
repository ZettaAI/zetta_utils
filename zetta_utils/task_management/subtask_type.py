from typing import Optional, Union

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from .db.models import SubtaskTypeModel
from .project import get_collection
from .types import SubtaskType


def get_subtask_type(
    db_session: Optional[Session] = None,
    project_name: str = "",
    subtask_type_id: str = "",
) -> SubtaskType:
    """
    Get a subtask type from the database.

    :param db_session: SQLAlchemy session.
    :param project_name: The project the subtask type belongs to.
    :param subtask_type_id: The unique identifier of the subtask type.
    :return: The subtask type data.
    :raises KeyError: If the subtask type does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    # Support both SQL and Firestore during migration
    if db_session is not None:
        try:
            query = (
                select(SubtaskTypeModel)
                .where(SubtaskTypeModel.subtask_type == subtask_type_id)
                .where(SubtaskTypeModel.project_name == project_name)
            )
            subtask_type = db_session.execute(query).scalar_one()
            return subtask_type.to_dict()
        except NoResultFound:
            raise KeyError(f"Subtask type not found: {subtask_type_id}")
    else:
        # Legacy Firestore path
        doc = get_collection(project_name, "subtask_types").document(subtask_type_id).get()
        if not doc.exists:
            raise KeyError(f"Subtask type not found: {subtask_type_id}")
        return doc.to_dict()


@typechecked
def create_subtask_type(
    db_session: Optional[Session] = None,
    project_name: str = "",
    data: Optional[SubtaskType] = None,
) -> str:
    """
    Create a new subtask type in the database.

    :param db_session: SQLAlchemy session.
    :param project_name: The project the subtask type belongs to.
    :param data: The subtask type data to create.
    :return: The subtask_type that was created.
    :raises ValueError: If the subtask type data is invalid or already exists.
    :raises RuntimeError: If the database operation fails.
    """
    # Support both SQL and Firestore during migration
    if data is None:
        raise ValueError("Subtask type data cannot be None")

    if db_session is not None:
        # Check if it already exists
        query = (
            select(SubtaskTypeModel)
            .where(SubtaskTypeModel.subtask_type == data["subtask_type"])
            .where(SubtaskTypeModel.project_name == project_name)
        )
        existing = db_session.execute(query).scalar_one_or_none()

        if existing:
            raise ValueError(f"Subtask type already exists: {data['subtask_type']}")

        # Create new subtask type
        model = SubtaskTypeModel.from_dict(project_name, data)
        db_session.add(model)
        db_session.commit()

        return data["subtask_type"]
    else:
        # Legacy Firestore path
        doc_ref = get_collection(project_name, "subtask_types").document(data["subtask_type"])

        if doc_ref.get().exists:
            raise ValueError(f"Subtask type already exists: {data['subtask_type']}")

        doc_ref.set(data)
        return data["subtask_type"]
