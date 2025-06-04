from typing import cast

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from .db.models import SubtaskTypeModel
from .types import SubtaskType


def get_subtask_type(
    db_session: Session,
    project_name: str,
    subtask_type: str,
) -> SubtaskType:
    """
    Retrieve a subtask type record from the database.

    :param db_session: SQLAlchemy session.
    :param project_name: The name of the project.
    :param subtask_type: The unique identifier of the subtask type.
    :return: The subtask type record.
    :raises KeyError: If the subtask type does not exist.
    """
    query = (
        select(SubtaskTypeModel)
        .where(SubtaskTypeModel.subtask_type == subtask_type)
        .where(SubtaskTypeModel.project_name == project_name)
    )
    try:
        result = db_session.execute(query).scalar_one()
        return cast(SubtaskType, result.to_dict())
    except NoResultFound as exc:
        raise KeyError(f"SubtaskType {subtask_type} not found in project {project_name}") from exc


@typechecked
def create_subtask_type(
    db_session: Session,
    project_name: str,
    data: SubtaskType,
) -> str:
    """
    Create a new subtask type record in the database.

    :param db_session: SQLAlchemy session.
    :param project_name: The name of the project.
    :param data: The subtask type data to create.
    :return: The subtask_type identifier of the created record.
    :raises ValueError: If the subtask type already exists.
    """
    query = (
        select(SubtaskTypeModel)
        .where(SubtaskTypeModel.subtask_type == data["subtask_type"])
        .where(SubtaskTypeModel.project_name == project_name)
    )
    existing = db_session.execute(query).scalar_one_or_none()

    if existing:
        raise ValueError(
            f"SubtaskType {data['subtask_type']} already exists in project {project_name}"
        )

    model = SubtaskTypeModel.from_dict(project_name, dict(data))
    db_session.add(model)
    db_session.commit()

    return data["subtask_type"]
