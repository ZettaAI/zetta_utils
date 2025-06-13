from typing import cast

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from .db.models import TaskTypeModel
from .db.session import get_session_context
from .types import TaskType


def get_task_type(
    *,
    project_name: str,
    task_type: str,
    db_session: Session | None = None,
) -> TaskType:
    """
    Retrieve a task type record from the database.

    :param project_name: The name of the project.
    :param task_type: The unique identifier of the task type.
    :param db_session: SQLAlchemy session (optional).
    :return: The task type record.
    :raises KeyError: If the task type does not exist.
    """
    with get_session_context(db_session) as session:
        query = (
            select(TaskTypeModel)
            .where(TaskTypeModel.task_type == task_type)
            .where(TaskTypeModel.project_name == project_name)
        )
        try:
            result = session.execute(query).scalar_one()
            return cast(TaskType, result.to_dict())
        except NoResultFound as exc:
            raise KeyError(
                f"TaskType {task_type} not found in project {project_name}"
            ) from exc


@typechecked
def create_task_type(
    *,
    project_name: str,
    data: TaskType,
    db_session: Session | None = None,
) -> str:
    """
    Create a new task type record in the database.

    :param project_name: The name of the project.
    :param data: The task type data to create.
    :param db_session: SQLAlchemy session (optional).
    :return: The task_type identifier of the created record.
    :raises ValueError: If the task type already exists.
    """
    with get_session_context(db_session) as session:
        query = (
            select(TaskTypeModel)
            .where(TaskTypeModel.task_type == data["task_type"])
            .where(TaskTypeModel.project_name == project_name)
        )
        existing = session.execute(query).scalar_one_or_none()

        if existing:
            raise ValueError(
                f"TaskType {data['task_type']} already exists in project {project_name}"
            )

        model = TaskTypeModel.from_dict(project_name, dict(data))
        session.add(model)
        session.commit()

        return data["task_type"]
