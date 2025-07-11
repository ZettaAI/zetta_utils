from typing import cast

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from .db.models import DependencyModel, TaskModel
from .db.session import get_session_context
from .types import Dependency, DependencyUpdate


def get_dependency(
    *, project_name: str, dependency_id: str, db_session: Session | None = None
) -> Dependency:
    """
    Retrieve a dependency record from the database.

    :param project_name: The name of the project.
    :param dependency_id: The unique identifier of the dependency.
    :param db_session: Database session to use (optional).
    :return: The dependency record.
    :raises KeyError: If the dependency does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    with get_session_context(db_session) as session:
        query = (
            select(DependencyModel)
            .where(DependencyModel.dependency_id == dependency_id)
            .where(DependencyModel.project_name == project_name)
        )
        try:
            dependency = session.execute(query).scalar_one()
            return cast(Dependency, dependency.to_dict())
        except NoResultFound as exc:
            raise KeyError(f"Dependency {dependency_id} not found") from exc


@typechecked
def create_dependency(
    *, project_name: str, data: Dependency, db_session: Session | None = None
) -> str:
    """
    Create a new dependency record in the database.

    :param project_name: The name of the project.
    :param data: The dependency data to create.
    :param db_session: Database session to use (optional).
    :return: The dependency_id of the created dependency.
    :raises ValueError: If the dependency data is invalid or already exists.
    :raises RuntimeError: If the database operation fails.
    """
    with get_session_context(db_session) as session:
        if data["task_id"] == data["dependent_on_task_id"]:
            raise ValueError("Task cannot depend on itself")

        # Check if dependency already exists
        query = (
            select(DependencyModel)
            .where(DependencyModel.dependency_id == data["dependency_id"])
            .where(DependencyModel.project_name == project_name)
        )
        existing = session.execute(query).scalar_one_or_none()

        if existing:
            raise ValueError(f"Dependency {data['dependency_id']} already exists")

        # Verify both tasks exist
        task_query = (
            select(TaskModel)
            .where(TaskModel.task_id == data["task_id"])
            .where(TaskModel.project_name == project_name)
        )
        task = session.execute(task_query).scalar_one_or_none()
        if not task:
            raise ValueError(f"Task {data['task_id']} not found")

        dependent_on_query = (
            select(TaskModel)
            .where(TaskModel.task_id == data["dependent_on_task_id"])
            .where(TaskModel.project_name == project_name)
        )
        dependent_on = session.execute(dependent_on_query).scalar_one_or_none()
        if not dependent_on:
            raise ValueError(f"Task {data['dependent_on_task_id']} not found")

        # Create new dependency
        dependency_data = {**data}
        dependency_data.setdefault("is_satisfied", False)

        model = DependencyModel.from_dict(project_name, dependency_data)
        session.add(model)
        session.commit()

        return data["dependency_id"]


@typechecked
def update_dependency(
    *,
    project_name: str,
    dependency_id: str,
    data: DependencyUpdate,
    db_session: Session | None = None,
) -> bool:
    """
    Update a dependency record in the database.

    :param project_name: The name of the project.
    :param dependency_id: The unique identifier of the dependency.
    :param data: The dependency data to update.
    :param db_session: Database session to use (optional).
    :return: True on success.
    :raises ValueError: If the update data is invalid.
    :raises KeyError: If the dependency does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    with get_session_context(db_session) as session:
        query = (
            select(DependencyModel)
            .where(DependencyModel.dependency_id == dependency_id)
            .where(DependencyModel.project_name == project_name)
        )

        try:
            dependency = session.execute(query).scalar_one()
        except NoResultFound as exc:
            raise KeyError(f"Dependency {dependency_id} not found") from exc

        # Apply updates generically
        for field, value in data.items():
            if hasattr(dependency, field):
                setattr(dependency, field, value)

        session.commit()
        return True


def get_dependencies_for_task(
    *, project_name: str, task_id: str, db_session: Session | None = None
) -> list[Dependency]:
    """
    Get all dependencies for a specific task.

    :param project_name: The name of the project.
    :param task_id: The task ID to get dependencies for.
    :param db_session: Database session to use (optional).
    :return: List of dependencies for the task.
    """
    with get_session_context(db_session) as session:
        query = (
            select(DependencyModel)
            .where(DependencyModel.task_id == task_id)
            .where(DependencyModel.project_name == project_name)
        )
        dependencies = session.execute(query).scalars().all()
        return [cast(Dependency, dep.to_dict()) for dep in dependencies]


def get_dependencies_depending_on_task(
    *, project_name: str, dependent_on_task_id: str, db_session: Session | None = None
) -> list[Dependency]:
    """
    Get all dependencies that depend on a specific task.

    :param project_name: The name of the project.
    :param dependent_on_task_id: The task ID that others depend on.
    :param db_session: Database session to use (optional).
    :return: List of dependencies that depend on the specified task.
    """
    with get_session_context(db_session) as session:
        query = (
            select(DependencyModel)
            .where(DependencyModel.dependent_on_task_id == dependent_on_task_id)
            .where(DependencyModel.project_name == project_name)
        )
        dependencies = session.execute(query).scalars().all()
        return [cast(Dependency, dep.to_dict()) for dep in dependencies]


def get_unsatisfied_dependencies_for_task(
    *, project_name: str, task_id: str, db_session: Session | None = None
) -> list[Dependency]:
    """
    Get all unsatisfied dependencies for a specific task.

    :param project_name: The name of the project.
    :param task_id: The task ID to get unsatisfied dependencies for.
    :param db_session: Database session to use (optional).
    :return: List of unsatisfied dependencies for the task.
    """
    with get_session_context(db_session) as session:
        query = (
            select(DependencyModel)
            .where(DependencyModel.task_id == task_id)
            .where(DependencyModel.project_name == project_name)
            .where(DependencyModel.is_satisfied == False)  # pylint: disable=singleton-comparison
        )
        dependencies = session.execute(query).scalars().all()
        return [cast(Dependency, dep.to_dict()) for dep in dependencies]
