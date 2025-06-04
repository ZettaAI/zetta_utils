from typing import cast

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from .db.models import DependencyModel, SubtaskModel
from .types import Dependency, DependencyUpdate


def get_dependency(db_session: Session, project_name: str, dependency_id: str) -> Dependency:
    """
    Retrieve a dependency record from the database.

    :param db_session: Database session to use.
    :param project_name: The name of the project.
    :param dependency_id: The unique identifier of the dependency.
    :return: The dependency record.
    :raises KeyError: If the dependency does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    query = (
        select(DependencyModel)
        .where(DependencyModel.dependency_id == dependency_id)
        .where(DependencyModel.project_name == project_name)
    )
    try:
        dependency = db_session.execute(query).scalar_one()
        return cast(Dependency, dependency.to_dict())
    except NoResultFound as exc:
        raise KeyError(f"Dependency {dependency_id} not found") from exc


@typechecked
def create_dependency(db_session: Session, project_name: str, data: Dependency) -> str:
    """
    Create a new dependency record in the database.

    :param db_session: Database session to use.
    :param project_name: The name of the project.
    :param data: The dependency data to create.
    :return: The dependency_id of the created dependency.
    :raises ValueError: If the dependency data is invalid or already exists.
    :raises RuntimeError: If the database operation fails.
    """
    if data["subtask_id"] == data["dependent_on_subtask_id"]:
        raise ValueError("Subtask cannot depend on itself")

    # Check if dependency already exists
    query = (
        select(DependencyModel)
        .where(DependencyModel.dependency_id == data["dependency_id"])
        .where(DependencyModel.project_name == project_name)
    )
    existing = db_session.execute(query).scalar_one_or_none()

    if existing:
        raise ValueError(f"Dependency {data['dependency_id']} already exists")

    # Verify both subtasks exist
    subtask_query = (
        select(SubtaskModel)
        .where(SubtaskModel.subtask_id == data["subtask_id"])
        .where(SubtaskModel.project_name == project_name)
    )
    subtask = db_session.execute(subtask_query).scalar_one_or_none()
    if not subtask:
        raise ValueError(f"Subtask {data['subtask_id']} not found")

    dependent_on_query = (
        select(SubtaskModel)
        .where(SubtaskModel.subtask_id == data["dependent_on_subtask_id"])
        .where(SubtaskModel.project_name == project_name)
    )
    dependent_on = db_session.execute(dependent_on_query).scalar_one_or_none()
    if not dependent_on:
        raise ValueError(f"Subtask {data['dependent_on_subtask_id']} not found")

    # Create new dependency
    dependency_data = {**data}
    dependency_data.setdefault("is_satisfied", False)

    model = DependencyModel.from_dict(project_name, dependency_data)
    db_session.add(model)
    db_session.commit()

    return data["dependency_id"]


@typechecked
def update_dependency(
    db_session: Session, project_name: str, dependency_id: str, data: DependencyUpdate
) -> bool:
    """
    Update a dependency record in the database.

    :param db_session: Database session to use.
    :param project_name: The name of the project.
    :param dependency_id: The unique identifier of the dependency.
    :param data: The dependency data to update.
    :return: True on success.
    :raises ValueError: If the update data is invalid.
    :raises KeyError: If the dependency does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    query = (
        select(DependencyModel)
        .where(DependencyModel.dependency_id == dependency_id)
        .where(DependencyModel.project_name == project_name)
    )

    try:
        dependency = db_session.execute(query).scalar_one()
    except NoResultFound as exc:
        raise KeyError(f"Dependency {dependency_id} not found") from exc

    # Apply updates generically
    for field, value in data.items():
        if hasattr(dependency, field):
            setattr(dependency, field, value)

    db_session.commit()
    return True


def get_dependencies_for_subtask(
    db_session: Session, project_name: str, subtask_id: str
) -> list[Dependency]:
    """
    Get all dependencies for a specific subtask.

    :param db_session: Database session to use.
    :param project_name: The name of the project.
    :param subtask_id: The subtask ID to get dependencies for.
    :return: List of dependencies for the subtask.
    """
    query = (
        select(DependencyModel)
        .where(DependencyModel.subtask_id == subtask_id)
        .where(DependencyModel.project_name == project_name)
    )
    dependencies = db_session.execute(query).scalars().all()
    return [cast(Dependency, dep.to_dict()) for dep in dependencies]


def get_dependencies_depending_on_subtask(
    db_session: Session, project_name: str, dependent_on_subtask_id: str
) -> list[Dependency]:
    """
    Get all dependencies that depend on a specific subtask.

    :param db_session: Database session to use.
    :param project_name: The name of the project.
    :param dependent_on_subtask_id: The subtask ID that others depend on.
    :return: List of dependencies that depend on the specified subtask.
    """
    query = (
        select(DependencyModel)
        .where(DependencyModel.dependent_on_subtask_id == dependent_on_subtask_id)
        .where(DependencyModel.project_name == project_name)
    )
    dependencies = db_session.execute(query).scalars().all()
    return [cast(Dependency, dep.to_dict()) for dep in dependencies]


def get_unsatisfied_dependencies_for_subtask(
    db_session: Session, project_name: str, subtask_id: str
) -> list[Dependency]:
    """
    Get all unsatisfied dependencies for a specific subtask.

    :param db_session: Database session to use.
    :param project_name: The name of the project.
    :param subtask_id: The subtask ID to get unsatisfied dependencies for.
    :return: List of unsatisfied dependencies for the subtask.
    """
    query = (
        select(DependencyModel)
        .where(DependencyModel.subtask_id == subtask_id)
        .where(DependencyModel.project_name == project_name)
        .where(DependencyModel.is_satisfied == False)  # pylint: disable=singleton-comparison
    )
    dependencies = db_session.execute(query).scalars().all()
    return [cast(Dependency, dep.to_dict()) for dep in dependencies]
