from typing import Literal, cast

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from zetta_utils.task_management.utils import generate_id_nonunique

from .db.models import TaskModel
from .types import Task, TaskUpdate

TaskStatus = Literal["pending_ingestion", "ingested", "fully_processed"]


def create_tasks_batch(
    db_session: Session, project_name: str, tasks: list[Task], batch_size: int = 500
) -> list[str]:
    """
    Create multiple tasks in a single batch operation.

    :param db_session: Database session to use.
    :param project_name: The name of the project.
    :param tasks: List of tasks to create.
    :param batch_size: Maximum number of operations per batch (default 500).
    :return: List of created task IDs.
    :raises ValueError: If a task already exists with different content.
    """
    task_ids = []
    all_task_ids = [task["task_id"] for task in tasks]

    # Check for existing tasks
    existing_tasks = {}
    if all_task_ids:
        query = (
            select(TaskModel)
            .where(TaskModel.task_id.in_(all_task_ids))
            .where(TaskModel.project_name == project_name)
        )
        for existing_task in db_session.execute(query).scalars():
            existing_tasks[str(existing_task.task_id)] = existing_task.to_dict()

    # Create a lookup dict for incoming tasks
    incoming_tasks_dict = {task["task_id"]: task for task in tasks}

    # Check for conflicts (existing tasks with different content)
    for task_id, existing_data in existing_tasks.items():
        incoming_data = incoming_tasks_dict[task_id]

        # Check if tasks are identical
        if existing_data != incoming_data:
            raise ValueError(
                f"Task {task_id} already exists with different content. "
                f"Existing: {existing_data}, Incoming: {incoming_data}"
            )

    # Filter out tasks that already exist
    filtered_tasks = [task for task in tasks if task["task_id"] not in existing_tasks]

    # Create new tasks in batches
    for i in range(0, len(filtered_tasks), batch_size):
        chunk = filtered_tasks[i : i + batch_size]

        for task in chunk:
            task_data = {**task, "id_nonunique": generate_id_nonunique()}
            model = TaskModel.from_dict(project_name, task_data)
            db_session.add(model)
            task_ids.append(task["task_id"])

    db_session.commit()

    # Return all task IDs (including existing ones that were identical)
    return [task["task_id"] for task in tasks]


@typechecked
def create_task(db_session: Session, project_name: str, data: Task) -> str:
    """
    Create a new task record in the database.

    :param db_session: Database session to use.
    :param project_name: The name of the project.
    :param data: The task data to create.
    :return: The task_id of the created task.
    :raises ValueError: If the task data is invalid or task already exists.
    :raises RuntimeError: If the database operation fails.
    """
    # Use create_tasks_batch for maximum code reuse
    result = create_tasks_batch(db_session, project_name, [data])
    return result[0]


def get_task(db_session: Session, project_name: str, task_id: str) -> Task:
    """
    Get a task by ID.

    :param db_session: Database session to use.
    :param project_name: The name of the project.
    :param task_id: The ID of the task.
    :return: The task data.
    :raises KeyError: If the task does not exist.
    """
    query = (
        select(TaskModel)
        .where(TaskModel.task_id == task_id)
        .where(TaskModel.project_name == project_name)
    )
    try:
        task = db_session.execute(query).scalar_one()
        return cast(Task, task.to_dict())
    except NoResultFound as exc:
        raise KeyError(f"Task {task_id} not found in project {project_name}") from exc


def update_task(db_session: Session, project_name: str, task_id: str, data: TaskUpdate) -> bool:
    """
    Update a task record in the database.

    :param db_session: Database session to use.
    :param project_name: The name of the project.
    :param task_id: The unique identifier of the task.
    :param data: The task data to update.
    :return: True on success.
    :raises ValueError: If the update data is invalid.
    :raises KeyError: If the task does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    # Validate status if it's being updated
    if "status" in data and data["status"] not in [
        "pending_ingestion",
        "ingested",
        "fully_processed",
    ]:
        raise ValueError("Invalid status value")

    query = (
        select(TaskModel)
        .where(TaskModel.task_id == task_id)
        .where(TaskModel.project_name == project_name)
    )

    try:
        task = db_session.execute(query).scalar_one()
    except NoResultFound as exc:
        raise KeyError(f"Task {task_id} not found in project {project_name}") from exc

    for field, value in data.items():
        if hasattr(task, field):
            setattr(task, field, value)

    db_session.commit()
    return True
