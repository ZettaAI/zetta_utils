from typing import Literal, cast

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from zetta_utils.task_management.utils import generate_id_nonunique

from .db.models import TaskModel
from .db.session import get_session_context
from .types import Task, TaskUpdate

TaskStatus = Literal["pending_ingestion", "ingested", "fully_processed"]


def create_tasks_batch(
    *,
    project_name: str,
    tasks: list[Task],
    batch_size: int = 500,
    db_session: Session | None = None,
) -> list[str]:
    """
    Create multiple tasks in a single batch operation.

    :param project_name: The name of the project.
    :param tasks: List of tasks to create.
    :param batch_size: Maximum number of operations per batch (default 500).
    :param db_session: Database session to use (optional).
    :return: List of created task IDs.
    :raises ValueError: If a task already exists with different content.
    """
    with get_session_context(db_session) as session:
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
            for existing_task in session.execute(query).scalars():
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
                session.add(model)
                task_ids.append(task["task_id"])

        session.commit()

        # Return all task IDs (including existing ones that were identical)
        return [task["task_id"] for task in tasks]


@typechecked
def create_task(*, project_name: str, data: Task, db_session: Session | None = None) -> str:
    """
    Create a new task record in the database.

    :param project_name: The name of the project.
    :param data: The task data to create.
    :param db_session: Database session to use (optional).
    :return: The task_id of the created task.
    :raises ValueError: If the task data is invalid or task already exists.
    :raises RuntimeError: If the database operation fails.
    """
    # Use create_tasks_batch for maximum code reuse
    result = create_tasks_batch(project_name=project_name, tasks=[data], db_session=db_session)
    return result[0]


def get_task(*, project_name: str, task_id: str, db_session: Session | None = None) -> Task:
    """
    Get a task by ID.

    :param project_name: The name of the project.
    :param task_id: The ID of the task.
    :param db_session: Database session to use (optional).
    :return: The task data.
    :raises KeyError: If the task does not exist.
    """
    with get_session_context(db_session) as session:
        query = (
            select(TaskModel)
            .where(TaskModel.task_id == task_id)
            .where(TaskModel.project_name == project_name)
        )
        try:
            task = session.execute(query).scalar_one()
            return cast(Task, task.to_dict())
        except NoResultFound as exc:
            raise KeyError(f"Task {task_id} not found in project {project_name}") from exc


def update_task(
    *, project_name: str, task_id: str, data: TaskUpdate, db_session: Session | None = None
) -> bool:
    """
    Update a task record in the database.

    :param project_name: The name of the project.
    :param task_id: The unique identifier of the task.
    :param data: The task data to update.
    :param db_session: Database session to use (optional).
    :return: True on success.
    :raises ValueError: If the update data is invalid.
    :raises KeyError: If the task does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    with get_session_context(db_session) as session:
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
            task = session.execute(query).scalar_one()
        except NoResultFound as exc:
            raise KeyError(f"Task {task_id} not found in project {project_name}") from exc

        for field, value in data.items():
            if hasattr(task, field):
                setattr(task, field, value)

        session.commit()
        return True
