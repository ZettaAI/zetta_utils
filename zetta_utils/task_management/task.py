from typing import List, Literal, Optional

from google.cloud import firestore
from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from zetta_utils.task_management.helpers import get_transaction
from zetta_utils.task_management.utils import generate_id_nonunique

from .db.models import TaskModel
from .project import get_collection, get_firestore_client
from .types import Task, TaskUpdate

TaskStatus = Literal["pending_ingestion", "ingested", "fully_processed"]


@typechecked
def create_task(
    db_session: Optional[Session] = None,
    project_name: str = "",
    data: Optional[Task] = None,
) -> str:
    """
    Create a new task record in the database.

    :param db_session: SQLAlchemy session.
    :param project_name: The name of the project.
    :param data: The task data to create.
    :return: The task_id of the created task.
    :raises ValueError: If the task data is invalid or task already exists.
    :raises RuntimeError: If the database operation fails.
    """
    if data is None:
        raise ValueError("Task data cannot be None")

    if db_session is not None:
        # Check if the task already exists
        query = (
            select(TaskModel)
            .where(TaskModel.task_id == data["task_id"])
            .where(TaskModel.project_name == project_name)
        )
        existing = db_session.execute(query).scalar_one_or_none()

        if existing:
            raise ValueError(f"Task {data['task_id']} already exists")

        # Create new task
        id_nonunique = generate_id_nonunique()
        model = TaskModel.from_dict(project_name, {**data, "_id_nonunique": id_nonunique})
        db_session.add(model)
        db_session.commit()

        return data["task_id"]
    else:
        # Legacy Firestore path
        task_ids = create_tasks_batch(project_name, [data], batch_size=1)
        return task_ids[0]


def get_task(
    db_session: Optional[Session] = None,
    project_name: str = "",
    task_id: str = "",
) -> Task:
    """
    Get a task by ID.

    :param db_session: SQLAlchemy session.
    :param project_name: The name of the project.
    :param task_id: The ID of the task.
    :return: The task data.
    :raises KeyError: If the task does not exist.
    """
    if db_session is not None:
        try:
            query = (
                select(TaskModel)
                .where(TaskModel.task_id == task_id)
                .where(TaskModel.project_name == project_name)
            )
            task = db_session.execute(query).scalar_one()
            return task.to_dict()
        except NoResultFound:
            raise KeyError(f"Task {task_id} not found")
    else:
        # Legacy Firestore path
        collection = get_collection(project_name, "tasks")
        doc = collection.document(task_id).get()
        if not doc.exists:
            raise KeyError(f"Task {task_id} not found")

        result = doc.to_dict()
        del result["_id_nonunique"]
        return result


def update_task(
    db_session: Optional[Session] = None,
    project_name: str = "",
    task_id: str = "",
    data: Optional[TaskUpdate] = None,
) -> bool:
    """
    Update a task record in the database.

    :param db_session: SQLAlchemy session.
    :param project_name: The name of the project.
    :param task_id: The unique identifier of the task.
    :param data: The task data to update.
    :return: True on success.
    :raises ValueError: If the update data is invalid.
    :raises KeyError: If the task does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    if data is None:
        raise ValueError("Update data cannot be None")

    # Validate status if it's being updated
    if "status" in data and data["status"] not in [
        "pending_ingestion",
        "ingested",
        "fully_processed",
    ]:
        raise ValueError("Invalid status value")

    if db_session is not None:
        # Get the task to update
        query = (
            select(TaskModel)
            .where(TaskModel.task_id == task_id)
            .where(TaskModel.project_name == project_name)
        )

        try:
            task = db_session.execute(query).scalar_one()

            # Update fields
            if "status" in data:
                task.status = data["status"]
            if "batch_id" in data:
                task.batch_id = data["batch_id"]
            if "ng_state" in data:
                task.ng_state = data["ng_state"]
            if "task_type" in data:
                task.task_type = data["task_type"]

            db_session.commit()
            return True
        except NoResultFound:
            raise KeyError(f"Task {task_id} not found")
    else:
        # Legacy Firestore path
        collection = get_collection(project_name, "tasks")
        doc_ref = collection.document(task_id)

        @firestore.transactional
        def update_in_transaction(transaction):
            doc = doc_ref.get(transaction=transaction)
            if not doc.exists:
                raise KeyError(f"Task {task_id} not found")

            transaction.update(doc_ref, data)
            return True

        return update_in_transaction(get_transaction())


def create_tasks_batch(
    db_session: Optional[Session] = None,
    project_name: str = "",
    tasks: Optional[List[Task]] = None,
    batch_size: int = 500,
) -> list[str]:
    """
    Create multiple tasks in a single batch.

    :param db_session: SQLAlchemy session.
    :param project_name: The name of the project.
    :param tasks: List of tasks to create.
    :param batch_size: Maximum number of operations per batch (default 500).
    :return: List of created task IDs.
    """
    if tasks is None:
        raise ValueError("Tasks list cannot be None")

    if db_session is not None:
        task_ids = []

        # Process tasks in chunks to match Firestore behavior
        for i in range(0, len(tasks), batch_size):
            chunk = tasks[i : i + batch_size]

            for task in chunk:
                # Check if task exists
                query = (
                    select(TaskModel)
                    .where(TaskModel.task_id == task["task_id"])
                    .where(TaskModel.project_name == project_name)
                )
                existing = db_session.execute(query).scalar_one_or_none()

                if existing:
                    raise ValueError(f"Task {task['task_id']} already exists")

                # Create task
                id_nonunique = generate_id_nonunique()
                model = TaskModel.from_dict(project_name, {**task, "_id_nonunique": id_nonunique})
                db_session.add(model)
                task_ids.append(task["task_id"])

            # Commit after each batch
            print("committing...")
            db_session.commit()
            print("committed...")

        return task_ids
    else:
        # Legacy Firestore path
        client = get_firestore_client()
        task_ids = []

        # Process tasks in batches
        for i in range(0, len(tasks), batch_size):
            batch = client.batch()
            chunk = tasks[i : i + batch_size]

            for task in chunk:
                doc_ref = get_collection(project_name, "tasks").document(task["task_id"])
                # Check if task exists before adding to batch
                if doc_ref.get().exists:
                    raise ValueError(f"Task {task['task_id']} already exists")
                task_ids.append(task["task_id"])
                batch.set(doc_ref, {**task, "_id_nonunique": generate_id_nonunique()})
            print("committing...")
            batch.commit()
            print("committed...")
        return task_ids
