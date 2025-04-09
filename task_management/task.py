from typing import Literal

from google.cloud import firestore
from typeguard import typechecked

from zetta_utils.task_management.helpers import get_transaction

from .project import get_collection, get_firestore_client
from .types import Task, TaskUpdate

TaskStatus = Literal["pending_ingestion", "ingested", "fully_processed"]


@typechecked
def create_task(project_name: str, data: Task) -> str:
    """
    Create a new task record in the project's Task collection.

    :param project_name: The name of the project.
    :param data: The task data to create.
    :return: The task_id of the created task.
    :raises ValueError: If the task data is invalid or task already exists.
    :raises RuntimeError: If the Firestore transaction fails.
    """
    task_ids = create_tasks_batch(project_name, [data], batch_size=1)
    return task_ids[0]


def get_task(project_name: str, task_id: str) -> Task:
    """
    Get a task by ID.

    :param project_name: The name of the project.
    :param task_id: The ID of the task.
    :return: The task data.
    :raises KeyError: If the task does not exist.
    """
    collection = get_collection(project_name, "tasks")
    doc = collection.document(task_id).get()
    if not doc.exists:
        raise KeyError(f"Task {task_id} not found")
    return doc.to_dict()


def update_task(project_name: str, task_id: str, data: TaskUpdate) -> bool:
    """
    Update a task record in the project's Task collection.

    :param project_name: The name of the project.
    :param task_id: The unique identifier of the task.
    :param data: The task data to update.
    :return: True on success.
    :raises ValueError: If the update data is invalid.
    :raises KeyError: If the task does not exist.
    :raises RuntimeError: If the Firestore transaction fails.
    """
    # Validate status if it's being updated
    if "status" in data and data["status"] not in [
        "pending_ingestion",
        "ingested",
        "fully_processed",
    ]:
        raise ValueError("Invalid status value")

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


def create_tasks_batch(project_name: str, tasks: list[Task], batch_size: int = 500) -> list[str]:
    """
    Create multiple tasks in a single batch write.

    :param project_name: The name of the project.
    :param tasks: List of tasks to create.
    :param batch_size: Maximum number of operations per batch (default 500).
    :return: List of created task IDs.
    """
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
            batch.set(doc_ref, task)

        batch.commit()

    return task_ids
