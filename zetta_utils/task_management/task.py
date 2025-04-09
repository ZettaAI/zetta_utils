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


@typechecked
def update_task(project_name: str, task_id: str, data: TaskUpdate) -> bool:
    """
    Update a task record in the project's Task collection.

    :param project_name: The name of the project.
    :param task_id: The unique identifier of the task.
    :param data: The task data to update.
    :return: True if the update was successful.
    :raises KeyError: If the task does not exist.
    :raises ValueError: If the task data is invalid.
    :raises RuntimeError: If the Firestore transaction fails.
    """
    client = get_firestore_client()
    transaction = get_transaction()
    task_ref = get_collection(project_name, "tasks").document(task_id)

    @firestore.transactional
    def update_in_transaction(transaction):
        doc = task_ref.get(transaction=transaction)
        if not doc.exists:
            raise KeyError(f"Task {task_id} not found")

        # Validate status if it's being updated
        if "status" in data and data["status"] not in [
            "pending_ingestion",
            "ingested",
            "fully_processed",
        ]:
            raise ValueError(f"Invalid status value: {data['status']}")

        transaction.update(task_ref, data)
        return True

    return update_in_transaction(transaction)


def create_tasks_batch(project_name: str, tasks: list["Task"], batch_size: int = 500) -> list[str]:
    """
    Create a batch of tasks in the project's Task collection.

    :param project_name: The name of the project.
    :param tasks: The list of task data to create.
    :param batch_size: The maximum number of tasks to create in a single batch.
    :return: The list of task_ids of the created tasks.
    :raises ValueError: If any task data is invalid or a task already exists.
    :raises RuntimeError: If the Firestore transaction fails.
    """
    client = get_firestore_client()
    collection = get_collection(project_name, "tasks")
    task_ids = []

    # Process in batches to avoid Firestore limits
    for i in range(0, len(tasks), batch_size):
        batch = client.batch()
        batch_tasks = tasks[i : i + batch_size]
        batch_task_ids = []

        for task in batch_tasks:
            task_id = task["task_id"]

            # Check if task already exists
            doc_ref = collection.document(task_id)
            if doc_ref.get().exists:
                raise ValueError(f"Task {task_id} already exists")

            batch_task_ids.append(task_id)
            batch.set(doc_ref, task)

        batch.commit()
        task_ids.extend(batch_task_ids)

    return task_ids
