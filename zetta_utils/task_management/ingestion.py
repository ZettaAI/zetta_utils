from typing import Literal

from google.cloud import firestore
from typeguard import typechecked

from .project import get_firestore_client
from .subtask_structure import create_subtask_structure


@typechecked
def ingest_task(
    project_name: str,
    task_id: str,
    subtask_structure: str,
    re_ingest: Literal["not_processed", "all"] | None = None,
    priority: int = 1,
) -> bool:
    """
    Ingest a task, changing its status from pending_ingestion to ingested.

    :param project_name: The name of the project.
    :param task_id: The ID of the task to ingest.
    :param subtask_structure: Name of subtask structure to create
    :param re_ingest: Controls re-ingestion behavior:
                      None - only ingest pending_ingestion tasks
                      "not_processed" - ingest pending_ingestion and ingested tasks
                      "all" - ingest all tasks including fully_processed ones
    :param priority: Priority for subtasks (default: 1)
    :return: True if the task was ingested, False otherwise.
    :raises ValueError: If the task is not in a valid status for ingestion.
    :raises KeyError: If the task does not exist.
    """
    return ingest_tasks(
        project_name=project_name,
        task_ids=[task_id],
        subtask_structure=subtask_structure,
        re_ingest=re_ingest,
        priority=priority,
        bundle_size=1,
    )


@typechecked
def ingest_tasks(
    project_name: str,
    task_ids: list[str],
    subtask_structure: str,
    re_ingest: Literal["not_processed", "all"] | None = None,
    priority: int = 1,
    bundle_size: int = 500,
) -> bool:
    """
    Ingest multiple tasks in bundles.

    :param project_name: The name of the project.
    :param task_ids: List of task IDs to ingest.
    :param subtask_structure: Name of subtask structure to create
    :param re_ingest: Controls re-ingestion behavior:
                      None - only ingest pending_ingestion tasks
                      "not_processed" - ingest pending_ingestion and ingested tasks
                      "all" - ingest all tasks including fully_processed ones
    :param priority: Priority for subtasks (default: 1)
    :param bundle_size: Maximum number of operations per atch (default 500)
    :return: True if all tasks with appropriate status were ingested, False otherwise.
    :raises ValueError: If any task is not in a valid status for ingestion.
    :raises KeyError: If any task does not exist.
    """
    client = get_firestore_client()
    task_refs = [
        client.collection(f"{project_name}_tasks").document(task_id) for task_id in task_ids
    ]
    task_docs = list(client.get_all(task_refs))
    tasks = {doc.id: doc.to_dict() for doc in task_docs}
    missing_task_ids = set(task_ids) - set(
        task_id for task_id, task in tasks.items() if task is not None
    )

    if missing_task_ids:
        raise KeyError(f"Tasks not found: {', '.join(missing_task_ids)}")
    task_ids_to_ingest = [
        task_id
        for task_id, task in tasks.items()
        if task["status"] == "pending_ingestion"
        or (task["status"] == "ingested" and re_ingest is not None)
        or (task["status"] == "fully_processed" and re_ingest == "all")
    ]

    if not task_ids_to_ingest:
        return False

    for i in range(0, len(task_ids_to_ingest), bundle_size):
        task_bundle = task_ids_to_ingest[i : i + bundle_size]

        @firestore.transactional
        def ingest_batch_in_transaction(transaction):
            ingested_task_ids = [
                task_id
                for task_id in task_bundle  # pylint: disable=cell-var-from-loop
                if tasks[task_id]["status"] == "ingested"
            ]
            active_subtasks = []
            for i in range(0, len(ingested_task_ids), 30):
                task_batch = ingested_task_ids[i : i + 30]

                subtasks_query = (
                    client.collection(f"{project_name}_subtasks")
                    .where("task_id", "in", task_batch)
                    .where("is_active", "==", True)
                )
                batch_subtasks = list(subtasks_query.stream(transaction=transaction))
                active_subtasks.extend(batch_subtasks)

            subtasks = client.collection(f"{project_name}_subtasks")
            for subtask in active_subtasks:
                transaction.update(subtasks.document(subtask.id), {"is_active": False})

            for task_id in task_bundle:  # pylint: disable=cell-var-from-loop
                create_subtask_structure(
                    client=client,
                    transaction=transaction,
                    project_name=project_name,
                    task_id=task_id,
                    subtask_structure=subtask_structure,
                    priority=priority,
                )

            return True

        ingest_batch_in_transaction(client.transaction())

    return True


def ingest_batch(
    project_name: str,
    batch_id: str,
    subtask_structure: str,
    re_ingest: Literal["not_processed", "all"] | None = None,
    priority: int = 1,
) -> bool:
    """
    Ingest all tasks in a batch, changing their status from pending_ingestion to ingested.

    :param project_name: The name of the project.
    :param batch_id: The ID of the batch to ingest.
    :param subtask_structure: Name of subtask structure to create
    :param re_ingest: Controls re-ingestion behavior:
                      None - only ingest pending_ingestion tasks
                      "not_processed" - ingest pending_ingestion and ingested tasks
                      "all" - ingest all tasks including fully_processed ones
    :param priority: Priority for subtasks (default: 1)
    :return: True if any tasks were ingested, False otherwise.
    """
    client = get_firestore_client()
    tasks_query = client.collection(f"{project_name}_tasks").where("batch_id", "==", batch_id)

    if re_ingest is None:
        tasks_query = tasks_query.where("status", "==", "pending_ingestion")
    elif re_ingest == "not_processed":
        tasks_query = tasks_query.where("status", "in", ["pending_ingestion", "ingested"])

    tasks = list(tasks_query.stream())
    if not tasks:
        return False

    # Get task IDs and ingest in bundles
    task_ids = [task_doc.id for task_doc in tasks]
    return ingest_tasks(
        project_name=project_name,
        task_ids=task_ids,
        subtask_structure=subtask_structure,
        re_ingest=re_ingest,
        priority=priority,
    )
