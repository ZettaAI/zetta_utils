from typing import Any, Literal

from sqlalchemy import select, update
from sqlalchemy.orm import Session
from tqdm import tqdm
from typeguard import typechecked

from .db.models import SubtaskModel, TaskModel
from .subtask_structure import create_subtask_structure


@typechecked
def ingest_task(
    db_session: Session,
    project_name: str,
    task_id: str,
    subtask_structure: str,
    subtask_structure_kwargs: dict[str, Any],
    re_ingest: Literal["not_processed", "all"] | None = None,
    priority: int = 1,
) -> bool:
    """
    Ingest a task, changing its status from pending_ingestion to ingested.

    :param db_session: Database session to use
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
        db_session=db_session,
        project_name=project_name,
        task_ids=[task_id],
        subtask_structure=subtask_structure,
        re_ingest=re_ingest,
        priority=priority,
        bundle_size=1,
        subtask_structure_kwargs=subtask_structure_kwargs,
    )


@typechecked
def ingest_tasks(
    db_session: Session,
    project_name: str,
    task_ids: list[str],
    subtask_structure: str,
    subtask_structure_kwargs: dict[str, Any],
    re_ingest: Literal["not_processed", "all"] | None = None,
    priority: int = 1,
    bundle_size: int = 500,
) -> bool:
    """
    Ingest multiple tasks in bundles.

    :param db_session: Database session to use
    :param project_name: The name of the project.
    :param task_ids: List of task IDs to ingest.
    :param subtask_structure: Name of subtask structure to create
    :param re_ingest: Controls re-ingestion behavior:
                      None - only ingest pending_ingestion tasks
                      "not_processed" - ingest pending_ingestion and ingested tasks
                      "all" - ingest all tasks including fully_processed ones
    :param priority: Priority for subtasks (default: 1)
    :param bundle_size: Maximum number of operations per batch (default 500)
    :return: True if all tasks with appropriate status were ingested, False otherwise.
    :raises ValueError: If any task is not in a valid status for ingestion.
    :raises KeyError: If any task does not exist.
    """
    print(f"Ingesting {len(task_ids)} tasks")

    # Get all tasks
    tasks_query = (
        select(TaskModel)
        .where(TaskModel.project_name == project_name)
        .where(TaskModel.task_id.in_(task_ids))
    )
    task_models = db_session.execute(tasks_query).scalars().all()
    tasks = {str(task.task_id): task.to_dict() for task in task_models}

    missing_task_ids = set(task_ids) - set(tasks.keys())
    if missing_task_ids:
        raise KeyError(f"Tasks not found: {', '.join(missing_task_ids)}")

    # Filter tasks based on re_ingest parameter
    task_ids_to_ingest = [
        task_id
        for task_id, task in tasks.items()
        if task["status"] == "pending_ingestion"
        or (task["status"] == "ingested" and re_ingest is not None)
        or (task["status"] == "fully_processed" and re_ingest == "all")
    ]

    if not task_ids_to_ingest:
        return False

    print(f"Ingesting {len(task_ids_to_ingest)} tasks")

    # Process in bundles
    for i in tqdm(list(range(0, len(task_ids_to_ingest), bundle_size))):
        task_bundle = task_ids_to_ingest[i : i + bundle_size]

        # For tasks that are already ingested, deactivate existing subtasks
        ingested_task_ids = [
            task_id for task_id in task_bundle if tasks[task_id]["status"] == "ingested"
        ]

        if ingested_task_ids:
            # Deactivate existing active subtasks for re-ingestion
            deactivate_query = (
                update(SubtaskModel)
                .where(SubtaskModel.project_name == project_name)
                .where(SubtaskModel.task_id.in_(ingested_task_ids))
                .where(SubtaskModel.is_active.is_(True))
                .values(is_active=False)
            )
            db_session.execute(deactivate_query)

        # Create subtask structure for each task
        for task_id in task_bundle:
            try:
                create_subtask_structure(
                    db_session=db_session,
                    project_name=project_name,
                    task_id=str(task_id),
                    subtask_structure=subtask_structure,
                    priority=priority,
                    subtask_structure_kwargs=subtask_structure_kwargs,
                )
            except RuntimeError as e:
                # Extract the original error message from the nested RuntimeError
                # Note: create_subtask_structure already handles rollback internally
                original_error = str(e).replace("Failed to create subtask structure: ", "")
                raise RuntimeError(f"Failed to ingest task bundle: {original_error}") from e

        try:
            db_session.commit()
        except Exception as exc:
            db_session.rollback()
            raise RuntimeError(f"Failed to ingest task bundle: {exc}") from exc

    return True


def ingest_batch(
    db_session: Session,
    project_name: str,
    batch_id: str,
    subtask_structure: str,
    subtask_structure_kwargs: dict[str, Any],
    re_ingest: Literal["not_processed", "all"] | None = None,
    priority: int = 1,
    bundle_size: int = 100,
) -> bool:
    """
    Ingest all tasks in a batch, changing their status from pending_ingestion to ingested.

    :param db_session: Database session to use
    :param project_name: The name of the project.
    :param batch_id: The ID of the batch to ingest.
    :param subtask_structure: Name of subtask structure to create
    :param re_ingest: Controls re-ingestion behavior:
                      None - only ingest pending_ingestion tasks
                      "not_processed" - ingest pending_ingestion and ingested tasks
                      "all" - ingest all tasks including fully_processed ones
    :param priority: Priority for subtasks (default: 1)
    :param bundle_size: Maximum number of operations per batch
    :return: True if any tasks were ingested, False otherwise.
    """
    # Build query based on re_ingest parameter
    tasks_query = (
        select(TaskModel)
        .where(TaskModel.project_name == project_name)
        .where(TaskModel.batch_id == batch_id)
    )

    if re_ingest is None:
        tasks_query = tasks_query.where(TaskModel.status == "pending_ingestion")
    elif re_ingest == "not_processed":
        tasks_query = tasks_query.where(TaskModel.status.in_(["pending_ingestion", "ingested"]))
    # For "all", no additional filter needed

    task_models = db_session.execute(tasks_query).scalars().all()
    if not task_models:
        return False

    # Get task IDs and ingest in bundles
    task_ids = [str(task.task_id) for task in task_models]
    print(f"Ingesting {len(task_ids)} tasks")
    return ingest_tasks(
        db_session=db_session,
        project_name=project_name,
        task_ids=task_ids,
        subtask_structure=subtask_structure,
        subtask_structure_kwargs=subtask_structure_kwargs,
        re_ingest=re_ingest,
        priority=priority,
        bundle_size=bundle_size,
    )
