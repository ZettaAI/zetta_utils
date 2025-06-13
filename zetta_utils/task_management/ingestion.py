from typing import Any, Literal

from sqlalchemy import select, update
from sqlalchemy.orm import Session
from tqdm import tqdm
from typeguard import typechecked

from .db.models import JobModel, TaskModel
from .db.session import get_session_context
from .task_structure import create_task_structure


@typechecked
def ingest_job(
    *,
    project_name: str,
    job_id: str,
    task_structure: str,
    task_structure_kwargs: dict[str, Any],
    re_ingest: Literal["not_processed", "all"] | None = None,
    priority: int = 1,
    db_session: Session | None = None,
) -> bool:
    """
    Ingest a job, changing its status from pending_ingestion to ingested.

    :param project_name: The name of the project.
    :param job_id: The ID of the job to ingest.
    :param task_structure: Name of task structure to create
    :param task_structure_kwargs: Keyword arguments for the task structure
    :param re_ingest: Controls re-ingestion behavior:
                      None - only ingest pending_ingestion jobs
                      "not_processed" - ingest pending_ingestion and ingested jobs
                      "all" - ingest all jobs including fully_processed ones
    :param priority: Priority for tasks (default: 1)
    :param db_session: Database session to use (optional)
    :return: True if the job was ingested, False otherwise.
    :raises ValueError: If the job is not in a valid status for ingestion.
    :raises KeyError: If the job does not exist.
    """
    return ingest_jobs(
        project_name=project_name,
        job_ids=[job_id],
        task_structure=task_structure,
        re_ingest=re_ingest,
        priority=priority,
        bundle_size=1,
        task_structure_kwargs=task_structure_kwargs,
        db_session=db_session,
    )


@typechecked
def ingest_jobs(
    *,
    project_name: str,
    job_ids: list[str],
    task_structure: str,
    task_structure_kwargs: dict[str, Any],
    re_ingest: Literal["not_processed", "all"] | None = None,
    priority: int = 1,
    bundle_size: int = 500,
    db_session: Session | None = None,
) -> bool:
    """
    Ingest multiple jobs in bundles.

    :param project_name: The name of the project.
    :param job_ids: List of job IDs to ingest.
    :param task_structure: Name of task structure to create
    :param task_structure_kwargs: Keyword arguments for the task structure
    :param re_ingest: Controls re-ingestion behavior:
                      None - only ingest pending_ingestion jobs
                      "not_processed" - ingest pending_ingestion and ingested jobs
                      "all" - ingest all jobs including fully_processed ones
    :param priority: Priority for tasks (default: 1)
    :param bundle_size: Maximum number of operations per batch (default 500)
    :param db_session: Database session to use (optional)
    :return: True if all jobs with appropriate status were ingested, False otherwise.
    :raises ValueError: If any job is not in a valid status for ingestion.
    :raises KeyError: If any job does not exist.
    """
    with get_session_context(db_session) as session:
        print(f"Ingesting {len(job_ids)} jobs")

        # Get all jobs
        jobs_query = (
            select(JobModel)
            .where(JobModel.project_name == project_name)
            .where(JobModel.job_id.in_(job_ids))
        )
        job_models = session.execute(jobs_query).scalars().all()
        jobs = {str(job.job_id): job.to_dict() for job in job_models}

        missing_job_ids = set(job_ids) - set(jobs.keys())
        if missing_job_ids:
            raise KeyError(f"Jobs not found: {', '.join(missing_job_ids)}")

        # Filter jobs based on re_ingest parameter
        job_ids_to_ingest = [
            job_id
            for job_id, job in jobs.items()
            if job["status"] == "pending_ingestion"
            or (job["status"] == "ingested" and re_ingest is not None)
            or (job["status"] == "fully_processed" and re_ingest == "all")
        ]

        if not job_ids_to_ingest:
            return False

        print(f"Ingesting {len(job_ids_to_ingest)} jobs")

        # Process in bundles
        for i in tqdm(list(range(0, len(job_ids_to_ingest), bundle_size))):
            job_bundle = job_ids_to_ingest[i : i + bundle_size]

            # For jobs that are already ingested, deactivate existing tasks
            ingested_job_ids = [
                job_id for job_id in job_bundle if jobs[job_id]["status"] == "ingested"
            ]

            if ingested_job_ids:
                # Deactivate existing active tasks for re-ingestion
                deactivate_query = (
                    update(TaskModel)
                    .where(TaskModel.project_name == project_name)
                    .where(TaskModel.job_id.in_(ingested_job_ids))
                    .where(TaskModel.is_active.is_(True))
                    .values(is_active=False)
                )
                session.execute(deactivate_query)

            # Create task structure for each job
            for job_id in job_bundle:
                try:
                    create_task_structure(
                        project_name=project_name,
                        job_id=str(job_id),
                        task_structure=task_structure,
                        priority=priority,
                        task_structure_kwargs=task_structure_kwargs,
                        db_session=session,
                    )
                except RuntimeError as e:
                    # Extract the original error message from the nested RuntimeError
                    # Note: create_task_structure already handles rollback internally
                    original_error = str(e).replace("Failed to create task structure: ", "")
                    raise RuntimeError(f"Failed to ingest job bundle: {original_error}") from e

            try:
                session.commit()
            except Exception as exc:  # pragma: no cover
                session.rollback()
                raise RuntimeError(f"Failed to ingest job bundle: {exc}") from exc

        return True


def ingest_batch(
    *,
    project_name: str,
    batch_id: str,
    task_structure: str,
    task_structure_kwargs: dict[str, Any],
    re_ingest: Literal["not_processed", "all"] | None = None,
    priority: int = 1,
    bundle_size: int = 100,
    db_session: Session | None = None,
) -> bool:
    """
    Ingest all jobs in a batch, changing their status from pending_ingestion to ingested.

    :param project_name: The name of the project.
    :param batch_id: The ID of the batch to ingest.
    :param task_structure: Name of task structure to create
    :param task_structure_kwargs: Keyword arguments for the task structure
    :param re_ingest: Controls re-ingestion behavior:
                      None - only ingest pending_ingestion jobs
                      "not_processed" - ingest pending_ingestion and ingested jobs
                      "all" - ingest all jobs including fully_processed ones
    :param priority: Priority for tasks (default: 1)
    :param bundle_size: Maximum number of operations per batch
    :param db_session: Database session to use (optional)
    :return: True if any jobs were ingested, False otherwise.
    """
    with get_session_context(db_session) as session:
        # Build query based on re_ingest parameter
        jobs_query = (
            select(JobModel)
            .where(JobModel.project_name == project_name)
            .where(JobModel.batch_id == batch_id)
        )

        if re_ingest is None:
            jobs_query = jobs_query.where(JobModel.status == "pending_ingestion")
        elif re_ingest == "not_processed":
            jobs_query = jobs_query.where(JobModel.status.in_(["pending_ingestion", "ingested"]))
        # For "all", no additional filter needed

        job_models = session.execute(jobs_query).scalars().all()
        if not job_models:
            return False

        # Get job IDs and ingest in bundles
        job_ids = [str(job.job_id) for job in job_models]
        print(f"Ingesting {len(job_ids)} jobs")
        return ingest_jobs(
            project_name=project_name,
            job_ids=job_ids,
            task_structure=task_structure,
            task_structure_kwargs=task_structure_kwargs,
            re_ingest=re_ingest,
            priority=priority,
            bundle_size=bundle_size,
            db_session=session,
        )
