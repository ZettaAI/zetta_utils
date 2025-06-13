from typing import Literal, cast

from sqlalchemy import func, select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from zetta_utils.task_management.utils import generate_id_nonunique

from .db.models import JobModel
from .db.session import get_session_context
from .types import Job, JobUpdate

JobStatus = Literal["pending_ingestion", "ingested", "fully_processed"]


def create_jobs_batch(
    *,
    project_name: str,
    jobs: list[Job],
    batch_size: int = 500,
    db_session: Session | None = None,
) -> list[str]:
    """
    Create multiple jobs in a single batch operation.

    :param project_name: The name of the project.
    :param jobs: List of jobs to create.
    :param batch_size: Maximum number of operations per batch (default 500).
    :param db_session: Database session to use (optional).
    :return: List of created job IDs.
    :raises ValueError: If a job already exists with different content.
    """
    with get_session_context(db_session) as session:
        job_ids = []
        all_job_ids = [job["job_id"] for job in jobs]

        # Check for existing jobs
        existing_jobs = {}
        if all_job_ids:
            query = (
                select(JobModel)
                .where(JobModel.job_id.in_(all_job_ids))
                .where(JobModel.project_name == project_name)
            )
            for existing_job in session.execute(query).scalars():
                existing_jobs[str(existing_job.job_id)] = existing_job.to_dict()

        # Create a lookup dict for incoming jobs
        incoming_jobs_dict = {job["job_id"]: job for job in jobs}

        # Check for conflicts (existing jobs with different content)
        for job_id, existing_data in existing_jobs.items():
            incoming_data = incoming_jobs_dict[job_id]

            # Check if jobs are identical
            if existing_data != incoming_data:
                raise ValueError(
                    f"Job {job_id} already exists with different content. "
                    f"Existing: {existing_data}, Incoming: {incoming_data}"
                )

        # Filter out jobs that already exist
        filtered_jobs = [job for job in jobs if job["job_id"] not in existing_jobs]

        # Create new jobs in batches
        for i in range(0, len(filtered_jobs), batch_size):
            chunk = filtered_jobs[i : i + batch_size]

            for job in chunk:
                job_data = {**job, "id_nonunique": generate_id_nonunique()}
                model = JobModel.from_dict(project_name, job_data)
                session.add(model)
                job_ids.append(job["job_id"])

        session.commit()

        # Return all job IDs (including existing ones that were identical)
        return [job["job_id"] for job in jobs]


@typechecked
def create_job(*, project_name: str, data: Job, db_session: Session | None = None) -> str:
    """
    Create a new job record in the database.

    :param project_name: The name of the project.
    :param data: The job data to create.
    :param db_session: Database session to use (optional).
    :return: The job_id of the created job.
    :raises ValueError: If the job data is invalid or job already exists.
    :raises RuntimeError: If the database operation fails.
    """
    # Use create_jobs_batch for maximum code reuse
    result = create_jobs_batch(project_name=project_name, jobs=[data], db_session=db_session)
    return result[0]


def get_job(*, project_name: str, job_id: str, db_session: Session | None = None) -> Job:
    """
    Get a job by ID.

    :param project_name: The name of the project.
    :param job_id: The ID of the job.
    :param db_session: Database session to use (optional).
    :return: The job data.
    :raises KeyError: If the job does not exist.
    """
    with get_session_context(db_session) as session:
        query = (
            select(JobModel)
            .where(JobModel.job_id == job_id)
            .where(JobModel.project_name == project_name)
        )
        try:
            job = session.execute(query).scalar_one()
            return cast(Job, job.to_dict())
        except NoResultFound as exc:
            raise KeyError(f"Job {job_id} not found in project {project_name}") from exc


def update_job(
    *, project_name: str, job_id: str, data: JobUpdate, db_session: Session | None = None
) -> bool:
    """
    Update a job record in the database.

    :param project_name: The name of the project.
    :param job_id: The unique identifier of the job.
    :param data: The job data to update.
    :param db_session: Database session to use (optional).
    :return: True on success.
    :raises ValueError: If the update data is invalid.
    :raises KeyError: If the job does not exist.
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
            select(JobModel)
            .where(JobModel.job_id == job_id)
            .where(JobModel.project_name == project_name)
        )

        try:
            job = session.execute(query).scalar_one()
        except NoResultFound as exc:
            raise KeyError(f"Job {job_id} not found in project {project_name}") from exc

        for field, value in data.items():
            if hasattr(job, field):
                setattr(job, field, value)

        session.commit()
        return True


def list_jobs_summary(
    *, project_name: str, db_session: Session | None = None
) -> dict:
    """
    Get a summary of jobs in a project with counts and sample job IDs.
    
    :param project_name: The name of the project
    :param db_session: Database session to use (optional)
    :return: Dictionary with counts and job ID lists
    """
    with get_session_context(db_session) as session:
        # Count pending ingestion jobs
        pending_count_query = (
            select(func.count(JobModel.job_id))
            .where(JobModel.project_name == project_name)
            .where(JobModel.status == "pending_ingestion")
        )
        pending_count = session.execute(pending_count_query).scalar() or 0
        
        # Count ingested jobs  
        ingested_count_query = (
            select(func.count(JobModel.job_id))
            .where(JobModel.project_name == project_name)
            .where(JobModel.status == "ingested")
        )
        ingested_count = session.execute(ingested_count_query).scalar() or 0
        
        # Count fully processed jobs
        completed_count_query = (
            select(func.count(JobModel.job_id))
            .where(JobModel.project_name == project_name)
            .where(JobModel.status == "fully_processed")
        )
        completed_count = session.execute(completed_count_query).scalar() or 0
        
        # Get first 5 pending ingestion job IDs
        pending_ids_query = (
            select(JobModel.job_id)
            .where(JobModel.project_name == project_name)
            .where(JobModel.status == "pending_ingestion")
            .order_by(JobModel.job_id)
            .limit(5)
        )
        pending_ids = list(session.execute(pending_ids_query).scalars().all())
        
        # Get first 5 ingested job IDs
        ingested_ids_query = (
            select(JobModel.job_id)
            .where(JobModel.project_name == project_name)
            .where(JobModel.status == "ingested")
            .order_by(JobModel.job_id)
            .limit(5)
        )
        ingested_ids = list(session.execute(ingested_ids_query).scalars().all())
        
        return {
            "pending_ingestion_count": pending_count,
            "ingested_count": ingested_count,
            "completed_count": completed_count,
            "pending_ingestion_ids": pending_ids,
            "ingested_ids": ingested_ids,
        }
