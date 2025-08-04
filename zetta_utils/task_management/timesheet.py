# pylint: disable=all
import time
from datetime import datetime, timedelta
from typing import cast

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from zetta_utils.log import get_logger
from zetta_utils.task_management.utils import generate_id_nonunique

from .db.models import TaskModel, TimesheetModel, TimesheetSubmissionModel, UserModel
from .db.session import get_session_context
from .exceptions import UserValidationError
from .task import get_task
from .user import get_user

logger = get_logger(__name__)


@typechecked
def submit_timesheet(
    *,
    project_name: str,
    user_id: str,
    duration_seconds: float,
    task_id: str,
    db_session: Session | None = None,
) -> None:
    """Submit a timesheet entry for a user.

    :param project_name: The name of the project
    :param user_id: The ID of the user submitting the timesheet
    :param duration_seconds: The duration of work in seconds
    :param task_id: The ID of the task to submit timesheet for
    :param db_session: Database session to use (optional)
    :raises UserValidationError: If user validation fails or task ID doesn't match
    :raises ValueError: If entry data is invalid
    :raises RuntimeError: If the database transaction fails
    """
    with get_session_context(db_session) as session:
        if duration_seconds <= 0:
            raise ValueError("Duration must be positive")

        # Get user and verify they have an active task (preserving original error handling)
        try:
            user_data = get_user(project_name=project_name, user_id=user_id, db_session=session)
        except KeyError:
            raise UserValidationError(f"User {user_id} not found")

        if not user_data["active_task"]:
            raise UserValidationError("User does not have an active task")

        # Verify the provided task_id matches the user's active task
        if user_data["active_task"] != task_id:
            raise UserValidationError(
                f"Provided task_id {task_id} does not match user's active task {user_data['active_task']}"
            )

        # Get the task and verify user is assigned (preserving original error handling)
        task_data = get_task(project_name=project_name, task_id=task_id, db_session=session)
        if task_data["active_user_id"] != user_id:
            raise UserValidationError("Task not assigned to this user")

        # ATOMIC TIMESHEET SUBMISSION: Now that validation is done, perform atomic operations
        logger.info(
            f"Submitting timesheet for user {user_id}, task {task_id}, duration {duration_seconds}s"
        )

        try:
            entry_id = f"{user_id}_{task_id}"

            # RACE-CONDITION SAFE UPSERT:
            # Try to insert first. If it fails due to unique constraint, update existing.
            try:
                # Attempt to create new entry first
                new_timesheet = TimesheetModel(
                    project_name=project_name,
                    entry_id=entry_id,
                    task_id=task_id,
                    user=user_id,
                    seconds_spent=int(duration_seconds),
                )
                session.add(new_timesheet)
                session.flush()  # Force immediate constraint check
                logger.info(
                    f"Created new timesheet entry for {user_id} on {task_id}: {duration_seconds}s"
                )
            except IntegrityError:
                # Entry already exists, rollback to savepoint and update existing
                session.rollback()

                # Now lock and update the existing entry
                locked_timesheet_query = (
                    select(TimesheetModel)
                    .where(TimesheetModel.project_name == project_name)
                    .where(TimesheetModel.entry_id == entry_id)
                    .with_for_update()
                )
                existing_timesheet = session.execute(locked_timesheet_query).scalar_one()
                existing_timesheet.seconds_spent += int(duration_seconds)
                logger.info(
                    f"Updated existing timesheet entry {entry_id}: added {duration_seconds}s, total now {existing_timesheet.seconds_spent}s"
                )

            # Lock and update task last_leased_ts atomically
            locked_task_query = (
                select(TaskModel)
                .where(TaskModel.project_name == project_name)
                .where(TaskModel.task_id == task_id)
                .with_for_update()
            )
            locked_task = session.execute(locked_task_query).scalar_one()
            locked_task.last_leased_ts = time.time()
            logger.info(f"Updated last_leased_ts for task {task_id}")

            # Create individual submission record for audit trail
            submission = TimesheetSubmissionModel(
                project_name=project_name,
                user_id=user_id,
                task_id=task_id,
                seconds_spent=int(duration_seconds),
                submitted_at=datetime.utcnow(),
            )
            session.add(submission)
            logger.info(f"Created timesheet submission record for user {user_id}, task {task_id}")

            session.commit()
            logger.info(f"Successfully submitted timesheet for user {user_id}, task {task_id}")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to submit timesheet for user {user_id}, task {task_id}: {e}")
            raise RuntimeError(f"Failed to submit timesheet: {e}")


@typechecked
def get_timesheet_submissions(
    *,
    project_name: str,
    user_id: str | None = None,
    task_id: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    db_session: Session | None = None,
) -> list[dict]:
    """Get timesheet submissions based on filters.

    :param project_name: The name of the project
    :param user_id: Filter by user ID (optional)
    :param task_id: Filter by task ID (optional)
    :param start_date: Filter submissions after this date (optional)
    :param end_date: Filter submissions before this date (optional)
    :param db_session: Database session to use (optional)
    :return: List of timesheet submission dictionaries
    """
    with get_session_context(db_session) as session:
        query = select(TimesheetSubmissionModel).where(
            TimesheetSubmissionModel.project_name == project_name
        )

        if user_id is not None:
            query = query.where(TimesheetSubmissionModel.user_id == user_id)

        if task_id is not None:
            query = query.where(TimesheetSubmissionModel.task_id == task_id)

        if start_date is not None:
            query = query.where(TimesheetSubmissionModel.submitted_at >= start_date)

        if end_date is not None:
            query = query.where(TimesheetSubmissionModel.submitted_at <= end_date)

        query = query.order_by(TimesheetSubmissionModel.submitted_at.desc())

        results = session.execute(query).scalars().all()
        return [result.to_dict() for result in results]


@typechecked
def get_user_work_history(
    *,
    project_name: str,
    user_id: str,
    days: int = 7,
    db_session: Session | None = None,
) -> dict:
    """Get detailed work history for a user.

    :param project_name: The name of the project
    :param user_id: The ID of the user
    :param days: Number of days to look back (default 7)
    :param db_session: Database session to use (optional)
    :return: Dictionary with work history summary
    """
    with get_session_context(db_session) as session:
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Get all submissions in date range
        submissions = get_timesheet_submissions(
            project_name=project_name,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            db_session=session,
        )

        # Calculate summary stats
        total_seconds = sum(sub["seconds_spent"] for sub in submissions)
        total_hours = total_seconds / 3600
        submission_count = len(submissions)

        # Group by task
        task_summary = {}
        for sub in submissions:
            task_id = sub["task_id"]
            if task_id not in task_summary:
                task_summary[task_id] = {
                    "seconds_spent": 0,
                    "submission_count": 0,
                }
            task_summary[task_id]["seconds_spent"] += sub["seconds_spent"]
            task_summary[task_id]["submission_count"] += 1

        return {
            "user_id": user_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_hours": total_hours,
            "submission_count": submission_count,
            "task_summary": task_summary,
            "submissions": submissions,
        }
