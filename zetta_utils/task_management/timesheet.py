# pylint: disable=all
import time
from typing import cast

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from zetta_utils.log import get_logger
from zetta_utils.task_management.utils import generate_id_nonunique

from .db.models import TaskModel, TimesheetModel, UserModel
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
        task_data = get_task(
            project_name=project_name, task_id=task_id, db_session=session
        )
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
                    job_id=task_data["job_id"],
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

            session.commit()
            logger.info(
                f"Successfully submitted timesheet for user {user_id}, task {task_id}"
            )

        except Exception as e:
            session.rollback()
            logger.error(
                f"Failed to submit timesheet for user {user_id}, task {task_id}: {e}"
            )
            raise RuntimeError(f"Failed to submit timesheet: {e}")
