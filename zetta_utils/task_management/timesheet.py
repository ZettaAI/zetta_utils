# pylint: disable=all
import time
from typing import cast

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from zetta_utils.log import get_logger
from zetta_utils.task_management.utils import generate_id_nonunique

from .db.models import SubtaskModel, TimesheetModel, UserModel
from .db.session import get_session_context
from .exceptions import UserValidationError
from .subtask import get_subtask
from .user import get_user

logger = get_logger(__name__)


@typechecked
def submit_timesheet(
    *,
    project_name: str,
    user_id: str,
    duration_seconds: float,
    subtask_id: str,
    db_session: Session | None = None,
) -> None:
    """Submit a timesheet entry for a user.

    :param project_name: The name of the project
    :param user_id: The ID of the user submitting the timesheet
    :param duration_seconds: The duration of work in seconds
    :param subtask_id: The ID of the subtask to submit timesheet for
    :param db_session: Database session to use (optional)
    :raises UserValidationError: If user validation fails or subtask ID doesn't match
    :raises ValueError: If entry data is invalid
    :raises RuntimeError: If the database transaction fails
    """
    with get_session_context(db_session) as session:
        if duration_seconds <= 0:
            raise ValueError("Duration must be positive")

        # Get user and verify they have an active subtask (preserving original error handling)
        try:
            user_data = get_user(project_name=project_name, user_id=user_id, db_session=session)
        except KeyError:
            raise UserValidationError(f"User {user_id} not found")

        if not user_data["active_subtask"]:
            raise UserValidationError("User does not have an active subtask")

        # Verify the provided subtask_id matches the user's active subtask
        if user_data["active_subtask"] != subtask_id:
            raise UserValidationError(
                f"Provided subtask_id {subtask_id} does not match user's active subtask {user_data['active_subtask']}"
            )

        # Get the subtask and verify user is assigned (preserving original error handling)
        subtask_data = get_subtask(
            project_name=project_name, subtask_id=subtask_id, db_session=session
        )
        if subtask_data["active_user_id"] != user_id:
            raise UserValidationError("Subtask not assigned to this user")

        # ATOMIC TIMESHEET SUBMISSION: Now that validation is done, perform atomic operations
        logger.info(
            f"Submitting timesheet for user {user_id}, subtask {subtask_id}, duration {duration_seconds}s"
        )

        try:
            entry_id = f"{user_id}_{subtask_id}"

            # RACE-CONDITION SAFE UPSERT:
            # Try to insert first. If it fails due to unique constraint, update existing.
            try:
                # Attempt to create new entry first
                new_timesheet = TimesheetModel(
                    project_name=project_name,
                    entry_id=entry_id,
                    subtask_id=subtask_id,
                    task_id=subtask_data["task_id"],
                    user=user_id,
                    seconds_spent=int(duration_seconds),
                )
                session.add(new_timesheet)
                session.flush()  # Force immediate constraint check
                logger.info(
                    f"Created new timesheet entry for {user_id} on {subtask_id}: {duration_seconds}s"
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

            # Lock and update subtask last_leased_ts atomically
            locked_subtask_query = (
                select(SubtaskModel)
                .where(SubtaskModel.project_name == project_name)
                .where(SubtaskModel.subtask_id == subtask_id)
                .with_for_update()
            )
            locked_subtask = session.execute(locked_subtask_query).scalar_one()
            locked_subtask.last_leased_ts = time.time()
            logger.info(f"Updated last_leased_ts for subtask {subtask_id}")

            session.commit()
            logger.info(
                f"Successfully submitted timesheet for user {user_id}, subtask {subtask_id}"
            )

        except Exception as e:
            session.rollback()
            logger.error(
                f"Failed to submit timesheet for user {user_id}, subtask {subtask_id}: {e}"
            )
            raise RuntimeError(f"Failed to submit timesheet: {e}")
