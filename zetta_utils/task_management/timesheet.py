# pylint: disable=all
import time

from google.cloud import firestore
from typeguard import typechecked

from .exceptions import UserValidationError
from .helpers import get_transaction, retry_transient_errors
from .project import get_collection, get_firestore_client


@retry_transient_errors
@typechecked
def submit_timesheet(
    project_name: str, user_id: str, duration_seconds: float, subtask_id: str
) -> None:
    """Submit a timesheet entry for a user.

    :param project_name: The name of the project
    :param user_id: The ID of the user submitting the timesheet
    :param duration_seconds: The duration of work in seconds
    :param subtask_id: The ID of the subtask to submit timesheet for
    :raises UserValidationError: If user validation fails or subtask ID doesn't match
    :raises ValueError: If entry data is invalid
    :raises RuntimeError: If the Firestore transaction fails
    """
    if duration_seconds <= 0:
        raise ValueError("Duration must be positive")

    client = get_firestore_client()
    user_ref = get_collection(project_name, "users").document(user_id)

    @firestore.transactional
    def submit_in_transaction(transaction):
        # Get user and verify they have an active subtask
        user_doc = user_ref.get(transaction=transaction)
        if not user_doc.exists:
            raise UserValidationError(f"User {user_id} not found")

        user_data = user_doc.to_dict()
        if not user_data["active_subtask"]:
            raise UserValidationError("User does not have an active subtask")

        # Verify the provided subtask_id matches the user's active subtask
        if user_data["active_subtask"] != subtask_id:
            raise UserValidationError(
                f"Provided subtask_id {subtask_id} does not match user's active subtask {user_data['active_subtask']}"
            )

        # Get the subtask and verify user is assigned
        subtask_ref = get_collection(project_name, "subtasks").document(subtask_id)
        subtask_doc = subtask_ref.get(transaction=transaction)
        if not subtask_doc.exists:
            raise UserValidationError(f"Subtask {subtask_id} not found")

        subtask_data = subtask_doc.to_dict()
        if subtask_data["active_user_id"] != user_id:
            raise UserValidationError("Subtask not assigned to this user")

        timesheet_ref = get_collection(project_name, "timesheets").document(
            f"{user_id}_{subtask_id}"
        )
        timesheet_doc = timesheet_ref.get(transaction=transaction)

        if timesheet_doc.exists:
            existing_data = timesheet_doc.to_dict()
            timesheet_data = {
                "duration_seconds": existing_data["duration_seconds"] + duration_seconds,
                "user_id": user_id,
                "subtask_id": subtask_id,
                "last_updated_ts": time.time(),
            }
            transaction.update(timesheet_ref, timesheet_data)
        else:
            timesheet_data = {
                "duration_seconds": duration_seconds,
                "user_id": user_id,
                "subtask_id": subtask_id,
                "created_ts": time.time(),
                "last_updated_ts": time.time(),
            }
            transaction.set(timesheet_ref, timesheet_data)

        transaction.update(subtask_ref, {"last_leased_ts": time.time()})

    submit_in_transaction(get_transaction())
