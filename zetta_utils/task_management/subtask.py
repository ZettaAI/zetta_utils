# pylint: disable=singleton-comparison
import time
from typing import Any, cast

from sqlalchemy import func, select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from zetta_utils import log
from zetta_utils.task_management.utils import generate_id_nonunique

from .db.models import (
    DependencyModel,
    JobModel,
    SubtaskModel,
    SubtaskTypeModel,
    UserModel,
)
from .db.session import get_session_context
from .exceptions import SubtaskValidationError, UserValidationError
from .types import Subtask, SubtaskUpdate

logger = log.get_logger("zetta_utils.task_management.subtask")

_MAX_IDLE_SECONDS = 90


def get_max_idle_seconds() -> float:
    return _MAX_IDLE_SECONDS


def _validate_subtask(db_session: Session, project_name: str, subtask: dict) -> Subtask:
    """
    Validate that a subtask's data is consistent and valid.

    :param db_session: Database session to use
    :param project_name: The name of the project.
    :param subtask: The subtask data to validate
    :raises SubtaskValidationError: If the subtask data is invalid
    """
    try:
        # Get subtask type from database
        query = (
            select(SubtaskTypeModel)
            .where(SubtaskTypeModel.project_name == project_name)
            .where(SubtaskTypeModel.subtask_type == subtask["subtask_type"])
        )
        subtask_type = db_session.execute(query).scalar_one()
        subtask_type_dict = subtask_type.to_dict()
    except NoResultFound as e:
        raise SubtaskValidationError(f"Subtask type not found: {subtask['subtask_type']}") from e

    # If subtask is not active, it cannot have completion status or completed user
    if not subtask.get("is_active"):
        if subtask.get("completion_status") != "":
            raise SubtaskValidationError("Inactive subtask cannot have completion status")
        if subtask.get("completed_user_id") != "":
            raise SubtaskValidationError("Inactive subtask cannot have completed user")
        return Subtask(**subtask)  # type: ignore # typeguard will check for us

    # Validate completion status if present
    completion_status = subtask.get("completion_status", "")
    if completion_status != "":
        if "completion_statuses" not in subtask_type_dict:
            raise SubtaskValidationError(
                f"Subtask type {subtask['subtask_type']} has no valid completion statuses"
            )
        if completion_status not in subtask_type_dict["completion_statuses"]:
            raise SubtaskValidationError(
                f"Completion status '{completion_status}' not allowed for this subtask type"
            )

        # If there's a completion status, must have completed user
        if not subtask.get("completed_user_id"):
            raise SubtaskValidationError("Completed subtask must have completed_user_id")

    return Subtask(**subtask)  # type: ignore # typeguard will check for us


@typechecked
def create_subtask(*, project_name: str, data: Subtask, db_session: Session | None = None) -> str:
    """Create a new subtask record"""
    with get_session_context(db_session) as session:
        # Validate subtask type exists
        query = (
            select(SubtaskTypeModel)
            .where(SubtaskTypeModel.project_name == project_name)
            .where(SubtaskTypeModel.subtask_type == data["subtask_type"])
        )
        try:
            session.execute(query).scalar_one()
        except NoResultFound as exc:
            raise SubtaskValidationError(f"Subtask type {data['subtask_type']} not found") from exc

        # Check if subtask already exists
        existing_query = (
            select(SubtaskModel)
            .where(SubtaskModel.subtask_id == data["subtask_id"])
            .where(SubtaskModel.project_name == project_name)
        )
        existing = session.execute(existing_query).scalar_one_or_none()
        if existing:
            raise SubtaskValidationError(f"Subtask {data['subtask_id']} already exists")

        # Create new subtask
        subtask_data = {**data, "id_nonunique": generate_id_nonunique()}
        model = SubtaskModel.from_dict(project_name, subtask_data)
        session.add(model)
        session.commit()

        return data["subtask_id"]


@typechecked
def update_subtask(
    *, project_name: str, subtask_id: str, data: SubtaskUpdate, db_session: Session | None = None
) -> bool:
    """Update a subtask record"""
    with get_session_context(db_session) as session:
        # Get current subtask
        query = (
            select(SubtaskModel)
            .where(SubtaskModel.subtask_id == subtask_id)
            .where(SubtaskModel.project_name == project_name)
        )
        try:
            subtask = session.execute(query).scalar_one()
        except NoResultFound as exc:
            raise KeyError(f"Subtask {subtask_id} not found") from exc

        current_data = subtask.to_dict()
        merged_data = {**current_data, **data}
        _validate_subtask(session, project_name, merged_data)

        # If completion status is changing, handle side effects
        if "completion_status" in data and "completed_user_id" in data:
            _handle_subtask_completion(
                session,
                project_name,
                subtask_id,
                data["completion_status"],
            )

        # Apply updates generically
        for field, value in data.items():
            if hasattr(subtask, field):
                setattr(subtask, field, value)

        session.commit()
        return True


@typechecked
def start_subtask(  # pylint: disable=too-many-branches
    *,
    project_name: str,
    user_id: str,
    subtask_id: str | None = None,
    db_session: Session | None = None,
) -> str | None:
    with get_session_context(db_session) as session:
        user_query = (
            select(UserModel)
            .where(UserModel.user_id == user_id)
            .where(UserModel.project_name == project_name)
        )
        try:
            user = session.execute(user_query).scalar_one()
        except NoResultFound as exc:
            raise UserValidationError(f"User {user_id} not found") from exc

        current_active_subtask_id = user.active_subtask
        if (
            subtask_id is not None
            and current_active_subtask_id != ""
            and current_active_subtask_id != subtask_id
        ):
            raise UserValidationError(
                f"User already has an active subtask {current_active_subtask_id} "
                f"which is different from requested subtask {subtask_id}"
            )

        if subtask_id is None and current_active_subtask_id == "":
            selected_subtask = _auto_select_subtask(session, project_name, user_id)
        elif subtask_id is not None:
            subtask_query = (
                select(SubtaskModel)
                .where(SubtaskModel.subtask_id == subtask_id)
                .where(SubtaskModel.project_name == project_name)
            )
            selected_subtask = session.execute(subtask_query).scalar_one_or_none()
            if not selected_subtask:
                raise SubtaskValidationError(f"Subtask {subtask_id} not found")
        else:
            subtask_query = (
                select(SubtaskModel)
                .where(SubtaskModel.subtask_id == current_active_subtask_id)
                .where(SubtaskModel.project_name == project_name)
            )
            selected_subtask = session.execute(subtask_query).scalar_one()

        if selected_subtask is not None:
            subtask_data = selected_subtask.to_dict()
            _validate_subtask(session, project_name, subtask_data)

            # Check if user is qualified for this subtask type
            if user.qualified_subtask_types is not None and len(user.qualified_subtask_types) == 0:
                raise UserValidationError("User not qualified for this subtask type")
            if (
                user.qualified_subtask_types
                and subtask_data["subtask_type"] not in user.qualified_subtask_types
            ):
                raise UserValidationError("User not qualified for this subtask type")

            current_time = time.time()
            # Check if task is idle and can be taken over
            if subtask_data["active_user_id"] != "":
                _atomic_subtask_takeover(
                    session,
                    project_name,
                    selected_subtask,
                    user_id,
                    subtask_data,
                )
            else:
                # Simple assignment case - still need some locking to prevent races
                locked_user_query = (
                    select(UserModel)
                    .where(UserModel.user_id == user_id)
                    .where(UserModel.project_name == project_name)
                    .with_for_update()
                )
                locked_user = session.execute(locked_user_query).scalar_one()

                locked_subtask_query = (
                    select(SubtaskModel)
                    .where(SubtaskModel.subtask_id == selected_subtask.subtask_id)
                    .where(SubtaskModel.project_name == project_name)
                    .with_for_update()
                )
                locked_subtask = session.execute(locked_subtask_query).scalar_one()

                # Assign subtask to user
                locked_user.active_subtask = locked_subtask.subtask_id
                locked_subtask.active_user_id = user_id
                locked_subtask.last_leased_ts = current_time

            session.flush()  # Ensure changes are written to DB before commit
            session.commit()
            return str(selected_subtask.subtask_id)
        return None


def release_subtask(
    *,
    project_name: str,
    user_id: str,
    subtask_id: str,
    completion_status: str = "",
    db_session: Session | None = None,
) -> bool:
    """
    Releases the active subtask for a user within the project.

    :param project_name: The name of the project.
    :param user_id: The unique identifier of the user.
    :param subtask_id: The ID of the subtask to release.
    :param completion_status: The completion status to set for the subtask upon release.
        Empty string means not completed.
    :param db_session: Database session to use (optional).
    :return: True if the operation completes successfully
    :raises SubtaskValidationError: If the subtask validation fails
    :raises UserValidationError: If the user validation fails
    :raises ValueError: If the completion status is invalid
    :raises RuntimeError: If the database operation fails.
    """
    with get_session_context(db_session) as session:
        # Get user
        user_query = (
            select(UserModel)
            .where(UserModel.user_id == user_id)
            .where(UserModel.project_name == project_name)
        )
        user = session.execute(user_query).scalar_one()

        if user.active_subtask == "":
            raise UserValidationError("User does not have an active subtask")

        if user.active_subtask != subtask_id:
            raise UserValidationError("Subtask ID does not match user's active subtask")

        # Get subtask
        subtask_query = (
            select(SubtaskModel)
            .where(SubtaskModel.subtask_id == subtask_id)
            .where(SubtaskModel.project_name == project_name)
        )
        try:
            subtask = session.execute(subtask_query).scalar_one()
        except NoResultFound as exc:
            raise SubtaskValidationError(f"Subtask {subtask_id} not found") from exc

        # Handle completion side effects if completing
        if completion_status:
            _handle_subtask_completion(
                session,
                project_name,
                subtask_id,
                completion_status,
            )

        # Release the subtask
        subtask.active_user_id = ""
        subtask.completion_status = completion_status
        subtask.completed_user_id = user_id if completion_status else ""
        user.active_subtask = ""

        session.commit()
        return True


def _satisfy_dependency_and_activate_if_ready(
    dep: DependencyModel,
    dep_data: dict,
    locked_dependencies: dict,
    locked_subtasks: dict,
) -> None:
    """
    Mark a dependency as satisfied and activate the dependent subtask if all dependencies are met.

    :param dep: The dependency model to satisfy
    :param dep_data: Dictionary representation of the dependency data
    :param locked_dependencies: Dictionary of pre-locked dependencies by subtask_id
    :param locked_subtasks: Dictionary of pre-locked subtasks by subtask_id
    """
    logger.info(f"Dependency {dep.dependency_id} satisfied")

    # Mark dependency satisfied
    dep.is_satisfied = True

    # Check if dependent subtask can be activated
    dependent_subtask_id = dep_data["subtask_id"]

    # Use pre-locked dependencies to count remaining unsatisfied
    remaining_unsatisfied = sum(
        1
        for locked_dep in locked_dependencies[dependent_subtask_id]
        if not locked_dep.is_satisfied
    )

    logger.info(f"Remaining dependencies: {remaining_unsatisfied}")

    if remaining_unsatisfied == 0:  # Current dep will be satisfied
        logger.info(f"Activating dependent subtask {dependent_subtask_id}")
        print(f"Activating dependent subtask {dependent_subtask_id}")

        # Use pre-locked subtask
        dependent_subtask = locked_subtasks[dependent_subtask_id]
        dependent_subtask.is_active = True


def _handle_subtask_completion(  # pylint: disable=too-many-locals
    db_session: Session,
    project_name: str,
    subtask_id: str,
    completion_status: str,
) -> None:
    """
    Handle all side effects of completing a subtask.

    :param db_session: The current database session
    :param project_name: The name of the project
    :param subtask_id: The ID of completed subtask
    :param completion_status: The new completion status
    :raises RuntimeError: If the database operation fails
    """
    logger.info(f"Handling subtask completion side effects for `{subtask_id}`")
    print (f"Handling subtask completion side effects for `{subtask_id}`")
    updates: list[tuple[DocumentSnapshot, dict]] = []

    # Get the subtask and its dependencies
    subtask_ref = get_collection(project_name, "subtasks").document(subtask_id)
    subtask_doc = subtask_ref.get(transaction=transaction)
    subtask_data = subtask_doc.to_dict()

    # Get dependencies that depend on this subtask
    deps = (
        get_collection(project_name, "dependencies")
        .where("dependent_on_subtask_id", "==", subtask_id)
        .where("is_satisfied", "==", False)
        .get(transaction=transaction)
    )
    deps = db_session.execute(deps_query).scalars().all()

    # Collect all subtask IDs that need locking for proper ordering
    affected_subtask_ids = set()
    for dep in deps:
        affected_subtask_ids.add(dep.subtask_id)

    # Always lock in alphabetical order to prevent deadlocks
    sorted_subtask_ids = sorted(affected_subtask_ids)

    # Pre-lock all affected subtasks in order
    locked_subtasks = {}
    for affected_subtask_id in sorted_subtask_ids:
        locked_subtask_query = (
            select(SubtaskModel)
            .where(SubtaskModel.subtask_id == affected_subtask_id)
            .where(SubtaskModel.project_name == project_name)
            .with_for_update()
        )
        locked_subtask = db_session.execute(locked_subtask_query).scalar_one()
        locked_subtasks[affected_subtask_id] = locked_subtask

    # Pre-lock all dependencies in alphabetical order by dependency_id to avoid deadlocks
    locked_dependencies: dict[str, Any] = {}
    sorted_dependency_ids = sorted([dep.dependency_id for dep in deps])
    for dependency_id in sorted_dependency_ids:
        locked_dep_query = (
            select(DependencyModel)
            .where(DependencyModel.dependency_id == dependency_id)
            .where(DependencyModel.project_name == project_name)
            .with_for_update()
        )
        locked_dep = db_session.execute(locked_dep_query).scalar_one()

    logger.info(f"Got the following dependencies: {deps}")
    print (f"Got the following dependencies: {deps}")
    # Update dependencies and dependent subtasks
    for dep in deps:
        dep_data = dep.to_dict()
        logger.info(f"Dep data: {dep_data}")
        print (f"Dep data: {dep_data}")
        if dep_data["required_completion_status"] == completion_status:
            _satisfy_dependency_and_activate_if_ready(
                dep, dep_data, locked_dependencies, locked_subtasks
            )

    # Check if job is complete
    job_id = subtask_data["job_id"]

    # Lock the job to prevent race conditions
    job_lock_query = (
        select(JobModel)
        .where(JobModel.job_id == job_id)
        .where(JobModel.project_name == project_name)
        .with_for_update()
    )
    locked_job = db_session.execute(job_lock_query).scalar_one()

    # Count incomplete subtasks for this job (excluding the current one being completed)
    incomplete_subtasks_query = (
        select(func.count(SubtaskModel.subtask_id))
        .where(SubtaskModel.job_id == job_id)
        .where(SubtaskModel.project_name == project_name)
        .where(SubtaskModel.is_active == True)
        .where(SubtaskModel.completion_status == "")
        .where(SubtaskModel.subtask_id != subtask_data["subtask_id"])  # Exclude current subtask
    )
    incomplete_count = db_session.execute(incomplete_subtasks_query).scalar()

    logger.info(f"Job {job_id} has {incomplete_count} remaining incomplete subtasks")

    # If this was the last incomplete subtask, mark job as fully_processed
    if incomplete_count == 0:
        logger.info(f"Marking job {job_id} as fully_processed")
        locked_job.status = "fully_processed"


def _auto_select_subtask(
    db_session: Session, project_name: str, user_id: str
) -> SubtaskModel | None:
    """
    Auto-select a subtask for a user atomically.

    This function performs atomic selection of an appropriate subtask for a user,
    taking into account their qualifications and existing assignments.

    :param db_session: Database session to use
    :param project_name: The name of the project
    :param user_id: The user requesting a subtask
    :return: The selected subtask model or None if no suitable subtask available
    :raises UserValidationError: If the user is not qualified for any subtasks
    """
    logger.info(f"Auto-selecting subtask for user {user_id}")

    # Get user qualifications
    user_query = (
        select(UserModel)
        .where(UserModel.user_id == user_id)
        .where(UserModel.project_name == project_name)
    )
    user = db_session.execute(user_query).scalar_one()

    qualified_types = user.qualified_subtask_types or []
    if not qualified_types:
        logger.info(f"User {user_id} has no qualified subtask types")
        return None

    logger.info(f"User {user_id} is qualified for: {qualified_types}")

    # Strategy 1: Look for subtasks explicitly assigned to this user
    for subtask_type in qualified_types:
        logger.info(f"Looking for assigned subtasks of type {subtask_type} for user {user_id}")
        query = (
            select(SubtaskModel)
            .where(SubtaskModel.project_name == project_name)
            .where(SubtaskModel.is_active == True)
            .where(SubtaskModel.is_paused == False)  # Exclude paused subtasks
            .where(SubtaskModel.completion_status == "")
            .where(SubtaskModel.subtask_type == subtask_type)
            .where(SubtaskModel.assigned_user_id == user_id)
            .where(SubtaskModel.active_user_id == "")  # Not currently active
            .order_by(SubtaskModel.priority.desc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        result = db_session.execute(query).scalar_one_or_none()
        if result:
            logger.info(f"Found assigned subtask {result.subtask_id} for user {user_id}")
            return result

    # Strategy 2: Look for any available subtasks (prioritized by priority)
    for subtask_type in qualified_types:
        logger.info(f"Looking for available subtasks of type {subtask_type}")
        query = (
            select(SubtaskModel)
            .where(SubtaskModel.project_name == project_name)
            .where(SubtaskModel.is_active == True)
            .where(SubtaskModel.is_paused == False)  # Exclude paused subtasks
            .where(SubtaskModel.completion_status == "")
            .where(SubtaskModel.subtask_type == subtask_type)
            .where(SubtaskModel.active_user_id == "")  # Not currently active
            .order_by(SubtaskModel.priority.desc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        result = db_session.execute(query).scalar_one_or_none()
        if result:
            logger.info(f"Found available subtask {result.subtask_id} of type {subtask_type}")
            return result

    # Strategy 3: Look for idle subtasks (held by other users but not recently active)
    oldest_allowed_ts = time.time() - get_max_idle_seconds()
    for subtask_type in qualified_types:
        logger.info(f"Looking for idle subtasks of type {subtask_type}")
        query = (
            select(SubtaskModel)
            .where(SubtaskModel.project_name == project_name)
            .where(SubtaskModel.is_active == True)
            .where(SubtaskModel.is_paused == False)  # Exclude paused subtasks
            .where(SubtaskModel.completion_status == "")
            .where(SubtaskModel.subtask_type == subtask_type)
            .where(SubtaskModel.last_leased_ts < oldest_allowed_ts)
            .order_by(SubtaskModel.last_leased_ts.desc())
            .limit(1)
            .with_for_update(skip_locked=True)  # Skip if another transaction has it locked
        )
        result = db_session.execute(query).scalar_one_or_none()
        if result:
            # Refresh to get latest state and verify it's still idle
            db_session.refresh(result)
            if (
                result.last_leased_ts < oldest_allowed_ts
                and result.completion_status == ""
                and not result.is_paused
            ):
                return result

    return None


def get_subtask(
    *, project_name: str, subtask_id: str, db_session: Session | None = None
) -> Subtask:
    """
    Retrieve a subtask record from the database.

    :param project_name: The name of the project.
    :param subtask_id: The unique identifier of the subtask.
    :param db_session: Database session to use (optional).
    :return: The subtask record.
    :raises KeyError: If the subtask does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    with get_session_context(db_session) as session:
        query = (
            select(SubtaskModel)
            .where(SubtaskModel.subtask_id == subtask_id)
            .where(SubtaskModel.project_name == project_name)
        )
        try:
            subtask = session.execute(query).scalar_one()
            result = subtask.to_dict()
            return cast(Subtask, result)
        except NoResultFound as exc:
            raise KeyError(f"Subtask {subtask_id} not found") from exc


@typechecked
def pause_subtask(
    *, project_name: str, subtask_id: str, db_session: Session | None = None
) -> bool:
    """
    Pause a subtask to prevent it from being auto-selected.
    Paused subtasks can still be manually selected.

    :param project_name: The name of the project.
    :param subtask_id: The ID of the subtask to pause.
    :param db_session: Database session to use (optional).
    :return: True if the operation completes successfully
    :raises KeyError: If the subtask does not exist.
    """
    with get_session_context(db_session) as session:
        # Lock subtask to prevent race conditions
        locked_subtask_query = (
            select(SubtaskModel)
            .where(SubtaskModel.subtask_id == subtask_id)
            .where(SubtaskModel.project_name == project_name)
            .with_for_update()
        )
        locked_subtask = session.execute(locked_subtask_query).scalar_one()

        # Set paused state
        locked_subtask.is_paused = True
        session.commit()
        return True


@typechecked
def unpause_subtask(
    *, project_name: str, subtask_id: str, db_session: Session | None = None
) -> bool:
    """
    Unpause a subtask to allow it to be auto-selected again.

    :param project_name: The name of the project.
    :param subtask_id: The ID of the subtask to unpause.
    :param db_session: Database session to use (optional).
    :return: True if the operation completes successfully
    :raises KeyError: If the subtask does not exist.
    """
    with get_session_context(db_session) as session:
        # Lock subtask to prevent race conditions
        locked_subtask_query = (
            select(SubtaskModel)
            .where(SubtaskModel.subtask_id == subtask_id)
            .where(SubtaskModel.project_name == project_name)
            .with_for_update()
        )
        locked_subtask = session.execute(locked_subtask_query).scalar_one()

        # Set unpaused state
        locked_subtask.is_paused = False
        session.commit()
        return True


def _atomic_subtask_takeover(
    db_session: Session,
    project_name: str,
    selected_subtask: SubtaskModel,
    user_id: str,
    subtask_data: dict,
) -> float:
    """
    Perform atomic takeover of a subtask from another user.

    This function handles the complex locking and validation required to safely
    take over a subtask from another user, ensuring no race conditions occur.

    :param db_session: Database session to use
    :param project_name: The name of the project
    :param selected_subtask: The subtask to take over
    :param user_id: The user who wants to take over the subtask
    :param subtask_data: Current subtask data (for logging/validation)
    :return: The check_time when the takeover was performed
    :raises SubtaskValidationError: If the subtask is no longer available for takeover
    """
    logger.info(
        f"Attempting atomic takeover of subtask {selected_subtask.subtask_id} "
        f"from user {subtask_data['active_user_id']} to {user_id}"
    )

    # Lock subtask - this prevents any other transaction from using it
    locked_subtask_query = (
        select(SubtaskModel)
        .where(SubtaskModel.subtask_id == selected_subtask.subtask_id)
        .where(SubtaskModel.project_name == project_name)
        .with_for_update()
    )
    logger.info(f"User {user_id}: About to acquire lock on subtask {selected_subtask.subtask_id}")

    locked_subtask = db_session.execute(locked_subtask_query).scalar_one()

    # Force refresh from database to get the absolute latest committed state
    db_session.refresh(locked_subtask)

    # NOW read the current state after acquiring the lock - this is the definitive state
    current_subtask_data = locked_subtask.to_dict()

    check_time = time.time()

    # Check if subtask is still available for takeover using the FRESH locked data
    is_idle = current_subtask_data["last_leased_ts"] <= check_time - get_max_idle_seconds()
    is_same_user = current_subtask_data["active_user_id"] == user_id

    if not (is_idle or is_same_user):
        # Another transaction got it first or it's not idle anymore
        raise SubtaskValidationError("Subtask is no longer available for takeover")

    # Lock current user with FOR UPDATE
    locked_user_query = (
        select(UserModel)
        .where(UserModel.user_id == user_id)
        .where(UserModel.project_name == project_name)
        .with_for_update()
    )
    locked_user = db_session.execute(locked_user_query).scalar_one()

    # Lock previous user with FOR UPDATE (if different from current user)
    locked_prev_user = None
    if (
        current_subtask_data["active_user_id"] != user_id
        and current_subtask_data["active_user_id"] != ""
    ):
        locked_prev_user_query = (
            select(UserModel)
            .where(UserModel.user_id == current_subtask_data["active_user_id"])
            .where(UserModel.project_name == project_name)
            .with_for_update()
        )
        locked_prev_user = db_session.execute(locked_prev_user_query).scalar_one_or_none()

    # Perform takeover
    if locked_prev_user:
        locked_prev_user.active_subtask = ""
        logger.info(
            f"User {user_id}: Cleared active_subtask for previous user "
            f"{locked_prev_user.user_id}"
        )

    # Assign to current user
    locked_user.active_subtask = locked_subtask.subtask_id
    locked_subtask.active_user_id = user_id
    locked_subtask.last_leased_ts = check_time

    return check_time
