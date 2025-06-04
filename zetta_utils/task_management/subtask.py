# pylint: disable=singleton-comparison
import time
from typing import cast

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from zetta_utils.log import get_logger
from zetta_utils.task_management.utils import generate_id_nonunique

from .db.models import (
    DependencyModel,
    SubtaskModel,
    SubtaskTypeModel,
    TaskModel,
    UserModel,
)
from .exceptions import SubtaskValidationError, UserValidationError
from .types import Subtask, SubtaskUpdate

logger = get_logger(__name__)

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
def create_subtask(db_session: Session, project_name: str, data: Subtask) -> str:
    """Create a new subtask record"""
    # Validate subtask type exists
    query = (
        select(SubtaskTypeModel)
        .where(SubtaskTypeModel.project_name == project_name)
        .where(SubtaskTypeModel.subtask_type == data["subtask_type"])
    )
    try:
        db_session.execute(query).scalar_one()
    except NoResultFound as exc:
        raise SubtaskValidationError(f"Subtask type {data['subtask_type']} not found") from exc

    # Check if subtask already exists
    existing_query = (
        select(SubtaskModel)
        .where(SubtaskModel.subtask_id == data["subtask_id"])
        .where(SubtaskModel.project_name == project_name)
    )
    existing = db_session.execute(existing_query).scalar_one_or_none()
    if existing:
        raise SubtaskValidationError(f"Subtask {data['subtask_id']} already exists")

    # Create new subtask
    subtask_data = {**data, "id_nonunique": generate_id_nonunique()}
    model = SubtaskModel.from_dict(project_name, subtask_data)
    db_session.add(model)
    db_session.commit()

    return data["subtask_id"]


@typechecked
def update_subtask(
    db_session: Session, project_name: str, subtask_id: str, data: SubtaskUpdate
) -> bool:
    """Update a subtask record"""
    # Get current subtask
    query = (
        select(SubtaskModel)
        .where(SubtaskModel.subtask_id == subtask_id)
        .where(SubtaskModel.project_name == project_name)
    )
    try:
        subtask = db_session.execute(query).scalar_one()
    except NoResultFound as exc:
        raise KeyError(f"Subtask {subtask_id} not found") from exc

    current_data = subtask.to_dict()
    merged_data = {**current_data, **data}
    _validate_subtask(db_session, project_name, merged_data)

    # If completion status is changing, handle side effects
    if "completion_status" in data and "completed_user_id" in data:
        _handle_subtask_completion(
            db_session,
            project_name,
            subtask_id,
            data["completion_status"],
        )

    # Apply updates generically
    for field, value in data.items():
        if hasattr(subtask, field):
            setattr(subtask, field, value)

    db_session.commit()
    return True


@typechecked
def start_subtask(  # pylint: disable=too-many-branches
    db_session: Session, project_name: str, user_id: str, subtask_id: str | None = None
) -> str | None:
    user_query = (
        select(UserModel)
        .where(UserModel.user_id == user_id)
        .where(UserModel.project_name == project_name)
    )
    try:
        user = db_session.execute(user_query).scalar_one()
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
        # ATOMIC AUTO-SELECTION: Lock and select subtask atomically to prevent race conditions
        selected_subtask = _auto_select_subtask(db_session, project_name, user_id)
    elif subtask_id is not None:
        subtask_query = (
            select(SubtaskModel)
            .where(SubtaskModel.subtask_id == subtask_id)
            .where(SubtaskModel.project_name == project_name)
        )
        selected_subtask = db_session.execute(subtask_query).scalar_one_or_none()
        if not selected_subtask:
            raise SubtaskValidationError(f"Subtask {subtask_id} not found")
    else:
        subtask_query = (
            select(SubtaskModel)
            .where(SubtaskModel.subtask_id == current_active_subtask_id)
            .where(SubtaskModel.project_name == project_name)
        )
        selected_subtask = db_session.execute(subtask_query).scalar_one()

    if selected_subtask is not None:
        subtask_data = selected_subtask.to_dict()
        _validate_subtask(db_session, project_name, subtask_data)

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
                db_session,
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
            locked_user = db_session.execute(locked_user_query).scalar_one()

            locked_subtask_query = (
                select(SubtaskModel)
                .where(SubtaskModel.subtask_id == selected_subtask.subtask_id)
                .where(SubtaskModel.project_name == project_name)
                .with_for_update()
            )
            locked_subtask = db_session.execute(locked_subtask_query).scalar_one()

            # Assign subtask to user
            locked_user.active_subtask = locked_subtask.subtask_id
            locked_subtask.active_user_id = user_id
            locked_subtask.last_leased_ts = current_time

        db_session.flush()  # Ensure changes are written to DB before commit
        db_session.commit()
        return str(selected_subtask.subtask_id)
    return None


def release_subtask(
    db_session: Session,
    project_name: str,
    user_id: str,
    subtask_id: str,
    completion_status: str = "",
) -> bool:
    """
    Releases the active subtask for a user within the project.

    :param db_session: Database session to use.
    :param project_name: The name of the project.
    :param user_id: The unique identifier of the user.
    :param subtask_id: The ID of the subtask to release.
    :param completion_status: The completion status to set for the subtask upon release.
        Empty string means not completed.
    :return: True if the operation completes successfully
    :raises SubtaskValidationError: If the subtask validation fails
    :raises UserValidationError: If the user validation fails
    :raises ValueError: If the completion status is invalid
    :raises RuntimeError: If the database operation fails.
    """
    # Get user
    user_query = (
        select(UserModel)
        .where(UserModel.user_id == user_id)
        .where(UserModel.project_name == project_name)
    )
    try:
        user = db_session.execute(user_query).scalar_one()
    except NoResultFound as exc:
        raise UserValidationError(f"User {user_id} not found") from exc

    if user.active_subtask == "":
        raise UserValidationError("User does not have an active subtask")

    # Check that the subtask being released matches the user's active subtask
    if user.active_subtask != subtask_id:
        raise UserValidationError("Subtask ID does not match user's active subtask")

    # Get subtask
    subtask_query = (
        select(SubtaskModel)
        .where(SubtaskModel.subtask_id == subtask_id)
        .where(SubtaskModel.project_name == project_name)
    )
    try:
        subtask = db_session.execute(subtask_query).scalar_one()
    except NoResultFound as exc:
        raise SubtaskValidationError(f"Subtask {subtask_id} not found") from exc

    # Handle completion side effects if completing
    if completion_status:
        _handle_subtask_completion(
            db_session,
            project_name,
            subtask_id,
            completion_status,
        )

    # Release the subtask
    subtask.active_user_id = ""
    subtask.completion_status = completion_status
    subtask.completed_user_id = user_id if completion_status else ""
    user.active_subtask = ""

    db_session.commit()
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


def _handle_subtask_completion(
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

    # Pre-lock all affected subtasks and their dependencies in order
    locked_subtasks = {}
    locked_dependencies = {}

    for affected_subtask_id in sorted_subtask_ids:
        # Lock the subtask itself
        subtask_lock_query = (
            select(SubtaskModel)
            .where(SubtaskModel.subtask_id == affected_subtask_id)
            .where(SubtaskModel.project_name == project_name)
            .with_for_update()
        )
        locked_subtasks[affected_subtask_id] = db_session.execute(subtask_lock_query).scalar_one()

        # Lock all dependencies for this subtask
        deps_lock_query = (
            select(DependencyModel)
            .where(DependencyModel.subtask_id == affected_subtask_id)
            .where(DependencyModel.project_name == project_name)
            .with_for_update()
        )
        locked_dependencies[affected_subtask_id] = list(
            db_session.execute(deps_lock_query).scalars().all()
        )

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

    # Check if task is complete - ATOMIC: Lock task first to prevent race conditions
    task_id = subtask_data["task_id"]

    # Lock the task to prevent race conditions during completion check
    task_lock_query = (
        select(TaskModel)
        .where(TaskModel.task_id == task_id)
        .where(TaskModel.project_name == project_name)
        .with_for_update()
    )
    locked_task = db_session.execute(task_lock_query).scalar_one_or_none()

    if locked_task:
        # Now check incomplete subtasks with the task locked
        incomplete_subtasks_query = (
            select(SubtaskModel)
            .where(SubtaskModel.task_id == task_id)
            .where(SubtaskModel.project_name == project_name)
            .where(SubtaskModel.is_active == True)
            .where(SubtaskModel.completion_status == "")
        )
        incomplete_subtasks = db_session.execute(incomplete_subtasks_query).scalars().all()

        # If this was the last incomplete subtask (current subtask is still counted as incomplete)
        if len(list(incomplete_subtasks)) == 1:
            locked_task.status = "fully_processed"

    db_session.commit()


def _auto_select_subtask(
    db_session: Session, project_name: str, user_id: str
) -> SubtaskModel | None:
    """Auto-select a subtask for a user based on priority and qualifications.
    Uses atomic locking to prevent race conditions.

    Selection criteria (in order):
    1. Assigned to user & matches qualified types (highest priority)
    2. Unassigned & matches qualified types (highest priority)
    3. Matches qualified types & idle > max idle seconds (most recently active)
    """
    user_query = (
        select(UserModel)
        .where(UserModel.user_id == user_id)
        .where(UserModel.project_name == project_name)
    )
    user = db_session.execute(user_query).scalar_one()

    qualified_types = cast(list[str], user.qualified_subtask_types)
    if not qualified_types:
        return None

    current_time = time.time()

    # 1. Check for tasks already assigned to user
    for subtask_type in qualified_types:
        query = (
            select(SubtaskModel)
            .where(SubtaskModel.project_name == project_name)
            .where(SubtaskModel.is_active == True)
            .where(SubtaskModel.is_paused == False)  # Exclude paused subtasks
            .where(SubtaskModel.assigned_user_id == user_id)
            .where(SubtaskModel.active_user_id == "")
            .where(SubtaskModel.completion_status == "")
            .where(SubtaskModel.subtask_type == subtask_type)
            .order_by(SubtaskModel.priority.desc())
            .limit(1)
            .with_for_update(skip_locked=True)  # Skip if another transaction has it locked
        )
        result = db_session.execute(query).scalar_one_or_none()
        if result:
            # Refresh to get latest state and verify it's still available
            db_session.refresh(result)
            if (
                result.active_user_id == ""
                and result.completion_status == ""
                and not result.is_paused
            ):
                return result

    # 2. Check for unassigned tasks
    for subtask_type in qualified_types:
        query = (
            select(SubtaskModel)
            .where(SubtaskModel.project_name == project_name)
            .where(SubtaskModel.is_active == True)
            .where(SubtaskModel.is_paused == False)  # Exclude paused subtasks
            .where(SubtaskModel.assigned_user_id == "")
            .where(SubtaskModel.active_user_id == "")
            .where(SubtaskModel.completion_status == "")
            .where(SubtaskModel.subtask_type == subtask_type)
            .order_by(SubtaskModel.priority.desc())
            .limit(1)
            .with_for_update(skip_locked=True)  # Skip if another transaction has it locked
        )
        result = db_session.execute(query).scalar_one_or_none()
        if result:
            # Refresh to get latest state and verify it's still available
            db_session.refresh(result)
            if (
                result.active_user_id == ""
                and result.completion_status == ""
                and not result.is_paused
            ):
                return result

    # 3. Check for idle tasks
    oldest_allowed_ts = current_time - get_max_idle_seconds()
    for subtask_type in qualified_types:
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


def get_subtask(db_session: Session, project_name: str, subtask_id: str) -> Subtask:
    """
    Retrieve a subtask record from the database.

    :param db_session: Database session to use.
    :param project_name: The name of the project.
    :param subtask_id: The unique identifier of the subtask.
    :return: The subtask record.
    :raises KeyError: If the subtask does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    query = (
        select(SubtaskModel)
        .where(SubtaskModel.subtask_id == subtask_id)
        .where(SubtaskModel.project_name == project_name)
    )
    try:
        subtask = db_session.execute(query).scalar_one()
        result = subtask.to_dict()
        return cast(Subtask, result)
    except NoResultFound as exc:
        raise KeyError(f"Subtask {subtask_id} not found") from exc


@typechecked
def pause_subtask(db_session: Session, project_name: str, subtask_id: str) -> bool:
    """
    Pause a subtask to prevent it from being auto-selected.
    Paused subtasks can still be manually selected.

    :param db_session: Database session to use.
    :param project_name: The name of the project.
    :param subtask_id: The ID of the subtask to pause.
    :return: True if the operation completes successfully
    :raises KeyError: If the subtask does not exist.
    """
    # Lock subtask to prevent race conditions
    locked_subtask_query = (
        select(SubtaskModel)
        .where(SubtaskModel.subtask_id == subtask_id)
        .where(SubtaskModel.project_name == project_name)
        .with_for_update()
    )
    try:
        locked_subtask = db_session.execute(locked_subtask_query).scalar_one()
    except NoResultFound as exc:
        raise KeyError(f"Subtask {subtask_id} not found") from exc

    # Set paused state
    locked_subtask.is_paused = True
    db_session.commit()
    return True


@typechecked
def unpause_subtask(db_session: Session, project_name: str, subtask_id: str) -> bool:
    """
    Unpause a subtask to allow it to be auto-selected again.

    :param db_session: Database session to use.
    :param project_name: The name of the project.
    :param subtask_id: The ID of the subtask to unpause.
    :return: True if the operation completes successfully
    :raises KeyError: If the subtask does not exist.
    """
    # Lock subtask to prevent race conditions
    locked_subtask_query = (
        select(SubtaskModel)
        .where(SubtaskModel.subtask_id == subtask_id)
        .where(SubtaskModel.project_name == project_name)
        .with_for_update()
    )
    try:
        locked_subtask = db_session.execute(locked_subtask_query).scalar_one()
    except NoResultFound as exc:
        raise KeyError(f"Subtask {subtask_id} not found") from exc

    # Set unpaused state
    locked_subtask.is_paused = False
    db_session.commit()
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
