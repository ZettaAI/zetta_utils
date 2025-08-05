# pylint: disable=singleton-comparison
import time
from typing import Any, cast

from sqlalchemy import func, select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from zetta_utils import log
from zetta_utils.task_management.utils import generate_id_nonunique

from .db.models import DependencyModel, TaskModel, TaskTypeModel, UserModel
from .db.session import get_session_context
from .exceptions import TaskValidationError, UserValidationError
from .ng_state.segment import get_segment_ng_state
from .types import Task, TaskUpdate

logger = log.get_logger("zetta_utils.task_management.task")

_MAX_IDLE_SECONDS = 90


def get_max_idle_seconds() -> float:
    return _MAX_IDLE_SECONDS


def _validate_task(db_session: Session, project_name: str, task: dict) -> Task:
    """
    Validate that a task's data is consistent and valid.

    :param db_session: Database session to use
    :param project_name: The name of the project.
    :param task: The task data to validate
    :raises TaskValidationError: If the task data is invalid
    """
    try:
        # Get task type from database
        query = (
            select(TaskTypeModel)
            .where(TaskTypeModel.project_name == project_name)
            .where(TaskTypeModel.task_type == task["task_type"])
        )
        task_type = db_session.execute(query).scalar_one()
        task_type_dict = task_type.to_dict()
    except NoResultFound as e:
        raise TaskValidationError(f"Task type not found: {task['task_type']}") from e

    # If task is not active, it cannot have completion status or completed user
    if not task.get("is_active"):
        if task.get("completion_status") != "":
            raise TaskValidationError("Inactive task cannot have completion status")
        if task.get("completed_user_id") != "":
            raise TaskValidationError("Inactive task cannot have completed user")
        return Task(**task)  # type: ignore # typeguard will check for us

    # Validate completion status if present
    completion_status = task.get("completion_status", "")
    if completion_status != "":
        if "completion_statuses" not in task_type_dict:
            raise TaskValidationError(
                f"Task type {task['task_type']} has no valid completion statuses"
            )
        if completion_status not in task_type_dict["completion_statuses"]:
            raise TaskValidationError(
                f"Completion status '{completion_status}' not allowed for this task type"
            )

        # If there's a completion status, must have completed user
        if not task.get("completed_user_id"):
            raise TaskValidationError("Completed task must have completed_user_id")

    return Task(**task)  # type: ignore # typeguard will check for us


@typechecked
def create_task(*, project_name: str, data: Task, db_session: Session | None = None) -> str:
    """Create a new task record"""
    with get_session_context(db_session) as session:
        # Validate task type exists
        query = (
            select(TaskTypeModel)
            .where(TaskTypeModel.project_name == project_name)
            .where(TaskTypeModel.task_type == data["task_type"])
        )
        try:
            session.execute(query).scalar_one()
        except NoResultFound as exc:
            raise TaskValidationError(f"Task type {data['task_type']} not found") from exc

        # Check if task already exists
        existing_query = (
            select(TaskModel)
            .where(TaskModel.task_id == data["task_id"])
            .where(TaskModel.project_name == project_name)
        )
        existing = session.execute(existing_query).scalar_one_or_none()
        if existing:
            raise TaskValidationError(f"Task {data['task_id']} already exists")

        # Create new task
        task_data = {**data, "id_nonunique": generate_id_nonunique()}
        model = TaskModel.from_dict(project_name, task_data)
        session.add(model)
        session.commit()

        return data["task_id"]


@typechecked
def update_task(
    *, project_name: str, task_id: str, data: TaskUpdate, db_session: Session | None = None
) -> bool:
    """Update a task record"""
    with get_session_context(db_session) as session:
        # Get current task
        query = (
            select(TaskModel)
            .where(TaskModel.task_id == task_id)
            .where(TaskModel.project_name == project_name)
        )
        try:
            task = session.execute(query).scalar_one()
        except NoResultFound as exc:
            raise KeyError(f"Task {task_id} not found") from exc

        current_data = task.to_dict()
        merged_data = {**current_data, **data}
        _validate_task(session, project_name, merged_data)

        # If completion status is changing, handle side effects
        if "completion_status" in data and "completed_user_id" in data:
            _handle_task_completion(
                session,
                project_name,
                task_id,
                data["completion_status"],
            )

        # Apply updates generically
        for field, value in data.items():
            if hasattr(task, field):
                setattr(task, field, value)

        session.commit()
        return True


@typechecked
def start_task(  # pylint: disable=too-many-branches
    *,
    project_name: str,
    user_id: str,
    task_id: str | None = None,
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

        current_active_task_id = user.active_task
        if (
            task_id is not None
            and current_active_task_id != ""
            and current_active_task_id != task_id
        ):
            raise UserValidationError(
                f"User already has an active task {current_active_task_id} "
                f"which is different from requested task {task_id}"
            )

        if task_id is None and current_active_task_id == "":
            selected_task = _auto_select_task(session, project_name, user_id)
        elif task_id is not None:
            task_query = (
                select(TaskModel)
                .where(TaskModel.task_id == task_id)
                .where(TaskModel.project_name == project_name)
            )
            selected_task = session.execute(task_query).scalar_one_or_none()
            if not selected_task:
                raise TaskValidationError(f"Task {task_id} not found")
        else:
            task_query = (
                select(TaskModel)
                .where(TaskModel.task_id == current_active_task_id)
                .where(TaskModel.project_name == project_name)
            )
            selected_task = session.execute(task_query).scalar_one()

        if selected_task is not None:
            task_data = selected_task.to_dict()
            _validate_task(session, project_name, task_data)

            # Check if user is qualified for this task type
            if user.qualified_task_types is not None and len(user.qualified_task_types) == 0:
                raise UserValidationError("User not qualified for this task type")
            if (
                user.qualified_task_types
                and task_data["task_type"] not in user.qualified_task_types
            ):
                raise UserValidationError("User not qualified for this task type")

            current_time = time.time()
            # Check if task is idle and can be taken over
            if task_data["active_user_id"] != "":
                _atomic_task_takeover(
                    session,
                    project_name,
                    selected_task,
                    user_id,
                    task_data,
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

                locked_task_query = (
                    select(TaskModel)
                    .where(TaskModel.task_id == selected_task.task_id)
                    .where(TaskModel.project_name == project_name)
                    .with_for_update()
                )
                locked_task = session.execute(locked_task_query).scalar_one()

                # Assign task to user
                locked_user.active_task = locked_task.task_id
                locked_task.active_user_id = user_id
                locked_task.last_leased_ts = current_time

                # Set first_start_ts only if it's null (first time starting this task)
                if locked_task.first_start_ts is None:
                    locked_task.first_start_ts = current_time

            session.flush()  # Ensure changes are written to DB before commit
            session.commit()
            return str(selected_task.task_id)
        return None


def release_task(
    *,
    project_name: str,
    user_id: str,
    task_id: str,
    completion_status: str = "",
    note: str | None = None,
    db_session: Session | None = None,
) -> bool:
    """
    Releases the active task for a user within the project.

    :param project_name: The name of the project.
    :param user_id: The unique identifier of the user.
    :param task_id: The ID of the task to release.
    :param completion_status: The completion status to set for the task upon release.
        Empty string means not completed.
    :param note: Optional note to save with the task.
    :param db_session: Database session to use (optional).
    :return: True if the operation completes successfully
    :raises TaskValidationError: If the task validation fails
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

        if user.active_task == "":
            raise UserValidationError("User does not have an active task")

        if user.active_task != task_id:
            raise UserValidationError("Task ID does not match user's active task")

        # Get task
        task_query = (
            select(TaskModel)
            .where(TaskModel.task_id == task_id)
            .where(TaskModel.project_name == project_name)
        )
        try:
            task = session.execute(task_query).scalar_one()
        except NoResultFound as exc:
            raise TaskValidationError(f"Task {task_id} not found") from exc

        # Update task status FIRST (before handling side effects)
        # This prevents race conditions in task completion detection
        task.active_user_id = ""
        task.completion_status = completion_status
        task.completed_user_id = user_id if completion_status else ""
        if note is not None:
            task.note = note
        user.active_task = ""

        # Flush changes to ensure they're visible within this transaction
        # (but not committed yet in case side effects fail)
        session.flush()

        # Handle completion side effects AFTER marking complete
        if completion_status:
            _handle_task_completion(
                session,
                project_name,
                task_id,
                completion_status,
            )

        session.commit()
        return True


def _satisfy_dependency_and_activate_if_ready(
    dep: DependencyModel,
    dep_data: dict,
    locked_dependencies: dict,
    locked_tasks: dict,
) -> None:
    """
    Mark a dependency as satisfied and activate the dependent task if all dependencies are met.

    :param dep: The dependency model to satisfy
    :param dep_data: Dictionary representation of the dependency data
    :param locked_dependencies: Dictionary of pre-locked dependencies by task_id
    :param locked_tasks: Dictionary of pre-locked tasks by task_id
    """
    logger.info(f"Dependency {dep.dependency_id} satisfied")

    # Mark dependency satisfied
    dep.is_satisfied = True

    # Check if dependent task can be activated
    dependent_task_id = dep_data["task_id"]

    # Use pre-locked dependencies to count remaining unsatisfied
    remaining_unsatisfied = sum(
        1 for locked_dep in locked_dependencies[dependent_task_id] if not locked_dep.is_satisfied
    )

    logger.info(f"Remaining dependencies: {remaining_unsatisfied}")

    if remaining_unsatisfied == 0:  # Current dep will be satisfied
        logger.info(f"Activating dependent task {dependent_task_id}")
        print(f"Activating dependent task {dependent_task_id}")

        # Use pre-locked task
        dependent_task = locked_tasks[dependent_task_id]
        dependent_task.is_active = True


def _handle_task_completion(  # pylint: disable=too-many-locals
    db_session: Session,
    project_name: str,
    task_id: str,
    completion_status: str,
) -> None:
    """
    Handle all side effects of completing a task.

    :param db_session: The current database session
    :param project_name: The name of the project
    :param task_id: The ID of completed task
    :param completion_status: The new completion status
    :raises RuntimeError: If the database operation fails
    """
    logger.info(f"Handling task completion side effects for `{task_id}`")

    # Get the task
    task_query = (
        select(TaskModel)
        .where(TaskModel.task_id == task_id)
        .where(TaskModel.project_name == project_name)
    )
    db_session.execute(task_query).scalar_one()

    # Get all dependencies that need to be checked/updated for this completion
    # We need to find all tasks that might be affected to implement proper lock ordering
    deps_query = (
        select(DependencyModel)
        .where(DependencyModel.dependent_on_task_id == task_id)
        .where(DependencyModel.project_name == project_name)
        .where(DependencyModel.is_satisfied == False)
    )
    deps = db_session.execute(deps_query).scalars().all()

    # Collect all task IDs that need locking for proper ordering
    affected_task_ids = set()
    for dep in deps:
        affected_task_ids.add(dep.task_id)

    # Always lock in alphabetical order to prevent deadlocks
    sorted_task_ids = sorted(affected_task_ids)

    # Pre-lock all affected tasks in order
    locked_tasks = {}
    for affected_task_id in sorted_task_ids:
        locked_task_query = (
            select(TaskModel)
            .where(TaskModel.task_id == affected_task_id)
            .where(TaskModel.project_name == project_name)
            .with_for_update()
        )
        locked_task = db_session.execute(locked_task_query).scalar_one()
        locked_tasks[affected_task_id] = locked_task

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

        # Group by task_id for easier processing
        locked_task_id = locked_dep.task_id
        if locked_task_id not in locked_dependencies:
            locked_dependencies[locked_task_id] = []
        locked_dependencies[locked_task_id].append(locked_dep)

    # Now process each dependency
    for dep in deps:
        dep_data = dep.to_dict()
        if dep_data["required_completion_status"] == completion_status:
            _satisfy_dependency_and_activate_if_ready(
                dep, dep_data, locked_dependencies, locked_tasks
            )


def _auto_select_task(db_session: Session, project_name: str, user_id: str) -> TaskModel | None:
    """
    Auto-select a task for a user atomically.

    This function performs atomic selection of an appropriate task for a user,
    taking into account their qualifications and existing assignments.

    :param db_session: Database session to use
    :param project_name: The name of the project
    :param user_id: The user requesting a task
    :return: The selected task model or None if no suitable task available
    :raises UserValidationError: If the user is not qualified for any tasks
    """
    logger.info(f"Auto-selecting task for user {user_id}")

    # Get user qualifications
    user_query = (
        select(UserModel)
        .where(UserModel.user_id == user_id)
        .where(UserModel.project_name == project_name)
    )
    user = db_session.execute(user_query).scalar_one()

    qualified_types = user.qualified_task_types or []
    if not qualified_types:
        logger.info(f"User {user_id} has no qualified task types")
        return None

    logger.info(f"User {user_id} is qualified for: {qualified_types}")

    # Strategy 1: Look for tasks explicitly assigned to this user
    logger.info(f"Looking for assigned tasks for user {user_id}")
    query = (
        select(TaskModel)
        .where(TaskModel.project_name == project_name)
        .where(TaskModel.is_active == True)
        .where(TaskModel.is_paused == False)  # Exclude paused tasks
        .where(TaskModel.completion_status == "")
        .where(TaskModel.task_type.in_(qualified_types))
        .where(TaskModel.assigned_user_id == user_id)
        .where(TaskModel.active_user_id == "")  # Not currently active
        .order_by(TaskModel.priority.desc(), TaskModel.id_nonunique)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    result = db_session.execute(query).scalar_one_or_none()
    if result:
        logger.info(f"Found assigned task {result.task_id} for user {user_id}")
        return result

    # Strategy 2: Look for any available tasks (prioritized by priority)
    logger.info("Looking for available tasks")
    query = (
        select(TaskModel)
        .where(TaskModel.project_name == project_name)
        .where(TaskModel.is_active == True)
        .where(TaskModel.is_paused == False)  # Exclude paused tasks
        .where(TaskModel.completion_status == "")
        .where(TaskModel.task_type.in_(qualified_types))
        .where(TaskModel.active_user_id == "")  # Not currently active
        .order_by(TaskModel.priority.desc(), TaskModel.id_nonunique)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    result = db_session.execute(query).scalar_one_or_none()
    if result:
        logger.info(f"Found available task {result.task_id} of type {result.task_type}")
        return result

    # Strategy 3: Look for idle tasks (held by other users but not recently active)
    oldest_allowed_ts = time.time() - get_max_idle_seconds()
    logger.info("Looking for idle tasks")
    query = (
        select(TaskModel)
        .where(TaskModel.project_name == project_name)
        .where(TaskModel.is_active == True)
        .where(TaskModel.is_paused == False)  # Exclude paused tasks
        .where(TaskModel.completion_status == "")
        .where(TaskModel.task_type.in_(qualified_types))
        .where(TaskModel.last_leased_ts < oldest_allowed_ts)
        .order_by(TaskModel.priority.desc(), TaskModel.last_leased_ts.desc())
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


def get_task(*, project_name: str, task_id: str, process_ng_state: bool = True, db_session: Session | None = None) -> Task:
    """
    Retrieve a task record from the database.

    :param project_name: The name of the project.
    :param task_id: The unique identifier of the task.
    :param process_ng_state: Whether to process ng_state seed_id format (default: True).
    :param db_session: Database session to use (optional).
    :return: The task record.
    :raises KeyError: If the task does not exist.
    :raises RuntimeError: If the database operation fails.
    """
    with get_session_context(db_session) as session:
        query = (
            select(TaskModel)
            .where(TaskModel.task_id == task_id)
            .where(TaskModel.project_name == project_name)
        )
        try:
            task = session.execute(query).scalar_one()
            # Handle ng_state and ng_state_initial special formats
            if process_ng_state:
                _process_ng_state_seed_id(session, project_name, task)

            result = task.to_dict()
            return cast(Task, result)
        except NoResultFound as exc:
            raise KeyError(f"Task {task_id} not found") from exc


@typechecked
def pause_task(*, project_name: str, task_id: str, db_session: Session | None = None) -> bool:
    """
    Pause a task to prevent it from being auto-selected.
    Paused tasks can still be manually selected.

    :param project_name: The name of the project.
    :param task_id: The ID of the task to pause.
    :param db_session: Database session to use (optional).
    :return: True if the operation completes successfully
    :raises KeyError: If the task does not exist.
    """
    with get_session_context(db_session) as session:
        # Lock task to prevent race conditions
        locked_task_query = (
            select(TaskModel)
            .where(TaskModel.task_id == task_id)
            .where(TaskModel.project_name == project_name)
            .with_for_update()
        )
        locked_task = session.execute(locked_task_query).scalar_one()

        # Set paused state
        locked_task.is_paused = True
        session.commit()
        return True


@typechecked
def unpause_task(*, project_name: str, task_id: str, db_session: Session | None = None) -> bool:
    """
    Unpause a task to allow it to be auto-selected again.

    :param project_name: The name of the project.
    :param task_id: The ID of the task to unpause.
    :param db_session: Database session to use (optional).
    :return: True if the operation completes successfully
    :raises KeyError: If the task does not exist.
    """
    with get_session_context(db_session) as session:
        # Lock task to prevent race conditions
        locked_task_query = (
            select(TaskModel)
            .where(TaskModel.task_id == task_id)
            .where(TaskModel.project_name == project_name)
            .with_for_update()
        )
        locked_task = session.execute(locked_task_query).scalar_one()

        # Set unpaused state
        locked_task.is_paused = False
        session.commit()
        return True


@typechecked
def reactivate_task(*, project_name: str, task_id: str, db_session: Session | None = None) -> bool:
    """
    Reactivate a completed task by clearing its completion status and completed user.

    :param project_name: The name of the project
    :param task_id: The ID of the task to reactivate
    :param db_session: Database session to use (optional)
    :return: True if the operation completes successfully
    :raises KeyError: If the task does not exist
    :raises TaskValidationError: If the task cannot be reactivated
    """
    with get_session_context(db_session) as session:
        # Get and lock the task
        task_query = (
            select(TaskModel)
            .where(TaskModel.task_id == task_id)
            .where(TaskModel.project_name == project_name)
            .with_for_update()
        )
        try:
            task = session.execute(task_query).scalar_one()
        except NoResultFound as exc:
            raise KeyError(f"Task {task_id} not found") from exc

        # Validate task can be reactivated
        if task.completion_status == "":
            raise TaskValidationError(f"Task {task_id} is already active (not completed)")

        # Clear completion status and completed user
        task.completion_status = ""
        task.completed_user_id = ""

        session.commit()
        logger.info(f"Reactivated task {task_id} in project {project_name}")
        return True


def get_paused_tasks_by_user(
    *, project_name: str, user_id: str, db_session: Session | None = None
) -> list[Task]:
    """
    Get all paused tasks assigned to a specific user.

    :param project_name: The name of the project
    :param user_id: The ID of the user
    :param db_session: Database session to use (optional)
    :return: List of paused tasks assigned to the user
    """
    with get_session_context(db_session) as session:
        query = (
            select(TaskModel)
            .where(TaskModel.project_name == project_name)
            .where(TaskModel.assigned_user_id == user_id)
            .where(TaskModel.is_paused == True)
            .where(TaskModel.is_active == True)
            .where(TaskModel.completion_status == "")
            .order_by(TaskModel.priority.desc(), TaskModel.task_id)
        )

        tasks = session.execute(query).scalars().all()
        return [cast(Task, task.to_dict()) for task in tasks]


def list_tasks_summary(*, project_name: str, db_session: Session | None = None) -> dict:
    """
    Get a summary of tasks in a project with counts and sample task IDs.

    :param project_name: The name of the project
    :param db_session: Database session to use (optional)
    :return: Dictionary with counts and task ID lists
    """
    with get_session_context(db_session) as session:
        # Count active tasks (empty completion status)
        active_count_query = (
            select(func.count(TaskModel.task_id))
            .where(TaskModel.project_name == project_name)
            .where(TaskModel.is_active == True)
            .where(TaskModel.completion_status == "")
        )
        active_count = session.execute(active_count_query).scalar() or 0

        # Count completed tasks (non-empty completion status)
        completed_count_query = (
            select(func.count(TaskModel.task_id))
            .where(TaskModel.project_name == project_name)
            .where(TaskModel.is_active == True)
            .where(TaskModel.completion_status != "")
        )
        completed_count = session.execute(completed_count_query).scalar() or 0

        # Count paused tasks
        paused_count_query = (
            select(func.count(TaskModel.task_id))
            .where(TaskModel.project_name == project_name)
            .where(TaskModel.is_paused == True)
        )
        paused_count = session.execute(paused_count_query).scalar() or 0

        # Get first 5 active unpaused task IDs
        active_unpaused_query = (
            select(TaskModel.task_id)
            .where(TaskModel.project_name == project_name)
            .where(TaskModel.is_active == True)
            .where(TaskModel.is_paused == False)
            .where(TaskModel.completion_status == "")
            .order_by(TaskModel.priority.desc(), TaskModel.task_id)
            .limit(5)
        )
        active_unpaused_ids = list(session.execute(active_unpaused_query).scalars().all())

        # Get first 5 active paused task IDs
        active_paused_query = (
            select(TaskModel.task_id)
            .where(TaskModel.project_name == project_name)
            .where(TaskModel.is_active == True)
            .where(TaskModel.is_paused == True)
            .where(TaskModel.completion_status == "")
            .order_by(TaskModel.priority.desc(), TaskModel.task_id)
            .limit(5)
        )
        active_paused_ids = list(session.execute(active_paused_query).scalars().all())

        return {
            "active_count": active_count,
            "completed_count": completed_count,
            "paused_count": paused_count,
            "active_unpaused_ids": active_unpaused_ids,
            "active_paused_ids": active_paused_ids,
        }


def _atomic_task_takeover(
    db_session: Session,
    project_name: str,
    selected_task: TaskModel,
    user_id: str,
    task_data: dict,
) -> float:
    """
    Perform atomic takeover of a task from another user.

    This function handles the complex locking and validation required to safely
    take over a task from another user, ensuring no race conditions occur.

    :param db_session: Database session to use
    :param project_name: The name of the project
    :param selected_task: The task to take over
    :param user_id: The user who wants to take over the task
    :param task_data: Current task data (for logging/validation)
    :return: The check_time when the takeover was performed
    :raises TaskValidationError: If the task is no longer available for takeover
    """
    logger.info(
        f"Attempting atomic takeover of task {selected_task.task_id} "
        f"from user {task_data['active_user_id']} to {user_id}"
    )

    # Lock task - this prevents any other transaction from using it
    locked_task_query = (
        select(TaskModel)
        .where(TaskModel.task_id == selected_task.task_id)
        .where(TaskModel.project_name == project_name)
        .with_for_update()
    )
    logger.info(f"User {user_id}: About to acquire lock on task {selected_task.task_id}")

    locked_task = db_session.execute(locked_task_query).scalar_one()

    # Force refresh from database to get the absolute latest committed state
    db_session.refresh(locked_task)

    # NOW read the current state after acquiring the lock - this is the definitive state
    current_task_data = locked_task.to_dict()

    check_time = time.time()

    # Check if task is still available for takeover using the FRESH locked data
    is_idle = current_task_data["last_leased_ts"] <= check_time - get_max_idle_seconds()
    is_same_user = current_task_data["active_user_id"] == user_id

    if not (is_idle or is_same_user):
        # Another transaction got it first or it's not idle anymore
        raise TaskValidationError("Task is no longer available for takeover")

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
        current_task_data["active_user_id"] != user_id
        and current_task_data["active_user_id"] != ""
    ):
        locked_prev_user_query = (
            select(UserModel)
            .where(UserModel.user_id == current_task_data["active_user_id"])
            .where(UserModel.project_name == project_name)
            .with_for_update()
        )
        locked_prev_user = db_session.execute(locked_prev_user_query).scalar_one_or_none()

    # Perform takeover
    if locked_prev_user:
        locked_prev_user.active_task = ""
        logger.info(
            f"User {user_id}: Cleared active_task for previous user " f"{locked_prev_user.user_id}"
        )

    # Assign to current user
    locked_user.active_task = locked_task.task_id
    locked_task.active_user_id = user_id
    locked_task.last_leased_ts = check_time

    # Set first_start_ts only if it's null (first time starting this task)
    if locked_task.first_start_ts is None:
        locked_task.first_start_ts = check_time

    return check_time


def _process_ng_state_seed_id(session: Session, project_name: str, task: TaskModel) -> None:
    """
    Process ng_state and ng_state_initial seed_id formats and update the database.
    
    Detects patterns like {"seed_id": 74732294451380972} in ng_state and ng_state_initial 
    and constructs the corresponding neuroglancer state, then updates the database.
    
    :param session: Database session
    :param project_name: The name of the project
    :param task: The task model
    """
    # Process ng_state
    ng_state = task.ng_state

    if isinstance(ng_state, dict) and "seed_id" in ng_state and len(ng_state) == 1:
        seed_id = ng_state["seed_id"]
        if isinstance(seed_id, int):
            logger.info(f"Processing ng_state seed_id {seed_id} for task {task.task_id}")
            
            try:
                # Generate neuroglancer state for the segment
                generated_ng_state = get_segment_ng_state(
                    project_name=project_name,
                    seed_id=seed_id,
                    include_certain_ends=True,
                    include_uncertain_ends=True,
                    include_breadcrumbs=True,
                    include_segment_type_layers=True,
                    db_session=session
                )
                
                # Update the database
                task.ng_state = generated_ng_state
                task.ng_state_initial = generated_ng_state
                session.commit()
                
                logger.info(f"Successfully generated and saved ng_state for seed_id {seed_id} in task {task.task_id}")
                
            except Exception as e:
                logger.error(f"Failed to generate ng_state for seed_id {seed_id} in task {task.task_id}: {e}")
                session.rollback()
