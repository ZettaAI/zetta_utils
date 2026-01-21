# pylint: disable=singleton-comparison,too-many-lines
import time
from datetime import datetime, timezone
from typing import Any, TypedDict, cast

from sqlalchemy import BigInteger, and_, func, or_, select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from typeguard import typechecked

from zetta_utils import log
from zetta_utils.task_management.utils import generate_id_nonunique

from .db.models import (
    DependencyModel,
    SegmentModel,
    SegmentTypeModel,
    TaskFeedbackModel,
    TaskModel,
    TaskTypeModel,
    UserModel,
)
from .db.session import get_session_context
from .exceptions import TaskValidationError, UserValidationError
from .ng_state.segment import get_segment_ng_state
from .types import Task, TaskUpdate

logger = log.get_logger("zetta_utils.task_management.task")

_MAX_IDLE_SECONDS = 90


def get_max_idle_seconds() -> float:
    return _MAX_IDLE_SECONDS


class _UserProjectQualifications(TypedDict):
    user_id: str
    task_types: list[str]
    segment_types: list[str]


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

        # Create new task with created_at timestamp
        task_data = {
            **data,
            "id_nonunique": generate_id_nonunique(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
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
def start_task(  # pylint: disable=too-many-branches disable=too-many-statements
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

            # Check if user is qualified for this segment type
            if user.qualified_segment_types is not None and len(user.qualified_segment_types) == 0:
                raise UserValidationError("User not qualified for this segment type")
            if user.qualified_segment_types and task_data.get("extra_data"):
                extra_data = task_data["extra_data"]
                if "seed_id" in extra_data:
                    seed_id = extra_data["seed_id"]
                    # Get segment from database to check expected_segment_type
                    segment_query = (
                        select(SegmentModel)
                        .where(SegmentModel.project_name == project_name)
                        .where(SegmentModel.seed_id == seed_id)
                    )
                    segment = session.execute(segment_query).scalar_one_or_none()
                    if segment and segment.expected_segment_type:
                        if segment.expected_segment_type not in user.qualified_segment_types:
                            raise UserValidationError("User not qualified for this segment type")

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

    qualified_segment_types = user.qualified_segment_types or []

    logger.info(f"User {user_id} is qualified for: {qualified_types}")
    logger.info(f"User {user_id} is qualified for segment types: {qualified_segment_types}")

    # Strategy 1: Look for tasks explicitly assigned to this user
    logger.info(f"Looking for assigned tasks for user {user_id}")
    query = (
        select(TaskModel)
        .outerjoin(
            SegmentModel,
            and_(
                func.cast(
                    TaskModel.extra_data.op("->>")("seed_id"),
                    BigInteger,
                )
                == SegmentModel.seed_id,
                SegmentModel.project_name == project_name,
            ),
        )
        .where(TaskModel.project_name == project_name)
        .where(TaskModel.is_active == True)
        .where(TaskModel.is_paused == False)  # Exclude paused tasks
        .where(TaskModel.completion_status == "")
        .where(TaskModel.task_type.in_(qualified_types))
        .where(TaskModel.assigned_user_id == user_id)
        .where(TaskModel.active_user_id == "")  # Not currently active
        .where(
            or_(
                # Alignment tasks (no seed_id) - skip segment filtering
                TaskModel.extra_data.op("->>")("seed_id").is_(None),
                # Segment tasks - require segment type match
                SegmentModel.expected_segment_type.in_(qualified_segment_types),
            )
        )
        .order_by(TaskModel.priority.desc(), TaskModel.id_nonunique)
        .limit(1)
        .with_for_update(skip_locked=True, of=TaskModel)
    )
    result = db_session.execute(query).scalar_one_or_none()
    if result:
        logger.info(f"Found assigned task {result.task_id} for user {user_id}")
        return result

    # Strategy 2: Look for any available tasks (prioritized by priority)
    logger.info("Looking for available tasks")
    query = (
        select(TaskModel)
        .outerjoin(
            SegmentModel,
            and_(
                func.cast(
                    TaskModel.extra_data.op("->>")("seed_id"),
                    BigInteger,
                )
                == SegmentModel.seed_id,
                SegmentModel.project_name == project_name,
            ),
        )
        .where(TaskModel.project_name == project_name)
        .where(TaskModel.is_active == True)
        .where(TaskModel.is_paused == False)  # Exclude paused tasks
        .where(TaskModel.completion_status == "")
        .where(TaskModel.task_type.in_(qualified_types))
        .where(TaskModel.active_user_id == "")  # Not currently active
        .where(
            or_(
                # Alignment tasks (no seed_id) - skip segment filtering
                TaskModel.extra_data.op("->>")("seed_id").is_(None),
                # Segment tasks - require segment type match
                SegmentModel.expected_segment_type.in_(qualified_segment_types),
            )
        )
        .order_by(TaskModel.priority.desc(), TaskModel.id_nonunique)
        .limit(1)
        .with_for_update(skip_locked=True, of=TaskModel)
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
        .outerjoin(
            SegmentModel,
            and_(
                func.cast(
                    TaskModel.extra_data.op("->>")("seed_id"),
                    BigInteger,
                )
                == SegmentModel.seed_id,
                SegmentModel.project_name == project_name,
            ),
        )
        .where(TaskModel.project_name == project_name)
        .where(TaskModel.is_active == True)
        .where(TaskModel.is_paused == False)  # Exclude paused tasks
        .where(TaskModel.completion_status == "")
        .where(TaskModel.task_type.in_(qualified_types))
        .where(TaskModel.last_leased_ts < oldest_allowed_ts)
        .where(
            or_(
                # Alignment tasks (no seed_id) - skip segment filtering
                TaskModel.extra_data.op("->>")("seed_id").is_(None),
                # Segment tasks - require segment type match
                SegmentModel.expected_segment_type.in_(qualified_segment_types),
            )
        )
        .order_by(TaskModel.priority.desc(), TaskModel.last_leased_ts.desc())
        .limit(1)
        .with_for_update(skip_locked=True, of=TaskModel)  # Skip if locked
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


def _select_assigned_task_across_projects(
    db_session: Session, user_qualifications: dict[str, _UserProjectQualifications]
) -> TaskModel | None:
    """Find a task explicitly assigned to the user across accessible projects."""
    logger.info("Looking for assigned tasks across all projects")
    for project_name, qualifications in user_qualifications.items():
        qualified_types = qualifications.get("task_types", [])
        qualified_segment_types = qualifications.get("segment_types", [])
        if not qualified_types:
            continue
        query = (
            select(TaskModel)
            .outerjoin(
                SegmentModel,
                and_(
                    func.cast(TaskModel.extra_data.op("->>")("seed_id"), BigInteger)
                    == SegmentModel.seed_id,
                    SegmentModel.project_name == project_name,
                ),
            )
            .where(TaskModel.project_name == project_name)
            .where(TaskModel.is_active.is_(True))
            .where(TaskModel.is_paused == False)
            .where(TaskModel.completion_status == "")
            .where(TaskModel.task_type.in_(qualified_types))
            .where(TaskModel.assigned_user_id == qualifications["user_id"])
            .where(TaskModel.active_user_id == "")
            .where(
                or_(
                    # Alignment tasks (no seed_id) - skip segment filtering
                    TaskModel.extra_data.op("->>")("seed_id").is_(None),
                    # Segment tasks - require segment type match
                    SegmentModel.expected_segment_type.in_(qualified_segment_types),
                )
            )
            .order_by(TaskModel.priority.desc(), TaskModel.id_nonunique)
            .limit(1)
            .with_for_update(skip_locked=True, of=TaskModel)
        )
        result = db_session.execute(query).scalar_one_or_none()
        if result:
            logger.info(
                f"Found assigned task {result.task_id} in project {project_name}"
            )
            return result
    return None


def _select_available_task_across_projects(
    db_session: Session, user_qualifications: dict[str, _UserProjectQualifications]
) -> TaskModel | None:
    """Find the highest-priority available task across projects, then lock it."""
    logger.info("Looking for available tasks across all projects")
    best_task: TaskModel | None = None
    best_priority: int | None = None
    for project_name, qualifications in user_qualifications.items():
        qualified_types = qualifications.get("task_types", [])
        qualified_segment_types = qualifications.get("segment_types", [])
        if not qualified_types:
            continue
        project_query = (
            select(TaskModel)
            .outerjoin(
                SegmentModel,
                and_(
                    func.cast(TaskModel.extra_data.op("->>")("seed_id"), BigInteger)
                    == SegmentModel.seed_id,
                    SegmentModel.project_name == project_name,
                ),
            )
            .where(TaskModel.project_name == project_name)
            .where(TaskModel.is_active.is_(True))
            .where(TaskModel.is_paused == False)
            .where(TaskModel.completion_status == "")
            .where(TaskModel.assigned_user_id == "")
            .where(TaskModel.task_type.in_(qualified_types))
            .where(TaskModel.active_user_id == "")
            .where(
                or_(
                    # Alignment tasks (no seed_id) - skip segment filtering
                    TaskModel.extra_data.op("->>")("seed_id").is_(None),
                    # Segment tasks - require segment type match
                    SegmentModel.expected_segment_type.in_(qualified_segment_types),
                )
            )
            .order_by(TaskModel.priority.desc(), TaskModel.id_nonunique)
            .limit(1)
        )
        result = db_session.execute(project_query).scalar_one_or_none()
        if result:
            if (
                best_task is None
                or result.priority > cast(int, best_priority)
                or (
                    result.priority == best_priority
                    and result.id_nonunique < best_task.id_nonunique
                )
            ):
                best_task = result
                best_priority = result.priority
    if best_task is None:
        return None
    # Lock and re-check availability
    lock_query = (
        select(TaskModel)
        .where(TaskModel.task_id == best_task.task_id)
        .where(TaskModel.project_name == best_task.project_name)
        .with_for_update(skip_locked=True)
    )
    locked = db_session.execute(lock_query).scalar_one_or_none()
    if (
        locked
        and locked.active_user_id == ""
        and locked.completion_status == ""
        and locked.assigned_user_id == ""
    ):
        logger.info(
            f"Found available task {locked.task_id} of type {locked.task_type} in project {locked.project_name}"  # pylint: disable=C0301
        )
        return locked
    return None


def _select_idle_task_across_projects(
    db_session: Session, user_qualifications: dict[str, _UserProjectQualifications]
) -> TaskModel | None:
    """Find the best idle task across projects, then lock and verify idleness."""
    oldest_allowed_ts = time.time() - get_max_idle_seconds()
    logger.info("Looking for idle tasks across all projects")
    best_idle: TaskModel | None = None
    best_priority: int | None = None
    best_lease: float | None = None
    for project_name, qualifications in user_qualifications.items():
        qualified_types = qualifications.get("task_types", [])
        qualified_segment_types = qualifications.get("segment_types", [])
        if not qualified_types:
            continue
        project_query = (
            select(TaskModel)
            .outerjoin(
                SegmentModel,
                and_(
                    func.cast(TaskModel.extra_data.op("->>")("seed_id"), BigInteger)
                    == SegmentModel.seed_id,
                    SegmentModel.project_name == project_name,
                ),
            )
            .where(TaskModel.project_name == project_name)
            .where(TaskModel.is_active.is_(True))
            .where(TaskModel.is_paused == False)
            .where(TaskModel.completion_status == "")
            .where(TaskModel.assigned_user_id == "")
            .where(TaskModel.task_type.in_(qualified_types))
            .where(TaskModel.last_leased_ts < oldest_allowed_ts)
            .where(
                or_(
                    # Alignment tasks (no seed_id) - skip segment filtering
                    TaskModel.extra_data.op("->>")("seed_id").is_(None),
                    # Segment tasks - require segment type match
                    SegmentModel.expected_segment_type.in_(qualified_segment_types),
                )
            )
            .order_by(TaskModel.priority.desc(), TaskModel.last_leased_ts.desc())
            .limit(1)
        )
        result = db_session.execute(project_query).scalar_one_or_none()
        if result:
            if (
                best_idle is None
                or result.priority > cast(int, best_priority)
                or (
                    result.priority == best_priority
                    and result.last_leased_ts > cast(float, best_lease)
                )
            ):
                best_idle = result
                best_priority = result.priority
                best_lease = result.last_leased_ts
    if best_idle is None:
        return None
    # Lock and verify still idle
    lock_query = (
        select(TaskModel)
        .where(TaskModel.task_id == best_idle.task_id)
        .where(TaskModel.project_name == best_idle.project_name)
        .with_for_update(skip_locked=True)
    )
    locked = db_session.execute(lock_query).scalar_one_or_none()
    if locked:
        db_session.refresh(locked)
        if (
            locked.last_leased_ts < oldest_allowed_ts
            and locked.completion_status == ""
            and not locked.is_paused
            and locked.assigned_user_id == ""
        ):
            logger.info(
                f"Found idle task {locked.task_id} in project {locked.project_name}"
            )
            return locked
    return None


def _auto_select_task_cross_project(
    db_session: Session, user_qualifications: dict[str, _UserProjectQualifications]
) -> TaskModel | None:
    """
    Auto-select an appropriate task across projects using three strategies:
    1) user-assigned, 2) available by priority, 3) idle recovery.
    """
    logger.info("Auto-selecting task across all projects")
    if not user_qualifications:
        logger.info("User has no project qualifications")
        return None
    # 1) Assigned to user
    result = _select_assigned_task_across_projects(db_session, user_qualifications)
    if result:
        return result
    # 2) Available by priority
    result = _select_available_task_across_projects(db_session, user_qualifications)
    if result:
        return result
    # 3) Idle task recovery
    return _select_idle_task_across_projects(db_session, user_qualifications)


# Backward-compatible wrapper; preferred name is _get_active_user_with_qualifications
def _build_user_qualifications_map(
    db_session: Session, user_id: str
) -> tuple[dict[str, _UserProjectQualifications], str | None, str | None]:
    return _get_active_user_with_qualifications(db_session, user_id)


def _get_active_user_with_qualifications(
    db_session: Session, user_id: str
) -> tuple[dict[str, _UserProjectQualifications], str | None, str | None]:
    """
    Build a map of project -> qualifications and detect current active project/task.
    Returns (qualifications_map, active_project, active_task_id).
    """
    users_query = select(UserModel).where(UserModel.user_id == user_id)
    users = db_session.execute(users_query).scalars().all()
    if not users:
        raise UserValidationError(f"User {user_id} not found in any project")
    user_qualifications: dict[str, _UserProjectQualifications] = {}
    current_active_project: str | None = None
    current_active_task_id: str | None = None
    for user in users:
        if user.active_task and user.active_task != "":
            if current_active_project is not None:
                raise UserValidationError(
                    "User has active tasks in multiple projects: "
                    f"{current_active_project} and {user.project_name}"
                )
            current_active_project = str(user.project_name)
            current_active_task_id = user.active_task
        user_qualifications[user.project_name] = {
            "user_id": user.user_id,
            "task_types": user.qualified_task_types or [],
            "segment_types": user.qualified_segment_types or [],
        }
    return user_qualifications, current_active_project, current_active_task_id


def _find_task_across_projects(
    db_session: Session, task_id: str, projects: list[str]
) -> TaskModel | None:
    for project_name in projects:
        task_query = (
            select(TaskModel)
            .where(TaskModel.task_id == task_id)
            .where(TaskModel.project_name == project_name)
        )
        found = db_session.execute(task_query).scalar_one_or_none()
        if found:
            return found
    return None


def _validate_user_for_task_in_project(
    db_session: Session, user_id: str, project_name: str, task_model: TaskModel
) -> None:
    """Validate that user is qualified for task type and segment type in a project."""
    project_user_query = (
        select(UserModel)
        .where(UserModel.user_id == user_id)
        .where(UserModel.project_name == project_name)
    )
    project_user = db_session.execute(project_user_query).scalar_one()
    task_data = task_model.to_dict()
    if (
        not project_user.qualified_task_types
        or task_data["task_type"] not in project_user.qualified_task_types
    ):
        raise UserValidationError("User not qualified for this task type")
    if project_user.qualified_segment_types and task_data.get("extra_data"):
        extra_data = task_data["extra_data"]
        if "seed_id" in extra_data:
            seed_id = extra_data["seed_id"]
            segment_query = (
                select(SegmentModel)
                .where(SegmentModel.project_name == project_name)
                .where(SegmentModel.seed_id == seed_id)
            )
            segment = db_session.execute(segment_query).scalar_one_or_none()
            if segment and segment.expected_segment_type:
                if segment.expected_segment_type not in project_user.qualified_segment_types:
                    raise UserValidationError("User not qualified for this segment type")


def _lock_and_assign_cross_project(
    db_session: Session,
    selected_task: TaskModel,
    user_id: str,
    user_qualifications: dict[str, _UserProjectQualifications],
) -> None:
    """Lock needed rows, clear other active tasks, and assign this task to the user."""
    project_name = str(selected_task.project_name)
    current_time = time.time()
    # Lock user in target project
    locked_user_query = (
        select(UserModel)
        .where(UserModel.user_id == user_id)
        .where(UserModel.project_name == project_name)
        .with_for_update()
    )
    locked_user = db_session.execute(locked_user_query).scalar_one()
    # Clear active tasks from other projects
    for other_project in user_qualifications.keys():
        if other_project == project_name:
            continue
        other_user_query = (
            select(UserModel)
            .where(UserModel.user_id == user_id)
            .where(UserModel.project_name == other_project)
            .with_for_update()
        )
        other_user = db_session.execute(other_user_query).scalar_one()
        other_user.active_task = ""
    # Lock the task
    locked_task_query = (
        select(TaskModel)
        .where(TaskModel.task_id == selected_task.task_id)
        .where(TaskModel.project_name == project_name)
        .with_for_update()
    )
    locked_task = db_session.execute(locked_task_query).scalar_one()
    # Assign
    locked_user.active_task = locked_task.task_id
    locked_task.active_user_id = user_id
    locked_task.last_leased_ts = current_time
    if locked_task.first_start_ts is None:
        locked_task.first_start_ts = current_time


def start_task_cross_project(
    *,
    user_id: str,
    task_id: str | None = None,
    db_session: Session | None = None,
) -> str | None:
    """
    Start a task for a user across all projects they have access to.
    Tasks are selected by highest priority across all projects.
    
    :param user_id: The user requesting a task
    :param task_id: Optional specific task ID to start
    :param db_session: Database session to use (optional)
    :return: The ID of the started task, or None if no task available
    :raises UserValidationError: If user validation fails
    :raises TaskValidationError: If task validation fails
    """
    with get_session_context(db_session) as session:
        # Build qualifications and detect any current active task
        (
            user_qualifications,
            current_active_project,
            current_active_task_id,
        ) = _build_user_qualifications_map(session, user_id)
        # If user has an active task and specific task_id requested, validate they match
        if task_id is not None and current_active_task_id is not None:
            if current_active_task_id != task_id:
                raise UserValidationError(
                    f"User already has an active task {current_active_task_id} "
                    f"which is different from requested task {task_id}"
                )
        # Select task based on whether task_id is specified
        if task_id is None and current_active_task_id is None:
            selected_task = _auto_select_task_cross_project(session, user_qualifications)
        elif task_id is not None:
            selected_task = _find_task_across_projects(
                session, task_id, list(user_qualifications.keys())
            )
            if not selected_task:
                raise TaskValidationError(
                    f"Task {task_id} not found in any accessible project"
                )
        else:
            # Return to current active task
            task_query = (
                select(TaskModel)
                .where(TaskModel.task_id == current_active_task_id)
                .where(TaskModel.project_name == current_active_project)
            )
            selected_task = session.execute(task_query).scalar_one()

        if selected_task is not None:
            logger.info(f"Selected task type: {type(selected_task)}, value: {selected_task}")
            project_name = str(selected_task.project_name)
            task_data = selected_task.to_dict()
            _validate_task(session, project_name, task_data)
            # Validate user for task type and segment type in that project
            _validate_user_for_task_in_project(session, user_id, project_name, selected_task)
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
                _lock_and_assign_cross_project(
                    session, selected_task, user_id, user_qualifications
                )
            session.flush()  # Ensure changes are written to DB before commit
            session.commit()
            return str(selected_task.task_id)

        return None


def get_task_cross_project(
    *,
    task_id: str,
    process_ng_state: bool = True,
    db_session: Session | None = None,
) -> tuple[Task, str]:
    """
    Retrieve a task record by ID across all projects.
    
    :param task_id: The unique identifier of the task
    :param process_ng_state: Whether to process ng_state seed_id format (default: True)
    :param db_session: Database session to use (optional)
    :return: Tuple of (task record, project_name)
    :raises KeyError: If the task does not exist in any project
    :raises RuntimeError: If the database operation fails
    """
    with get_session_context(db_session) as session:
        # Find the task across all projects
        cross_project_query = (
            select(TaskModel)
            .where(TaskModel.task_id == task_id)
        )
        try:
            task_model = session.execute(cross_project_query).scalar_one()
            project_name = str(task_model.project_name)

            # Handle ng_state and ng_state_initial special formats
            if process_ng_state:
                _process_ng_state_seed_id(session, project_name, task_model)

            result = task_model.to_dict()
            return cast(Task, result), project_name
        except NoResultFound as exc:
            raise KeyError(f"Task {task_id} not found in any project") from exc


def get_task_feedback_cross_project(
    *,
    user_id: str,
    limit: int = 20,
    skip: int = 0,
    db_session: Session | None = None,
) -> list[dict]:
    """
    Get task feedback entries for a user across all projects they have access to.
    
    :param user_id: The ID of the user to get feedback for
    :param limit: Maximum number of feedback entries to return (default: 20)
    :param db_session: Database session to use (optional)
    :param skip: Number of records to skip for pagination (default: 0)
    :return: List of feedback entries with task and feedback data across all projects
    :raises UserValidationError: If the user is not found in any project
    :raises RuntimeError: If the database operation fails
    """
    with get_session_context(db_session) as session:
        # Build user qualifications to find all accessible projects
        user_qualifications, _, _ = _build_user_qualifications_map(session, user_id)

        if not user_qualifications:
            raise UserValidationError(f"User {user_id} not found in any project")

        project_names = list(user_qualifications.keys())

        # Query feedback entries across all accessible projects
        feedback_query = (
            select(TaskFeedbackModel)
            .where(TaskFeedbackModel.project_name.in_(project_names))
            .where(TaskFeedbackModel.user_id == user_id)
            .order_by(TaskFeedbackModel.created_at.desc())
            .limit(limit)
            .offset(skip)
        )

        feedbacks = session.execute(feedback_query).scalars().all()

        feedback_data = []
        for feedback in feedbacks:
            # Get original task data
            original_task = (
                session.query(TaskModel)
                .filter(
                    TaskModel.project_name == feedback.project_name,
                    TaskModel.task_id == feedback.task_id,
                )
                .first()
            )

            # Get feedback task data
            feedback_task = (
                session.query(TaskModel)
                .filter(
                    TaskModel.project_name == feedback.project_name,
                    TaskModel.task_id == feedback.feedback_task_id,
                )
                .first()
            )

            # Map completion status to feedback type
            feedback_type = feedback_task.completion_status if feedback_task else "Unknown"
            feedback_color = "red"  # Default to red for unknown statuses

            if feedback_type == "Accurate":
                feedback_color = "green"
            elif feedback_type == "Fair":
                feedback_color = "yellow"
            elif feedback_type == "Inaccurate":
                feedback_color = "red"

            feedback_data.append(
                {
                    "project_name": feedback.project_name,  # Include project name
                    "task_id": feedback.task_id,
                    "task_link": original_task.ng_state if original_task else None,
                    "feedback_link": feedback_task.ng_state if feedback_task else None,
                    "feedback": feedback_type,
                    "feedback_color": feedback_color,
                    "note": feedback_task.note if feedback_task else None,
                    "created_at": feedback.created_at.isoformat() if feedback.created_at else None,
                    "user_id": feedback.user_id,
                    "feedback_id": feedback.feedback_id,
                    "feedback_task_id": feedback.feedback_task_id,
                }
            )

        return feedback_data


def get_task(
    *,
    project_name: str,
    task_id: str,
    process_ng_state: bool = True,
    db_session: Session | None = None,
) -> Task:
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
                    db_session=session,
                )
                # Update the database
                task.ng_state = generated_ng_state
                task.ng_state_initial = generated_ng_state
                session.commit()
                logger.info(
                    f"Successfully generated and saved ng_state for seed_id {seed_id} "
                    f"in task {task.task_id}"
                )
            except RuntimeError as e:
                logger.error(
                    f"Failed to generate ng_state for seed_id {seed_id} "
                    f"in task {task.task_id}: {e}"
                )
                session.rollback()


@typechecked
def add_segment_type_and_instructions(task_dict: dict, project_name: str, db_session: Session | None = None) -> dict: # pylint: disable=line-too-long
    """
    Add segment type and instructions to task dictionary if task has a segment_seed_id.

    :param task_dict: The task dictionary to enhance
    :param project_name: The project name
    :param db_session: Optional database session to use
    :return: The enhanced task dictionary with segment type and instructions
    """
    if not task_dict.get("segment_seed_id"):
        return task_dict

    seed_id = task_dict["segment_seed_id"]
    with get_session_context(db_session) as session:
        segment_query = select(SegmentModel).where(
            SegmentModel.project_name == project_name, SegmentModel.seed_id == seed_id
        )
        segment = session.execute(segment_query).scalar_one_or_none()
        if segment:
            task_dict["segment_type"] = segment.expected_segment_type
            # Fetch instructions from segment type
            if segment.expected_segment_type:
                segment_type_query = select(SegmentTypeModel).where(
                    SegmentTypeModel.project_name == project_name,
                    SegmentTypeModel.type_name == segment.expected_segment_type,
                )
                segment_type = session.execute(segment_type_query).scalar_one_or_none()
                if segment_type:
                    task_dict["instruction"] = segment_type.instruction
                    task_dict["instruction_link"] = segment_type.instruction_link

    return task_dict
