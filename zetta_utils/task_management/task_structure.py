import copy
import json
from typing import Any, Callable, Mapping

from coolname import generate_slug
from sqlalchemy import select
from sqlalchemy.orm import Session

from zetta_utils.log import get_logger
from zetta_utils.task_management.db.models import DependencyModel, JobModel, TaskModel
from zetta_utils.task_management.db.session import get_session_context
from zetta_utils.task_management.utils import generate_id_nonunique

logger = get_logger(__name__)

# Registry to store all registered task structures
_TASK_STRUCTURES: dict[str, Callable] = {}


def register_task_structure(name: str):
    """
    Decorator to register a task structure implementation.

    :param name: The name of the task structure
    :return: Decorator function
    """

    def decorator(func: Callable):
        _TASK_STRUCTURES[name] = func
        return func

    return decorator


def get_available_structures() -> list[str]:  # pragma: no cover
    """
    Get a list of all available task structure names.

    :return: List of structure names
    """
    return list(_TASK_STRUCTURES.keys())


def create_task_structure(
    *,
    project_name: str,
    job_id: str,
    task_structure: str,
    task_structure_kwargs: Mapping[str, Any],
    priority: int = 1,
    db_session: Session | None = None,
) -> bool:
    """
    Create a predefined task structure for a job.

    :param project_name: The name of the project
    :param job_id: The ID of the job
    :param task_structure: The name of the task structure to create
    :param task_structure_kwargs: Keyword arguments for the task structure
    :param priority: The priority of the task
    :param db_session: Database session to use (optional)
    :return: True if successful
    :raises ValueError: If the task structure is not registered
    :raises KeyError: If the job does not exist
    :raises RuntimeError: If the database operation fails
    """
    with get_session_context(db_session) as session:
        logger.info(f"Creating atomic task structure '{task_structure}' for job {job_id}")
        suffix = generate_slug(4)

        # ATOMIC BULK OPERATION: Lock the job and create all tasks/dependencies atomically
        try:
            # Lock the job to prevent concurrent modifications
            job_lock_query = (
                select(JobModel)
                .where(JobModel.project_name == project_name)
                .where(JobModel.job_id == job_id)
                .with_for_update()
            )
            locked_job = session.execute(job_lock_query).scalar_one()
            job_data = locked_job.to_dict()
            ng_state = job_data["ng_state"]

            if task_structure not in _TASK_STRUCTURES:
                raise ValueError(f"Task structure '{task_structure}' is not registered")

            structure_func = _TASK_STRUCTURES[task_structure]

            # Call the structure function to create all tasks and dependencies
            # This happens within the same transaction as the job lock
            # Any errors here (like missing task types) will cause rollback
            structure_func(
                db_session=session,
                project_name=project_name,
                job_id=job_id,
                batch_id=job_data["batch_id"],
                ng_state=ng_state,
                priority=priority,
                suffix=suffix,
                **task_structure_kwargs,
            )

            # Update job status to ingested atomically
            locked_job.status = "ingested"

            # Commit all changes atomically - if this fails, everything rolls back
            session.commit()
            logger.info(
                f"Successfully created task structure '{task_structure}' for job {job_id}"
            )

            return True

        except Exception as e:
            session.rollback()
            logger.error(
                f"Failed to create task structure '{task_structure}' for job {job_id}: {e}"
            )
            raise RuntimeError(f"Failed to create task structure: {e}") from e


@register_task_structure("segmentation_proofread_1pass")
def segmentation_proofread_simple(
    db_session: Session,
    project_name: str,
    job_id: str,
    batch_id: str,
    ng_state: dict,
    priority: int,
    suffix: str,
):
    """
    Create a simple segmentation proofread structure with three tasks:
    1. segmentation_proofread (active)
    2. segmentation_verify (inactive, depends on proofread → done)
    3. segmentation_proofread_expert (inactive, depends on proofread → need_help)

    :param db_session: Database session to use
    :param project_name: Project name
    :param job_id: Job ID
    :param batch_id: Batch ID
    :param ng_state: The ng_state for the task
    :param priority: The priority of the task
    :param suffix: The suffix for the task
    """
    id_suffix = f"_{suffix}"

    # Create base IDs for tasks and dependencies
    proofread_id = f"{job_id}_proofread{id_suffix}"
    verify_id = f"{job_id}_verify{id_suffix}"
    expert_id = f"{job_id}_proofread_expert{id_suffix}"
    dep_verify_id = f"{job_id}_dep_verify{id_suffix}"
    dep_expert_id = f"{job_id}_dep_expert{id_suffix}"

    # 1. Create segmentation_proofread task (active)
    print("Creating segmentation_proofread task")
    proofread_task = TaskModel(
        project_name=project_name,
        task_id=proofread_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state,
        ng_state_initial=ng_state,
        priority=priority,
        batch_id=batch_id,
        task_type="segmentation_proofread",
        is_active=True,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(proofread_task)

    # 2. Create segmentation_verify task (inactive)
    print("Creating segmentation_verify task")
    verify_task = TaskModel(
        project_name=project_name,
        task_id=verify_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state,
        ng_state_initial=ng_state,
        priority=priority,
        batch_id=batch_id,
        task_type="segmentation_verify",
        is_active=False,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(verify_task)

    # 3. Create segmentation_proofread_expert task (inactive)
    print("Creating segmentation_proofread_expert task")
    expert_task = TaskModel(
        project_name=project_name,
        task_id=expert_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state,
        ng_state_initial=ng_state,
        priority=priority,
        batch_id=batch_id,
        task_type="segmentation_proofread_expert",
        is_active=False,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(expert_task)

    # 4. Create dependency: verify depends on proofread being "done"
    print("Creating dependency for segmentation_verify")
    dep_verify = DependencyModel(
        project_name=project_name,
        dependency_id=dep_verify_id,
        task_id=verify_id,
        dependent_on_task_id=proofread_id,
        required_completion_status="done",
        is_satisfied=False,
    )
    db_session.add(dep_verify)

    # 5. Create dependency: expert depends on proofread being "need_help"
    print("Creating dependency for segmentation_proofread_expert")
    dep_expert = DependencyModel(
        project_name=project_name,
        dependency_id=dep_expert_id,
        task_id=expert_id,
        dependent_on_task_id=proofread_id,
        required_completion_status="need_help",
        is_satisfied=False,
    )
    db_session.add(dep_expert)


@register_task_structure("segmentation_proofread_simple_1pass")
def segmentation_proofread_simple_1pass(
    db_session: Session,
    project_name: str,
    job_id: str,
    batch_id: str,
    ng_state: dict,
    priority: int,
    suffix: str,
):
    """
    Create a simple 1-pass segmentation proofread structure with one task.

    :param db_session: Database session to use
    :param project_name: Project name
    :param job_id: Job ID
    :param batch_id: Batch ID
    :param ng_state: The ng_state for the task
    :param priority: The priority of the task
    :param suffix: The suffix for the task
    """
    id_suffix = f"_{suffix}"

    # Create base IDs for tasks
    proofread_id = f"{job_id}_proofread{id_suffix}"

    # Create segmentation_proofread task (active)
    print("Creating segmentation_proofread task")
    proofread_task = TaskModel(
        project_name=project_name,
        task_id=proofread_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state,
        ng_state_initial=ng_state,
        priority=priority,
        batch_id=batch_id,
        task_type="segmentation_proofread",
        is_active=True,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(proofread_task)


@register_task_structure("segmentation_proofread_two_path")
def segmentation_proofread_two_path(
    db_session: Session,
    project_name: str,
    job_id: str,
    batch_id: str,
    ng_state: dict,
    priority: int,
    suffix: str,
    validation_layer_path: str,
):
    """
    Create a two-path segmentation proofread structure with validation layer.

    :param db_session: Database session to use
    :param project_name: Project name
    :param job_id: Job ID
    :param batch_id: Batch ID
    :param ng_state: The ng_state for the task
    :param priority: The priority of the task
    :param suffix: The suffix for the task
    :param validation_layer_path: Path to the validation layer
    """
    id_suffix = f"_{suffix}"

    # Modify ng_state to include validation layer
    # Create validation version of ng_state
    ng_state_validation = copy.deepcopy(ng_state)
    if "layers" in ng_state_validation:
        # Add validation layer
        validation_layer = {
            "source": validation_layer_path,
            "type": "segmentation",
            "name": "validation",
        }
        ng_state_validation["layers"].append(validation_layer)

    # ng_state_validation is already a dict, no need to convert to string

    # Create base IDs for tasks and dependencies
    proofread_id = f"{job_id}_proofread{id_suffix}"
    verify_id = f"{job_id}_verify{id_suffix}"
    expert_id = f"{job_id}_proofread_expert{id_suffix}"
    dep_verify_id = f"{job_id}_dep_verify{id_suffix}"
    dep_expert_id = f"{job_id}_dep_expert{id_suffix}"

    # 1. Create segmentation_proofread task (active)
    print("Creating segmentation_proofread task")
    proofread_task = TaskModel(
        project_name=project_name,
        task_id=proofread_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state,
        ng_state_initial=ng_state,
        priority=priority,
        batch_id=batch_id,
        task_type="segmentation_proofread",
        is_active=True,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(proofread_task)

    # 2. Create segmentation_verify task with validation layer (inactive)
    print("Creating segmentation_verify task with validation layer")
    verify_task = TaskModel(
        project_name=project_name,
        task_id=verify_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state_validation,
        ng_state_initial=ng_state_validation,
        priority=priority,
        batch_id=batch_id,
        task_type="segmentation_verify",
        is_active=False,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(verify_task)

    # 3. Create segmentation_proofread_expert task (inactive)
    print("Creating segmentation_proofread_expert task")
    expert_task = TaskModel(
        project_name=project_name,
        task_id=expert_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state,
        ng_state_initial=ng_state,
        priority=priority,
        batch_id=batch_id,
        task_type="segmentation_proofread_expert",
        is_active=False,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(expert_task)

    # 4. Create dependency: verify depends on proofread being "done"
    print("Creating dependency for segmentation_verify")
    dep_verify = DependencyModel(
        project_name=project_name,
        dependency_id=dep_verify_id,
        task_id=verify_id,
        dependent_on_task_id=proofread_id,
        required_completion_status="done",
        is_satisfied=False,
    )
    db_session.add(dep_verify)

    # 5. Create dependency: expert depends on proofread being "need_help"
    print("Creating dependency for segmentation_proofread_expert")
    dep_expert = DependencyModel(
        project_name=project_name,
        dependency_id=dep_expert_id,
        task_id=expert_id,
        dependent_on_task_id=proofread_id,
        required_completion_status="need_help",
        is_satisfied=False,
    )
    db_session.add(dep_expert)


@register_task_structure("seg_v0_auto_verify")
def seg_v0_auto_verify(
    db_session: Session,
    project_name: str,
    job_id: str,
    batch_id: str,
    ng_state: dict,
    priority: int,
    suffix: str,
):
    """
    Create a seg_v0_auto_verify structure with three tasks:
    1. seg_trace (active) - if done → unlocks seg_auto_verify, if out_of_scope → nothing
    2. seg_auto_verify (inactive) - if fail → unlocks seg_trace_expert, if pass → complete
    3. seg_trace_expert (inactive) - depends on auto_verify → fail

    :param db_session: Database session to use
    :param project_name: Project name
    :param job_id: Job ID
    :param batch_id: Batch ID
    :param ng_state: The ng_state for the task
    :param priority: The priority of the task
    :param suffix: The suffix for the task
    """
    id_suffix = f"_{suffix}"

    # Create base IDs for tasks and dependencies
    trace_id = f"{job_id}_trace{id_suffix}"
    auto_verify_id = f"{job_id}_auto_verify{id_suffix}"
    expert_id = f"{job_id}_trace_expert{id_suffix}"
    dep_auto_verify_id = f"{job_id}_dep_auto_verify{id_suffix}"
    dep_expert_id = f"{job_id}_dep_expert{id_suffix}"

    # 1. Create seg_trace task (active)
    print (ng_state)
    print (type(ng_state))
    print("Creating seg_trace task")
    trace_task = TaskModel(
        project_name=project_name,
        task_id=trace_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state,
        ng_state_initial=ng_state,
        priority=priority,
        batch_id=batch_id,
        task_type="seg_trace",
        is_active=True,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(trace_task)

    # 2. Create seg_auto_verify task (inactive)
    print("Creating seg_auto_verify task")
    auto_verify_task = TaskModel(
        project_name=project_name,
        task_id=auto_verify_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state,
        ng_state_initial=ng_state,
        priority=priority,
        batch_id=batch_id,
        task_type="seg_auto_verify",
        is_active=False,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
        extra_data={"trace_task_id": trace_id},
    )
    db_session.add(auto_verify_task)

    # 3. Create seg_trace_expert task (inactive)
    print("Creating seg_trace_expert task")
    expert_task = TaskModel(
        project_name=project_name,
        task_id=expert_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state,
        ng_state_initial=ng_state,
        priority=priority,
        batch_id=batch_id,
        task_type="seg_trace_expert",
        is_active=False,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(expert_task)

    # 4. Create dependency: auto_verify depends on trace being "done"
    print("Creating dependency for seg_auto_verify")
    dep_auto_verify = DependencyModel(
        project_name=project_name,
        dependency_id=dep_auto_verify_id,
        task_id=auto_verify_id,
        dependent_on_task_id=trace_id,
        required_completion_status="done",
        is_satisfied=False,
    )
    db_session.add(dep_auto_verify)

    # 5. Create dependency: expert depends on auto_verify being "fail"
    print("Creating dependency for seg_trace_expert")
    dep_expert = DependencyModel(
        project_name=project_name,
        dependency_id=dep_expert_id,
        task_id=expert_id,
        dependent_on_task_id=auto_verify_id,
        required_completion_status="fail",
        is_satisfied=False,
    )
    db_session.add(dep_expert)
