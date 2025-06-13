import copy
import json
from typing import Any, Callable, Mapping

from coolname import generate_slug
from sqlalchemy import select
from sqlalchemy.orm import Session

from zetta_utils.log import get_logger
from zetta_utils.task_management.db.models import (
    DependencyModel,
    JobModel,
    SubtaskModel,
)
from zetta_utils.task_management.db.session import get_session_context
from zetta_utils.task_management.utils import generate_id_nonunique

logger = get_logger(__name__)

# Registry to store all registered subtask structures
_SUBTASK_STRUCTURES: dict[str, Callable] = {}


def register_subtask_structure(name: str):
    """
    Decorator to register a subtask structure implementation.

    :param name: The name of the subtask structure
    :return: Decorator function
    """

    def decorator(func: Callable):
        _SUBTASK_STRUCTURES[name] = func
        return func

    return decorator


def get_available_structures() -> list[str]:  # pragma: no cover
    """
    Get a list of all available subtask structure names.

    :return: List of structure names
    """
    return list(_SUBTASK_STRUCTURES.keys())


def create_subtask_structure(
    *,
    project_name: str,
    job_id: str,
    subtask_structure: str,
    subtask_structure_kwargs: Mapping[str, Any],
    priority: int = 1,
    db_session: Session | None = None,
) -> bool:
    """
    Create a predefined subtask structure for a job.

    :param project_name: The name of the project
    :param job_id: The ID of the job
    :param subtask_structure: The name of the subtask structure to create
    :param subtask_structure_kwargs: Keyword arguments for the subtask structure
    :param priority: The priority of the subtask
    :param db_session: Database session to use (optional)
    :return: True if successful
    :raises ValueError: If the subtask structure is not registered
    :raises KeyError: If the job does not exist
    :raises RuntimeError: If the database operation fails
    """
    with get_session_context(db_session) as session:
        logger.info(f"Creating atomic subtask structure '{subtask_structure}' for job {job_id}")
        suffix = generate_slug(4)

        # ATOMIC BULK OPERATION: Lock the job and create all subtasks/dependencies atomically
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

            if subtask_structure not in _SUBTASK_STRUCTURES:
                raise ValueError(f"Subtask structure '{subtask_structure}' is not registered")

            structure_func = _SUBTASK_STRUCTURES[subtask_structure]

            # Call the structure function to create all subtasks and dependencies
            # This happens within the same transaction as the job lock
            # Any errors here (like missing subtask types) will cause rollback
            structure_func(
                db_session=session,
                project_name=project_name,
                job_id=job_id,
                batch_id=job_data["batch_id"],
                ng_state=ng_state,
                priority=priority,
                suffix=suffix,
                **subtask_structure_kwargs,
            )

            # Update job status to ingested atomically
            locked_job.status = "ingested"

            # Commit all changes atomically - if this fails, everything rolls back
            session.commit()
            logger.info(
                f"Successfully created subtask structure '{subtask_structure}' for job {job_id}"
            )

            return True

        except Exception as e:
            session.rollback()
            logger.error(
                f"Failed to create subtask structure '{subtask_structure}' for job {job_id}: {e}"
            )
            raise RuntimeError(f"Failed to create subtask structure: {e}") from e


@register_subtask_structure("segmentation_proofread_1pass")
def segmentation_proofread_simple(
    db_session: Session,
    project_name: str,
    job_id: str,
    batch_id: str,
    ng_state: str,
    priority: int,
    suffix: str,
):
    """
    Create a simple segmentation proofread structure with three subtasks:
    1. segmentation_proofread (active)
    2. segmentation_verify (inactive, depends on proofread → done)
    3. segmentation_proofread_expert (inactive, depends on proofread → need_help)

    :param db_session: Database session to use
    :param project_name: Project name
    :param job_id: Job ID
    :param batch_id: Batch ID
    :param ng_state: The ng_state for the subtask
    :param priority: The priority of the subtask
    :param suffix: The suffix for the subtask
    """
    id_suffix = f"_{suffix}"

    # Create base IDs for subtasks and dependencies
    proofread_id = f"{job_id}_proofread{id_suffix}"
    verify_id = f"{job_id}_verify{id_suffix}"
    expert_id = f"{job_id}_proofread_expert{id_suffix}"
    dep_verify_id = f"{job_id}_dep_verify{id_suffix}"
    dep_expert_id = f"{job_id}_dep_expert{id_suffix}"

    # 1. Create segmentation_proofread subtask (active)
    print("Creating segmentation_proofread subtask")
    proofread_subtask = SubtaskModel(
        project_name=project_name,
        subtask_id=proofread_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state,
        ng_state_initial=ng_state,
        priority=priority,
        batch_id=batch_id,
        subtask_type="segmentation_proofread",
        is_active=True,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(proofread_subtask)

    # 2. Create segmentation_verify subtask (inactive)
    print("Creating segmentation_verify subtask")
    verify_subtask = SubtaskModel(
        project_name=project_name,
        subtask_id=verify_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state,
        ng_state_initial=ng_state,
        priority=priority,
        batch_id=batch_id,
        subtask_type="segmentation_verify",
        is_active=False,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(verify_subtask)

    # 3. Create segmentation_proofread_expert subtask (inactive)
    print("Creating segmentation_proofread_expert subtask")
    expert_subtask = SubtaskModel(
        project_name=project_name,
        subtask_id=expert_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state,
        ng_state_initial=ng_state,
        priority=priority,
        batch_id=batch_id,
        subtask_type="segmentation_proofread_expert",
        is_active=False,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(expert_subtask)

    # 4. Create dependency: verify depends on proofread being "done"
    print("Creating dependency for segmentation_verify")
    dep_verify = DependencyModel(
        project_name=project_name,
        dependency_id=dep_verify_id,
        subtask_id=verify_id,
        dependent_on_subtask_id=proofread_id,
        required_completion_status="done",
        is_satisfied=False,
    )
    db_session.add(dep_verify)

    # 5. Create dependency: expert depends on proofread being "need_help"
    print("Creating dependency for segmentation_proofread_expert")
    dep_expert = DependencyModel(
        project_name=project_name,
        dependency_id=dep_expert_id,
        subtask_id=expert_id,
        dependent_on_subtask_id=proofread_id,
        required_completion_status="need_help",
        is_satisfied=False,
    )
    db_session.add(dep_expert)


@register_subtask_structure("segmentation_proofread_simple_1pass")
def segmentation_proofread_simple_1pass(
    db_session: Session,
    project_name: str,
    job_id: str,
    batch_id: str,
    ng_state: str,
    priority: int,
    suffix: str,
):
    """
    Create a simple 1-pass segmentation proofread structure with one subtask.

    :param db_session: Database session to use
    :param project_name: Project name
    :param job_id: Job ID
    :param batch_id: Batch ID
    :param ng_state: The ng_state for the subtask
    :param priority: The priority of the subtask
    :param suffix: The suffix for the subtask
    """
    id_suffix = f"_{suffix}"

    # Create base IDs for subtasks
    proofread_id = f"{job_id}_proofread{id_suffix}"

    # Create segmentation_proofread subtask (active)
    print("Creating segmentation_proofread subtask")
    proofread_subtask = SubtaskModel(
        project_name=project_name,
        subtask_id=proofread_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state,
        ng_state_initial=ng_state,
        priority=priority,
        batch_id=batch_id,
        subtask_type="segmentation_proofread",
        is_active=True,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(proofread_subtask)


@register_subtask_structure("segmentation_proofread_two_path")
def segmentation_proofread_two_path(
    db_session: Session,
    project_name: str,
    job_id: str,
    batch_id: str,
    ng_state: str,
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
    :param ng_state: The ng_state for the subtask
    :param priority: The priority of the subtask
    :param suffix: The suffix for the subtask
    :param validation_layer_path: Path to the validation layer
    """
    id_suffix = f"_{suffix}"

    # Modify ng_state to include validation layer
    ng_state_dict = json.loads(ng_state) if isinstance(ng_state, str) else ng_state

    # Create validation version of ng_state
    ng_state_validation = copy.deepcopy(ng_state_dict)
    if "layers" in ng_state_validation:
        # Add validation layer
        validation_layer = {
            "source": validation_layer_path,
            "type": "segmentation",
            "name": "validation",
        }
        ng_state_validation["layers"].append(validation_layer)

    ng_state_validation_str = json.dumps(ng_state_validation)

    # Create base IDs for subtasks and dependencies
    proofread_id = f"{job_id}_proofread{id_suffix}"
    verify_id = f"{job_id}_verify{id_suffix}"
    expert_id = f"{job_id}_proofread_expert{id_suffix}"
    dep_verify_id = f"{job_id}_dep_verify{id_suffix}"
    dep_expert_id = f"{job_id}_dep_expert{id_suffix}"

    # 1. Create segmentation_proofread subtask (active)
    print("Creating segmentation_proofread subtask")
    proofread_subtask = SubtaskModel(
        project_name=project_name,
        subtask_id=proofread_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state,
        ng_state_initial=ng_state,
        priority=priority,
        batch_id=batch_id,
        subtask_type="segmentation_proofread",
        is_active=True,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(proofread_subtask)

    # 2. Create segmentation_verify subtask with validation layer (inactive)
    print("Creating segmentation_verify subtask with validation layer")
    verify_subtask = SubtaskModel(
        project_name=project_name,
        subtask_id=verify_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state_validation_str,
        ng_state_initial=ng_state_validation_str,
        priority=priority,
        batch_id=batch_id,
        subtask_type="segmentation_verify",
        is_active=False,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(verify_subtask)

    # 3. Create segmentation_proofread_expert subtask (inactive)
    print("Creating segmentation_proofread_expert subtask")
    expert_subtask = SubtaskModel(
        project_name=project_name,
        subtask_id=expert_id,
        job_id=job_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=ng_state,
        ng_state_initial=ng_state,
        priority=priority,
        batch_id=batch_id,
        subtask_type="segmentation_proofread_expert",
        is_active=False,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(expert_subtask)

    # 4. Create dependency: verify depends on proofread being "done"
    print("Creating dependency for segmentation_verify")
    dep_verify = DependencyModel(
        project_name=project_name,
        dependency_id=dep_verify_id,
        subtask_id=verify_id,
        dependent_on_subtask_id=proofread_id,
        required_completion_status="done",
        is_satisfied=False,
    )
    db_session.add(dep_verify)

    # 5. Create dependency: expert depends on proofread being "need_help"
    print("Creating dependency for segmentation_proofread_expert")
    dep_expert = DependencyModel(
        project_name=project_name,
        dependency_id=dep_expert_id,
        subtask_id=expert_id,
        dependent_on_subtask_id=proofread_id,
        required_completion_status="need_help",
        is_satisfied=False,
    )
    db_session.add(dep_expert)
