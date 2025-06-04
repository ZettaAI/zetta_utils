import copy
import json
from typing import Any, Callable, Mapping

from coolname import generate_slug
from sqlalchemy import select
from sqlalchemy.orm import Session

from zetta_utils.log import get_logger
from zetta_utils.task_management.db.models import (
    DependencyModel,
    SubtaskModel,
    TaskModel,
)
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
    db_session: Session,
    project_name: str,
    task_id: str,
    subtask_structure: str,
    subtask_structure_kwargs: Mapping[str, Any],
    priority: int = 1,
) -> bool:
    """
    Create a predefined subtask structure for a task.

    :param db_session: Database session to use
    :param project_name: The name of the project
    :param task_id: The ID of the task
    :param subtask_structure: The name of the subtask structure to create
    :param priority: The priority of the subtask
    :return: True if successful
    :raises ValueError: If the subtask structure is not registered
    :raises KeyError: If the task does not exist
    :raises RuntimeError: If the database operation fails
    """
    logger.info(f"Creating atomic subtask structure '{subtask_structure}' for task {task_id}")
    suffix = generate_slug(4)

    # ATOMIC BULK OPERATION: Lock the task and create all subtasks/dependencies atomically
    try:
        # Lock the task to prevent concurrent modifications
        task_lock_query = (
            select(TaskModel)
            .where(TaskModel.project_name == project_name)
            .where(TaskModel.task_id == task_id)
            .with_for_update()
        )
        locked_task = db_session.execute(task_lock_query).scalar_one()
        task_data = locked_task.to_dict()
        ng_state = task_data["ng_state"]

        if subtask_structure not in _SUBTASK_STRUCTURES:
            raise ValueError(f"Subtask structure '{subtask_structure}' is not registered")

        structure_func = _SUBTASK_STRUCTURES[subtask_structure]

        # Call the structure function to create all subtasks and dependencies
        # This happens within the same transaction as the task lock
        # Any errors here (like missing subtask types) will cause rollback
        structure_func(
            db_session=db_session,
            project_name=project_name,
            task_id=task_id,
            batch_id=task_data["batch_id"],
            ng_state=ng_state,
            priority=priority,
            suffix=suffix,
            **subtask_structure_kwargs,
        )

        # Update task status to ingested atomically
        locked_task.status = "ingested"

        # Commit all changes atomically - if this fails, everything rolls back
        db_session.commit()
        logger.info(
            f"Successfully created subtask structure '{subtask_structure}' for task {task_id}"
        )

        return True

    except Exception as e:
        db_session.rollback()
        logger.error(
            f"Failed to create subtask structure '{subtask_structure}' for task {task_id}: {e}"
        )
        raise RuntimeError(f"Failed to create subtask structure: {e}") from e


@register_subtask_structure("segmentation_proofread_1pass")
def segmentation_proofread_simple(
    db_session: Session,
    project_name: str,
    task_id: str,
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
    :param task_id: Task ID
    :param batch_id: Batch ID
    :param ng_state: The ng_state for the subtask
    :param priority: The priority of the subtask
    :param suffix: The suffix for the subtask
    """
    id_suffix = f"_{suffix}"

    # Create base IDs for subtasks and dependencies
    proofread_id = f"{task_id}_proofread{id_suffix}"
    verify_id = f"{task_id}_verify{id_suffix}"
    expert_id = f"{task_id}_proofread_expert{id_suffix}"
    dep_verify_id = f"{task_id}_dep_verify{id_suffix}"
    dep_expert_id = f"{task_id}_dep_expert{id_suffix}"

    # 1. Create segmentation_proofread subtask (active)
    print("Creating segmentation_proofread subtask")
    proofread_subtask = SubtaskModel(
        project_name=project_name,
        subtask_id=proofread_id,
        task_id=task_id,
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
        task_id=task_id,
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
        task_id=task_id,
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

    # 4. Create dependency for verify (depends on proofread → done)
    print("Creating dependency for verify")
    verify_dependency = DependencyModel(
        project_name=project_name,
        dependency_id=dep_verify_id,
        subtask_id=verify_id,
        dependent_on_subtask_id=proofread_id,
        required_completion_status="done",
        is_satisfied=False,
    )
    db_session.add(verify_dependency)

    # 5. Create dependency for expert (depends on proofread → need_help)
    print("Creating dependency for expert")
    expert_dependency = DependencyModel(
        project_name=project_name,
        dependency_id=dep_expert_id,
        subtask_id=expert_id,
        dependent_on_subtask_id=proofread_id,
        required_completion_status="need_help",
        is_satisfied=False,
    )
    db_session.add(expert_dependency)


@register_subtask_structure("segmentation_proofread_simple_1pass")
def segmentation_proofread_simple_1pass(
    db_session: Session,
    project_name: str,
    task_id: str,
    batch_id: str,
    ng_state: str,
    priority: int,
    suffix: str,
):
    """
    Create a simple segmentation proofread structure with just one subtask:
    1. segmentation_proofread (active)

    :param db_session: Database session to use
    :param project_name: Project name
    :param task_id: Task ID
    :param batch_id: Batch ID
    :param ng_state: The ng_state for the subtask
    :param priority: The priority of the subtask
    :param suffix: The suffix for the subtask
    """
    id_suffix = f"_{suffix}"

    # Create base ID for the subtask
    proofread_id = f"{task_id}_proofread{id_suffix}"

    # Create segmentation_proofread subtask (active)
    proofread_subtask = SubtaskModel(
        project_name=project_name,
        subtask_id=proofread_id,
        task_id=task_id,
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
    task_id: str,
    batch_id: str,
    ng_state: str,
    priority: int,
    suffix: str,
    validation_layer_path: str,
):
    """
    Create a two-path segmentation proofread structure with a consolidation step.

    This structure creates two independent proofread subtasks. When both are completed,
    a consolidation subtask becomes active to merge the results.

    :param db_session: Database session to use
    :param project_name: Project name
    :param task_id: ID of the parent task
    :param batch_id: ID of the batch
    :param ng_state: Neuroglancer state JSON
    :param priority: Priority of the subtasks
    :param suffix: Suffix for the subtask IDs
    """
    ng_state_parsed = json.loads(ng_state)
    ng_state_val = copy.deepcopy(ng_state_parsed)
    ng_state_cons = copy.deepcopy(ng_state_parsed)

    segmentation_found = False
    for layer in ng_state_val["layers"]:
        if layer["name"] == "Segmentation":
            segmentation_found = True
            layer["source"] = validation_layer_path
            break

    if not segmentation_found:
        raise ValueError("No `Segmentation` layer found in Neuroglancer state")

    ng_state_cons["layers"].append(
        {
            "type": "segmentation",
            "source": validation_layer_path,
            "segments": [],
            "name": "Validation Segmentation",
        }
    )

    # Generate IDs
    proofread1_id = f"{task_id}_proofread_{suffix}"
    proofread2_id = f"{task_id}_proofread_val_{suffix}"
    consolidate_id = f"{task_id}_consolidate_{suffix}"

    # Dependency IDs
    dep_consolidate_from_p1_id = f"{consolidate_id}_from_p1_{suffix}"
    dep_consolidate_from_p2_id = f"{consolidate_id}_from_p2_{suffix}"

    # 1. Create first proofread subtask
    proofread1_subtask = SubtaskModel(
        project_name=project_name,
        subtask_id=proofread1_id,
        task_id=task_id,
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
    db_session.add(proofread1_subtask)

    # 2. Create second proofread subtask
    proofread2_subtask = SubtaskModel(
        project_name=project_name,
        subtask_id=proofread2_id,
        task_id=task_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=json.dumps(ng_state_val),
        ng_state_initial=json.dumps(ng_state_val),
        priority=priority,
        batch_id=batch_id,
        subtask_type="segmentation_proofread",
        is_active=True,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(proofread2_subtask)

    # 3. Create consolidation subtask (inactive until both proofreads are done)
    consolidate_subtask = SubtaskModel(
        project_name=project_name,
        subtask_id=consolidate_id,
        task_id=task_id,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        ng_state=json.dumps(ng_state_cons),
        ng_state_initial=json.dumps(ng_state_cons),
        priority=priority,
        batch_id=batch_id,
        subtask_type="segmentation_consolidate",
        is_active=False,
        last_leased_ts=0.0,
        completion_status="",
        id_nonunique=generate_id_nonunique(),
    )
    db_session.add(consolidate_subtask)

    # 4. Create dependency for consolidate (depends on proofread1 → done)
    consolidate_dependency1 = DependencyModel(
        project_name=project_name,
        dependency_id=dep_consolidate_from_p1_id,
        subtask_id=consolidate_id,
        dependent_on_subtask_id=proofread1_id,
        required_completion_status="done",
        is_satisfied=False,
    )
    db_session.add(consolidate_dependency1)

    # 5. Create dependency for consolidate (depends on proofread2 → done)
    consolidate_dependency2 = DependencyModel(
        project_name=project_name,
        dependency_id=dep_consolidate_from_p2_id,
        subtask_id=consolidate_id,
        dependent_on_subtask_id=proofread2_id,
        required_completion_status="done",
        is_satisfied=False,
    )
    db_session.add(consolidate_dependency2)
