from typing import Any, Callable, Mapping

from coolname import generate_slug
from google.cloud import firestore

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


def create_subtask_dag(
    client: firestore.Client,
    transaction: firestore.Transaction,
    project_name: str,
    task_id: str,
    subtask_structure: str,
    priority: int = 1,
    subtask_structure_kwargs: Mapping[str, Any] | None = None,
) -> bool:
    """
    Create a predefined subtask structure for a task.

    :param client: Firestore client to use
    :param transaction: Existing transaction to use
    :param project_name: The name of the project
    :param task_id: The ID of the task
    :param subtask_structure: The name of the subtask structure to create
    :param priority: The priority of the subtask
    :return: True if successful
    :raises ValueError: If the subtask structure is not registered
    :raises KeyError: If the task does not exist
    """
    suffix = generate_slug(4)

    # Get the task reference
    task_ref = get_collection(project_name, "tasks").document(task_id)

    task_doc = task_ref.get()
    if not task_doc.exists:
        raise KeyError(f"Task {task_id} not found")

    task_data = task_doc.to_dict()

    ng_state = task_data["ng_state"]
    if subtask_structure_kwargs is None:
        subtask_structure_kwargs = {}

    structure_func = _SUBTASK_STRUCTURES[subtask_structure]
    structure_func(
        client=client,
        transaction=transaction,
        project_name=project_name,
        task_id=task_id,
        batch_id=task_data["batch_id"],
        ng_state=ng_state,
        priority=priority,
        suffix=suffix,
        **subtask_structure_kwargs,
    )

    transaction.update(task_ref, {"status": "ingested"})
    return True


@register_subtask_structure("segmentation_proofread_simple")
def segmentation_proofread_simple(
    client: firestore.Client,
    transaction: firestore.Transaction,
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

    :param client: Firestore client
    :param transaction: Current transaction
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

    # Collection references
    subtasks_coll = get_collection(project_name, "subtasks")
    deps_coll = get_collection(project_name, "dependencies")

    # 1. Create segmentation_proofread subtask (active)
    proofread_subtask = {
        "task_id": task_id,
        "subtask_id": proofread_id,
        "assigned_user_id": "",
        "active_user_id": "",
        "completed_user_id": "",
        "ng_state_initial": ng_state,
        "ng_state": ng_state,
        "priority": priority,
        "batch_id": batch_id,
        "subtask_type": "segmentation_proofread",
        "is_active": True,
        "last_leased_ts": 0.0,
        "completion_status": "",
    }
    transaction.set(subtasks_coll.document(proofread_id), proofread_subtask)

    # 2. Create segmentation_verify subtask (inactive)
    verify_subtask = {
        "task_id": task_id,
        "subtask_id": verify_id,
        "assigned_user_id": "",
        "active_user_id": "",
        "completed_user_id": "",
        "ng_state_initial": ng_state,
        "ng_state": ng_state,
        "priority": priority,
        "batch_id": batch_id,
        "subtask_type": "segmentation_verify",
        "is_active": False,
        "last_leased_ts": 0.0,
        "completion_status": "",
    }
    transaction.set(subtasks_coll.document(verify_id), verify_subtask)

    # 3. Create segmentation_proofread_expert subtask (inactive)
    expert_subtask = {
        "task_id": task_id,
        "subtask_id": expert_id,
        "assigned_user_id": "",
        "active_user_id": "",
        "completed_user_id": "",
        "ng_state_initial": ng_state,
        "ng_state": ng_state,
        "priority": priority,
        "batch_id": batch_id,
        "subtask_type": "segmentation_proofread_expert",
        "is_active": False,
        "last_leased_ts": 0.0,
        "completion_status": "",
    }
    transaction.set(subtasks_coll.document(expert_id), expert_subtask)

    # 4. Create dependency for verify (depends on proofread → done)
    verify_dependency = {
        "dependency_id": dep_verify_id,
        "dependent_subtask_id": verify_id,
        "prerequisite_subtask_id": proofread_id,
        "required_completion_status": "done",
        "is_satisfied": False,
    }
    transaction.set(deps_coll.document(dep_verify_id), verify_dependency)

    # 5. Create dependency for expert (depends on proofread → need_help)
    expert_dependency = {
        "dependency_id": dep_expert_id,
        "dependent_subtask_id": expert_id,
        "prerequisite_subtask_id": proofread_id,
        "required_completion_status": "need_help",
        "is_satisfied": False,
    }
    transaction.set(deps_coll.document(dep_expert_id), expert_dependency)
