from typeguard import typechecked

from .project import get_collection
from .types import SubtaskType


def get_subtask_type(project_name: str, subtask_type_id: str) -> SubtaskType:
    """
    Get a subtask type from the global subtask_types collection.

    :param subtask_type_id: The unique identifier of the subtask type.
    :return: The subtask type data.
    :raises KeyError: If the subtask type does not exist.
    :raises RuntimeError: If the Firestore operation fails.
    """
    doc = get_collection(project_name, "subtask_types").document(subtask_type_id).get()
    if not doc.exists:
        raise KeyError(f"Subtask type not found: {subtask_type_id}")
    return doc.to_dict()


@typechecked
def create_subtask_type(project_name: str, data: SubtaskType) -> str:
    """
    Create a new subtask type in the global subtask_types collection.

    :param data: The subtask type data to create.
    :return: The subtask_type that was created.
    :raises ValueError: If the subtask type data is invalid or already exists.
    :raises RuntimeError: If the Firestore operation fails.
    """
    doc_ref =get_collection(project_name, "subtask_types").document(data["subtask_type"])

    if doc_ref.get().exists:
        raise ValueError(f"Subtask type already exists: {data['subtask_type']}")

    doc_ref.set(data)
    return data["subtask_type"]
