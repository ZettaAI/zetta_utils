from google.cloud import firestore
from typeguard import typechecked

from .helpers import get_transaction
from .project import get_collection
from .types import Dependency, DependencyUpdate


def get_dependency(project_name: str, dependency_id: str) -> Dependency:
    """
    Retrieve a dependency record from the project's Dependency collection.

    :param project_name: The name of the project.
    :param dependency_id: The unique identifier of the dependency.
    :return: The dependency record.
    :raises KeyError: If the dependency does not exist.
    :raises RuntimeError: If the Firestore transaction fails.
    """
    collection = get_collection(project_name, "dependencies")
    doc = collection.document(dependency_id).get()
    if not doc.exists:
        raise KeyError(f"Dependency {dependency_id} not found")
    return doc.to_dict()


@typechecked
def create_dependency(project_name: str, data: Dependency) -> str:
    """
    Create a new dependency record in the project's Dependency collection.

    :param project_name: The name of the project.
    :param data: The dependency data to create.
    :return: The dependency_id of the created dependency.
    :raises ValueError: If the dependency data is invalid or already exists.
    :raises RuntimeError: If the Firestore transaction fails.
    """

    if data["dependent_subtask_id"] == data["dependent_on_subtask_id"]:
        raise ValueError("Subtask cannot depend on itself")

    collection = get_collection(project_name, "dependencies")
    doc_ref = collection.document(data["dependency_id"])

    @firestore.transactional
    def create_in_transaction(transaction):
        doc = doc_ref.get(transaction=transaction)
        if doc.exists:
            raise ValueError(f"Dependency {data['dependency_id']} already exists")

        # Verify both subtasks exist
        subtask_collection = get_collection(project_name, "subtasks")
        subtask_doc = subtask_collection.document(data["dependent_subtask_id"]).get(
            transaction=transaction
        )
        if not subtask_doc.exists:
            raise ValueError(f"Subtask {data['dependent_subtask_id']} not found")

        dependent_on_doc = subtask_collection.document(data["dependent_on_subtask_id"]).get(
            transaction=transaction
        )
        if not dependent_on_doc.exists:
            raise ValueError(f"Subtask {data['dependent_on_subtask_id']} not found")

        # Set is_satisfied to False by default
        data.setdefault("is_satisfied", False)

        transaction.set(doc_ref, data)
        return data["dependency_id"]

    return create_in_transaction(get_transaction())


@typechecked
def update_dependency(project_name: str, dependency_id: str, data: DependencyUpdate) -> bool:
    """
    Update a dependency record in the project's Dependency collection.

    :param project_name: The name of the project.
    :param dependency_id: The unique identifier of the dependency.
    :param data: The dependency data to update.
    :return: True on success.
    :raises ValueError: If the update data is invalid.
    :raises KeyError: If the dependency does not exist.
    :raises RuntimeError: If the Firestore transaction fails.
    """
    collection = get_collection(project_name, "dependencies")
    doc_ref = collection.document(dependency_id)

    @firestore.transactional
    def update_in_transaction(transaction):
        doc = doc_ref.get(transaction=transaction)
        if not doc.exists:
            raise KeyError(f"Dependency {dependency_id} not found")

        transaction.update(doc_ref, data)
        return True

    return update_in_transaction(get_transaction())
