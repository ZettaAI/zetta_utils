from google.cloud import firestore
from typeguard import typechecked

from zetta_utils.task_management.helpers import get_transaction

from .project import get_collection
from .types import User, UserUpdate


def get_user(project_name: str, user_id: str) -> User:
    """
    Retrieve a user record from the project's User collection.

    :param project_name: The name of the project.
    :param user_id: The unique identifier of the user.
    :return: The user record.
    :raises KeyError: If the user does not exist.
    :raises RuntimeError: If the Firestore transaction fails.
    """
    collection = get_collection(project_name, "users")
    doc = collection.document(user_id).get()
    if not doc.exists:
        raise KeyError(f"User {user_id} not found")
    return doc.to_dict()


@typechecked
def create_user(project_name: str, data: User) -> str:
    """
    Create a new user record in the project's User collection.

    :param project_name: The name of the project.
    :param data: The user data to create, must contain user_id, hourly_rate, and active_subtask.
    :return: The user_id of the created user.
    :raises ValueError: If the user data is invalid or user already exists.
    :raises RuntimeError: If the Firestore transaction fails.
    """

    collection = get_collection(project_name, "users")
    doc_ref = collection.document(data["user_id"])

    @firestore.transactional
    def create_in_transaction(transaction):
        doc = doc_ref.get(transaction=transaction)
        if doc.exists:
            raise ValueError(f"User {data['user_id']} already exists")
        transaction.set(doc_ref, data)
        return data["user_id"]

    return create_in_transaction(get_transaction())


def update_user(project_name: str, user_id: str, data: UserUpdate) -> bool:
    """
    Update a user record in the project's User collection.

    :param project_name: The name of the project.
    :param user_id: The unique identifier of the user.
    :param data: The user data to update.
    :return: True on success.
    :raises ValueError: If the update data is invalid.
    :raises KeyError: If the user does not exist.
    :raises RuntimeError: If the Firestore transaction fails.
    """
    collection = get_collection(project_name, "users")
    doc_ref = collection.document(user_id)

    @firestore.transactional
    def update_in_transaction(transaction):
        doc = doc_ref.get(transaction=transaction)
        if not doc.exists:
            raise KeyError(f"User {user_id} not found")

        transaction.update(doc_ref, data)
        return True

    return update_in_transaction(get_transaction())
