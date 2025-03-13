import time
from typing import Optional

from google.cloud import firestore
from google.cloud.firestore_v1.base_document import DocumentSnapshot
from typeguard import typechecked

from .exceptions import SubtaskValidationError, UserValidationError
from .helpers import get_transaction, retry_transient_errors
from .project import get_collection, get_firestore_client
from .subtask_type import get_subtask_type
from .types import Subtask, SubtaskUpdate

_MAX_IDLE_SECONDS = 90


def get_max_idle_seconds() -> float:
    return _MAX_IDLE_SECONDS


def _validate_subtask(subtask: dict) -> Subtask:
    """
    Validate that a subtask's data is consistent and valid.

    :param project_name: The name of the project.
    :param subtask: The subtask data to validate
    :raises SubtaskValidationError: If the subtask data is invalid
    """
    try:
        subtask_type = get_subtask_type(subtask["subtask_type"])
    except KeyError as e:
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
        if "completion_statuses" not in subtask_type:
            raise SubtaskValidationError(
                f"Subtask type {subtask['subtask_type']} has no valid completion statuses"
            )
        if completion_status not in subtask_type["completion_statuses"]:
            raise SubtaskValidationError(
                f"Completion status '{completion_status}' not allowed for this subtask type"
            )

        # If there's a completion status, must have completed user
        if not subtask.get("completed_user_id"):
            raise SubtaskValidationError("Completed subtask must have completed_user_id")

    return Subtask(**subtask)  # type: ignore # typeguard will check for us


@typechecked
def create_subtask(project_name: str, data: Subtask) -> str:
    """Create a new subtask record"""
    get_subtask_type(data["subtask_type"])

    collection = get_collection(project_name, "subtasks")
    doc_ref = collection.document(data["subtask_id"])

    @firestore.transactional
    def create_in_transaction(transaction):
        doc = doc_ref.get(transaction=transaction)
        if doc.exists:
            raise SubtaskValidationError(f"Subtask {data['subtask_id']} already exists")
        transaction.set(doc_ref, data)
        return data["subtask_id"]

    return create_in_transaction(get_transaction())


@typechecked
def update_subtask(project_name: str, subtask_id: str, data: SubtaskUpdate) -> bool:
    """Update a subtask record"""
    collection = get_collection(project_name, "subtasks")
    doc_ref = collection.document(subtask_id)

    @firestore.transactional
    def update_in_transaction(transaction):
        doc = doc_ref.get(transaction=transaction)
        if not doc.exists:
            raise KeyError(f"Subtask {subtask_id} not found")

        current_data = doc.to_dict()
        merged_data = {**current_data, **data}
        _validate_subtask(merged_data)

        # If completion status is changing, handle side effects
        if "completion_status" in data and "completed_user_id" in data:
            updates = _handle_subtask_completion(
                transaction,
                project_name,
                subtask_id,
                data["completion_status"],
            )
            # Apply all completion-related updates
            for ref, update_data in updates:
                transaction.update(ref, update_data)

        # Apply the main update
        transaction.update(doc_ref, data)
        return True

    return update_in_transaction(get_transaction())


@retry_transient_errors
def start_subtask(project_name: str, user_id: str, subtask_id: Optional[str] = None) -> str | None:
    client = get_firestore_client()
    user_ref = client.collection(f"{project_name}_users").document(user_id)

    @firestore.transactional
    def start_in_transaction(transaction):
        user_doc = user_ref.get(transaction=transaction)
        if not user_doc.exists:
            raise UserValidationError(f"User {user_id} not found")

        user_data = user_doc.to_dict()
        current_active_subtask_id = user_data.get("active_subtask")
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
            selected_subtask = _auto_select_subtask(client, project_name, user_id, transaction)
        elif subtask_id is not None:
            selected_subtask = (
                client.collection(f"{project_name}_subtasks")
                .document(subtask_id)
                .get(transaction=transaction)
            )
            assert selected_subtask is not None
            if not selected_subtask.exists:
                raise SubtaskValidationError(f"Subtask {subtask_id} not found")
        else:
            selected_subtask = (
                client.collection(f"{project_name}_subtasks")
                .document(current_active_subtask_id)
                .get(transaction=transaction)
            )
            assert selected_subtask is not None
            assert selected_subtask.exists

        if selected_subtask is not None:
            selected_subtask_data = selected_subtask.to_dict()
            assert selected_subtask_data is not None
            subtask_data = _validate_subtask(selected_subtask_data)

            # Check if user is qualified for this subtask type
            if "qualified_subtask_types" in user_data and subtask_data[
                "subtask_type"
            ] not in user_data.get("qualified_subtask_types", []):
                raise UserValidationError("User not qualified for this subtask type")

            current_time = time.time()
            # Check if task is idle and can be taken over
            if subtask_data["active_user_id"] != "":
                if (
                    subtask_data["last_leased_ts"] <= current_time - get_max_idle_seconds()
                    or subtask_data["active_user_id"] == user_id
                ):
                    previous_user_ref = client.collection(f"{project_name}_users").document(
                        subtask_data["active_user_id"]
                    )
                    transaction.update(previous_user_ref, {"active_subtask": ""})
                    transaction.update(selected_subtask.reference, {"active_user_id": ""})
                else:
                    raise SubtaskValidationError("Subtask is already active")

            transaction.update(user_ref, {"active_subtask": selected_subtask.id})
            transaction.update(
                selected_subtask.reference,
                {
                    "active_user_id": user_id,
                    "last_leased_ts": current_time,
                },
            )
            return selected_subtask.id
        return None

    return start_in_transaction(get_transaction())


@retry_transient_errors
def release_subtask(
    project_name: str, user_id: str, subtask_id: str, completion_status: str = ""
) -> bool:
    """
    Releases the active subtask for a user within the project.

    :param project_name: The name of the project.
    :param user_id: The unique identifier of the user.
    :param completion_status: The completion status to set for the subtask upon release.
        Empty string means not completed.
    :return: True if the operation completes successfully
    :raises SubtaskValidationError: If the subtask validation fails
    :raises UserValidationError: If the user validation fails
    :raises ValueError: If the completion status is invalid
    :raises RuntimeError: If the Firestore transaction fails.
    """
    client = get_firestore_client()
    user_ref = client.collection(f"{project_name}_users").document(user_id)

    @firestore.transactional
    def release_in_transaction(transaction):
        # Get all the data we need first
        user_doc = user_ref.get(transaction=transaction)
        if not user_doc.exists:
            raise UserValidationError(f"User {user_id} not found")

        user_data = user_doc.to_dict()
        if user_data["active_subtask"] == "":
            raise UserValidationError("User does not have an active subtask")

        # Check that the subtask being released matches the user's active subtask
        if user_data["active_subtask"] != subtask_id:
            raise UserValidationError("Subtask ID does not match user's active subtask")

        subtask_ref = client.collection(f"{project_name}_subtasks").document(subtask_id)
        subtask_doc = subtask_ref.get(transaction=transaction)
        if not subtask_doc.exists:
            raise SubtaskValidationError(f"Subtask {subtask_id} not found")

        # Prepare updates
        updates = []
        if completion_status:
            updates.extend(
                _handle_subtask_completion(
                    transaction,
                    project_name,
                    subtask_id,
                    completion_status,
                )
            )

        # Add basic release updates
        updates.append(
            (
                subtask_ref,
                {
                    "active_user_id": "",
                    "completion_status": completion_status,
                    "completed_user_id": user_id if completion_status else "",
                },
            )
        )
        updates.append((user_ref, {"active_subtask": ""}))

        # Apply all updates
        for ref, update_data in updates:
            transaction.update(ref, update_data)

        return True

    return release_in_transaction(get_transaction())


def _handle_subtask_completion(
    transaction: firestore.Transaction,
    project_name: str,
    subtask_id: str,
    completion_status: str,
) -> list[tuple[DocumentSnapshot, dict]]:
    """
    Handle all side effects of completing a subtask.

    :param transaction: The current Firestore transaction
    :param project_name: The name of the project
    :param subtask_id: The ID of completed subtask
    :param completion_status: The new completion status
    :param completed_user_id: The user who completed the subtask
    :return: List of (doc_ref, update_data) tuples to be applied
    :raises RuntimeError: If the Firestore transaction fails
    """
    client = get_firestore_client()
    updates: list[tuple[DocumentSnapshot, dict]] = []

    # Get the subtask and its dependencies
    subtask_ref = client.collection(f"{project_name}_subtasks").document(subtask_id)
    subtask_doc = subtask_ref.get(transaction=transaction)
    subtask_data = subtask_doc.to_dict()

    # Get dependencies that depend on this subtask
    deps = (
        client.collection(f"{project_name}_dependencies")
        .where("dependent_on_subtask_id", "==", subtask_id)
        .where("is_satisfied", "==", False)
        .get(transaction=transaction)
    )

    # Update dependencies and dependent subtasks
    for dep in deps:
        dep_data = dep.to_dict()
        if dep_data["required_completion_status"] == completion_status:
            # Mark dependency satisfied
            updates.append((dep.reference, {"is_satisfied": True}))

            # Check if dependent subtask can be activated
            dependent_subtask_id = dep_data["subtask_id"]
            other_deps = (
                client.collection(f"{project_name}_dependencies")
                .where("subtask_id", "==", dependent_subtask_id)
                .where("is_satisfied", "==", False)
                .get(transaction=transaction)
            )

            # If this was the last unsatisfied dependency
            if len(list(other_deps)) <= 1:
                dependent_subtask_ref = client.collection(f"{project_name}_subtasks").document(
                    dependent_subtask_id
                )
                updates.append((dependent_subtask_ref, {"is_active": True}))

    # Check if task is complete
    task_id = subtask_data["task_id"]
    incomplete_subtasks = (
        client.collection(f"{project_name}_subtasks")
        .where("task_id", "==", task_id)
        .where("is_active", "==", True)
        .where("completion_status", "==", "")
        .get(transaction=transaction)
    )

    # If this was the last incomplete subtask
    if len(list(incomplete_subtasks)) == 1:
        task_ref = client.collection(f"{project_name}_tasks").document(task_id)
        task_doc = task_ref.get(transaction=transaction)
        if task_doc.exists:
            updates.append((task_ref, {"status": "fully_processed"}))

    return updates


def _auto_select_subtask(
    client: firestore.Client, project_name: str, user_id: str, transaction: firestore.Transaction
) -> DocumentSnapshot | None:
    """Auto-select a subtask for a user based on priority and qualifications.

    Selection criteria (in order):
    1. Assigned to user & matches qualified types (highest priority)
    2. Unassigned & matches qualified types (highest priority)
    3. Matches qualified types & idle > max idle seconds (most recently active)
    """
    user_doc = (
        client.collection(f"{project_name}_users").document(user_id).get(transaction=transaction)
    )
    assert user_doc.exists

    qualified_types = user_doc.get("qualified_subtask_types")
    if not qualified_types:
        return None

    subtasks_collection = client.collection(f"{project_name}_subtasks")
    current_time = time.time()

    # 1. Check for tasks already assigned to user
    for subtask_type in qualified_types:
        query = (
            subtasks_collection.where("is_active", "==", True)
            .where("assigned_user_id", "==", user_id)
            .where("active_user_id", "==", "")
            .where("completion_status", "==", "")
            .where("subtask_type", "==", subtask_type)
            .order_by("priority", direction=firestore.Query.DESCENDING)
            .limit(1)
        )
        results = list(query.get(transaction=transaction))
        if results:
            return results[0]

    # 2. Check for unassigned tasks
    for subtask_type in qualified_types:
        query = (
            subtasks_collection.where("is_active", "==", True)
            .where("assigned_user_id", "==", "")
            .where("active_user_id", "==", "")
            .where("completion_status", "==", "")
            .where("subtask_type", "==", subtask_type)
            .order_by("priority", direction=firestore.Query.DESCENDING)
            .limit(1)
        )
        results = list(query.get(transaction=transaction))
        if results:
            return results[0]

    # 3. Check for idle tasks
    oldest_allowed_ts = current_time - get_max_idle_seconds()
    for subtask_type in qualified_types:
        query = (
            subtasks_collection.where("is_active", "==", True)
            .where("completion_status", "==", "")
            .where("subtask_type", "==", subtask_type)
            .where("last_leased_ts", "<", oldest_allowed_ts)
            .order_by("last_leased_ts", direction=firestore.Query.DESCENDING)
            .limit(1)
        )
        results = list(query.get(transaction=transaction))
        if results:
            return results[0]

    return None


def get_subtask(project_name: str, subtask_id: str) -> Subtask:
    """
    Retrieve a subtask record from the project's Subtask collection.

    :param project_name: The name of the project.
    :param subtask_id: The unique identifier of the subtask.
    :return: The subtask record.
    :raises KeyError: If the subtask does not exist.
    :raises RuntimeError: If the Firestore transaction fails.
    """
    collection = get_collection(project_name, "subtasks")
    doc = collection.document(subtask_id).get()
    if not doc.exists:
        raise KeyError(f"Subtask {subtask_id} not found")
    return doc.to_dict()
