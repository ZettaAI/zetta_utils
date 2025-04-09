import os
from typing import Any, Literal

from google.api_core import exceptions
from google.cloud import firestore, firestore_admin_v1

# Configure default client settings
DEFAULT_CLIENT_CONFIG: dict[str, str] = {
    "project": "zetta-research",
    "database": "task-management",
}


def get_firestore_client() -> firestore.Client:  # pragma: no cover
    """Get a configured Firestore client"""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        config = {}
    else:
        config = DEFAULT_CLIENT_CONFIG.copy()  # pragma: no cover
    return firestore.Client(**config)


CollectionType = Literal["users", "subtasks", "timesheets", "dependencies", "tasks"]


def get_collection(project_name: str, collection_type: str) -> firestore.CollectionReference:
    """Get a collection reference with the proper project prefix"""
    client = get_firestore_client()
    result = client.collection(f"projects/{project_name}/{collection_type}")
    return result


def _create_indexes(
    project_name: str, admin_client: firestore_admin_v1.FirestoreAdminClient
) -> None:  # pragma: no cover
    """Create required indexes for a project's collections.

    :param project_name: Name of the project
    :param admin_client: Firestore admin client
    """
    # Create required indexes for subtasks
    subtasks_parent = (
        f"projects/{DEFAULT_CLIENT_CONFIG['project']}/databases/"
        f"{DEFAULT_CLIENT_CONFIG['database']}/collectionGroups/"
        f"subtasks"
    )
    subtask_indexes = [
        firestore_admin_v1.Index(
            query_scope=firestore_admin_v1.Index.QueryScope.COLLECTION,
            fields=[
                firestore_admin_v1.Index.IndexField(
                    field_path="is_active",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="assigned_user_id",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="active_user_id",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="completion_status",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="subtask_type",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="priority",
                    order=firestore_admin_v1.Index.IndexField.Order.DESCENDING,
                ),
            ],
        ),
        firestore_admin_v1.Index(
            query_scope=firestore_admin_v1.Index.QueryScope.COLLECTION,
            fields=[
                firestore_admin_v1.Index.IndexField(
                    field_path="is_active",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="assigned_user_id",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="active_user_id",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="completion_status",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="subtask_type",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="priority",
                    order=firestore_admin_v1.Index.IndexField.Order.DESCENDING,
                ),
            ],
        ),
        firestore_admin_v1.Index(
            query_scope=firestore_admin_v1.Index.QueryScope.COLLECTION,
            fields=[
                firestore_admin_v1.Index.IndexField(
                    field_path="is_active",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="completion_status",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="subtask_type",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="last_leased_ts",
                    order=firestore_admin_v1.Index.IndexField.Order.DESCENDING,
                ),
            ],
        ),
        firestore_admin_v1.Index(
            query_scope=firestore_admin_v1.Index.QueryScope.COLLECTION,
            fields=[
                firestore_admin_v1.Index.IndexField(
                    field_path="task_id", order=firestore_admin_v1.Index.IndexField.Order.ASCENDING
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="is_active",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="completion_status",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
            ],
        ),
    ]

    # For dependencies collection
    deps_parent = (
        f"projects/{DEFAULT_CLIENT_CONFIG['project']}/databases/"
        f"{DEFAULT_CLIENT_CONFIG['database']}/collectionGroups/"
        f"dependencies"
    )
    dependency_indexes = [
        firestore_admin_v1.Index(
            query_scope=firestore_admin_v1.Index.QueryScope.COLLECTION,
            fields=[
                firestore_admin_v1.Index.IndexField(
                    field_path="dependent_on_subtask_id",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
                firestore_admin_v1.Index.IndexField(
                    field_path="is_satisfied",
                    order=firestore_admin_v1.Index.IndexField.Order.ASCENDING,
                ),
            ],
        )
    ]

    # Create all indexes
    for parent, indexes in [(subtasks_parent, subtask_indexes), (deps_parent, dependency_indexes)]:
        for index in indexes:
            try:
                print (parent, index)
                admin_client.create_index(parent=parent, index=index)
                print(f"Created index: {index}")
            except exceptions.AlreadyExists:
                print("Index already exists")
    print("All index creation operations completed")


def create_project_tables(project_name: str) -> None:
    """Create any missing collections and required indexes for a project."""
    client = get_firestore_client()
    collections = ["users", "subtasks", "timesheets", "dependencies", "tasks"]

    # Add project to global projects collection if it doesn't exist
    projects_ref = client.collection("projects")
    project_doc = projects_ref.document(project_name)
    if not project_doc.get().exists:
        project_doc.set({"project_name": project_name, "created_ts": firestore.SERVER_TIMESTAMP})

    for coll_name in collections:
        get_collection(project_name, coll_name)

    if not os.environ.get("FIRESTORE_EMULATOR_HOST"):  # pragma: no cover
        admin_client = firestore_admin_v1.FirestoreAdminClient()
        _create_indexes(project_name, admin_client)


def get_project(project_name: str) -> dict[str, Any]:
    """
    Get a project by name.

    :param project_name: The name of the project.
    :return: The project data.
    :raises KeyError: If the project does not exist.
    """
    client = get_firestore_client()
    doc = client.collection("projects").document(project_name).get()
    if not doc.exists:
        raise KeyError(f"Project {project_name} not found")
    return doc.to_dict()
