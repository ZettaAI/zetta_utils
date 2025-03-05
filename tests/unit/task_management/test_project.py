# pylint: disable=redefined-outer-name,unused-argument
import pytest
from google.cloud import firestore

from zetta_utils.task_management.project import (
    create_project_tables,
    get_collection,
    get_project,
)


@pytest.fixture
def project_name_project() -> str:
    return "test_project_project"


@pytest.fixture(autouse=True)
def clean_collections(firestore_emulator, project_name_project):
    client = firestore.Client()
    collections = [
        f"{project_name_project}_users",
        f"{project_name_project}_subtasks",
        f"{project_name_project}_timesheets",
        f"{project_name_project}_dependencies",
        f"{project_name_project}_tasks",
        "projects",
    ]
    for coll in collections:
        for doc in client.collection(coll).list_documents():
            doc.delete()
    yield
    for coll in collections:
        for doc in client.collection(coll).list_documents():
            doc.delete()


def test_get_collection(project_name_project):
    """Test that get_collection returns the correct collection reference"""
    collection = get_collection(project_name_project, "users")
    assert collection.id == f"{project_name_project}_users"


def test_create_project_tables(project_name_project):
    """Test creating project tables"""
    # First create the project tables
    create_project_tables(project_name_project)

    # Then verify the project document was created
    client = firestore.Client()
    project_doc = client.collection("projects").document(project_name_project).get()
    assert project_doc.exists
    assert project_doc.get("project_name") == project_name_project


def test_get_project_success(project_name_project):
    """Test getting a project that exists"""
    create_project_tables(project_name_project)

    project = get_project(project_name_project)
    assert project["project_name"] == project_name_project


def test_get_project_not_found():
    """Test that getting a non-existent project raises KeyError"""
    with pytest.raises(KeyError, match="Project non_existent_project not found"):
        get_project("non_existent_project")
