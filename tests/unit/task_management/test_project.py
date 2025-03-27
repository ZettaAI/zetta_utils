# pylint: disable=redefined-outer-name,unused-argument
import pytest
from google.cloud import firestore

from zetta_utils.task_management.project import create_project_tables, get_project


def test_create_project_tables(project_name):
    """Test creating project tables"""
    # First create the project tables
    create_project_tables(project_name)

    # Then verify the project document was created
    client = firestore.Client()
    project_doc = client.collection("projects").document(project_name).get()
    assert project_doc.exists
    assert project_doc.get("project_name") == project_name


def test_get_project_success(project_name):
    """Test getting a project that exists"""
    create_project_tables(project_name)

    project = get_project(project_name)
    assert project["project_name"] == project_name


def test_get_project_not_found():
    """Test that getting a non-existent project raises KeyError"""
    with pytest.raises(KeyError, match="Project non_existent_project not found"):
        get_project("non_existent_project")
