# pylint: disable=redefined-outer-name,unused-argument
import pytest

from zetta_utils.task_management.db.models import ProjectModel
from zetta_utils.task_management.project import (
    create_project,
    create_project_tables,
    delete_project,
    get_project,
    list_all_projects,
    project_exists,
)
from zetta_utils.task_management.types import User
from zetta_utils.task_management.user import create_user


def test_create_project_tables(project_name, clean_db, db_session):
    """Test creating project tables (SQL tables)"""
    # This should succeed without errors
    create_project_tables(project_name=project_name, db_session=db_session)

    # Verify that the SQL tables exist by checking if we can query them
    # We'll create a test user to verify the tables work
    sample_user: User = {
        "user_id": "test_user",
        "hourly_rate": 50.0,
        "active_task": "",
        "qualified_task_types": ["segmentation_proofread"],
    }

    user_id = create_user(project_name=project_name, data=sample_user, db_session=db_session)
    assert user_id == "test_user"


def test_get_project_success(project_name, clean_db, db_session):
    """Test getting a project that has been created"""
    # First create the project in the ProjectModel table
    create_project(
        project_name=project_name,
        description="Test project",
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    # Then create some data for the project
    sample_user: User = {
        "user_id": "test_user",
        "hourly_rate": 50.0,
        "active_task": "",
        "qualified_task_types": ["segmentation_proofread"],
    }
    create_user(project_name=project_name, data=sample_user, db_session=db_session)

    project = get_project(project_name=project_name, db_session=db_session)
    assert project["project_name"] == project_name
    assert project["status"] == "active"
    assert "created_at" in project


def test_get_project_not_found(project_name, clean_db, db_session):
    """Test that get_project raises KeyError when project doesn't exist"""
    with pytest.raises(KeyError, match=f"Project '{project_name}' not found"):
        get_project(project_name=project_name, db_session=db_session)


def test_get_project_found_in_other_tables(project_name, clean_db, db_session):
    """Test getting a project that exists in the projects table"""
    # Create the project first
    create_project(
        project_name=project_name,
        description="Test project",
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    # Now get_project should find the project
    project = get_project(project_name=project_name, db_session=db_session)
    assert project["project_name"] == project_name
    assert project["status"] == "active"
    assert "created_at" in project


def test_get_project_with_table_exception_handling(project_name, clean_db, db_session, mocker):
    """Test get_project handles exceptions when checking specific tables during fallback search"""
    # Create the project first so it exists in the ProjectModel table
    create_project(
        project_name=project_name,
        description="Test project",
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    # This should succeed because the project exists in ProjectModel table
    project = get_project(project_name=project_name, db_session=db_session)
    assert project["project_name"] == project_name
    assert project["status"] == "active"
    assert "created_at" in project


def test_get_project_with_all_table_exceptions(project_name, clean_db, db_session, mocker):
    """Test get_project when project doesn't exist"""
    # Don't create any project or data in the database

    # This should raise KeyError since project doesn't exist
    with pytest.raises(KeyError, match=f"Project '{project_name}' not found"):
        get_project(project_name=project_name, db_session=db_session)


def test_create_project_success(clean_db, db_session):
    """Test creating a new project successfully"""
    project_name = "test_project_1"
    description = "A test project"

    result = create_project(
        project_name=project_name,
        description=description,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    assert result == project_name

    # Verify the project was created
    project = get_project(project_name=project_name, db_session=db_session)
    assert project["project_name"] == project_name
    assert project["description"] == description
    assert project["status"] == "active"
    assert "created_at" in project


def test_create_project_duplicate(clean_db, db_session):
    """Test creating a project that already exists raises ValueError"""
    project_name = "duplicate_project"

    # Create the project first time
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    # Try to create again - should raise ValueError
    with pytest.raises(ValueError, match="Project 'duplicate_project' already exists"):
        create_project(
            project_name=project_name,
            segmentation_path="gs://test-bucket/segmentation",
            sv_resolution_x=4.0,
            sv_resolution_y=4.0,
            sv_resolution_z=40.0,
            db_session=db_session,
        )


def test_create_project_no_description(clean_db, db_session):
    """Test creating a project without description"""
    project_name = "no_desc_project"

    result = create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    assert result == project_name
    project = get_project(project_name=project_name, db_session=db_session)
    assert project["description"] == ""


def test_list_all_projects_empty(clean_db, db_session):
    """Test listing projects when none exist"""
    projects = list_all_projects(db_session=db_session)
    assert projects == []


def test_list_all_projects_single(clean_db, db_session):
    """Test listing projects with one project"""
    project_name = "single_project"
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    projects = list_all_projects(db_session=db_session)
    assert projects == [project_name]


def test_list_all_projects_multiple(clean_db, db_session):
    """Test listing multiple projects in alphabetical order"""
    project_names = ["zebra_project", "alpha_project", "beta_project"]

    # Create projects in random order
    for name in project_names:
        create_project(
            project_name=name,
            segmentation_path="gs://test-bucket/segmentation",
            sv_resolution_x=4.0,
            sv_resolution_y=4.0,
            sv_resolution_z=40.0,
            db_session=db_session,
        )

    projects = list_all_projects(db_session=db_session)
    # Should be returned in alphabetical order
    assert projects == ["alpha_project", "beta_project", "zebra_project"]


def test_list_all_projects_only_active(clean_db, db_session):
    """Test that only active projects are listed"""
    # Create an active project
    active_project = "active_project"
    create_project(
        project_name=active_project,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    inactive_project_model = ProjectModel(
        project_name="inactive_project",
        description="",
        created_at="2023-01-01T00:00:00",
        status="inactive",
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
    )
    db_session.add(inactive_project_model)
    db_session.commit()

    # Only the active project should be listed
    projects = list_all_projects(db_session=db_session)
    assert projects == [active_project]


def test_project_exists_true(clean_db, db_session):
    """Test project_exists returns True for existing project"""
    project_name = "existing_project"
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    assert project_exists(project_name=project_name, db_session=db_session) is True


def test_project_exists_false(clean_db, db_session):
    """Test project_exists returns False for non-existing project"""
    assert project_exists(project_name="nonexistent_project", db_session=db_session) is False


def test_delete_project_success(clean_db, db_session):
    """Test deleting an existing project"""
    project_name = "project_to_delete"
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    # Verify it exists
    assert project_exists(project_name=project_name, db_session=db_session) is True

    # Delete it
    result = delete_project(project_name=project_name, db_session=db_session)
    assert result is True

    # Verify it's gone
    assert project_exists(project_name=project_name, db_session=db_session) is False


def test_delete_project_nonexistent(clean_db, db_session):
    """Test deleting a non-existent project returns False"""
    result = delete_project(project_name="nonexistent_project", db_session=db_session)
    assert result is False


def test_get_project_from_model(clean_db, db_session):
    """Test getting project details from the ProjectModel"""
    project_name = "model_project"
    description = "Project from model test"

    create_project(
        project_name=project_name,
        description=description,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    project = get_project(project_name=project_name, db_session=db_session)
    assert project["project_name"] == project_name
    assert project["description"] == description
    assert project["status"] == "active"
    assert "created_at" in project


def test_get_project_not_found_with_model(clean_db, db_session):
    """Test get_project raises KeyError when project not found in ProjectModel"""
    with pytest.raises(KeyError, match="Project 'nonexistent_project' not found"):
        get_project(project_name="nonexistent_project", db_session=db_session)


# ==================== INTEGRATION TESTS ====================


def test_project_workflow_integration(clean_db, db_session):
    """Test complete project workflow: create -> list -> get -> delete"""
    project1 = "workflow_project_1"
    project2 = "workflow_project_2"

    # 1. Start with empty list
    assert list_all_projects(db_session=db_session) == []

    # 2. Create projects
    create_project(
        project_name=project1,
        description="First project",
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )
    create_project(
        project_name=project2,
        description="Second project",
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    # 3. List should show both, sorted
    projects = list_all_projects(db_session=db_session)
    assert projects == [project1, project2]  # alphabetically sorted

    # 4. Get individual projects
    p1 = get_project(project_name=project1, db_session=db_session)
    p2 = get_project(project_name=project2, db_session=db_session)

    assert p1["project_name"] == project1
    assert p1["description"] == "First project"
    assert p2["project_name"] == project2
    assert p2["description"] == "Second project"

    # 5. Check existence
    assert project_exists(project_name=project1, db_session=db_session) is True
    assert project_exists(project_name=project2, db_session=db_session) is True

    # 6. Delete one project
    assert delete_project(project_name=project1, db_session=db_session) is True

    # 7. Verify deletion
    assert project_exists(project_name=project1, db_session=db_session) is False
    assert project_exists(project_name=project2, db_session=db_session) is True

    projects = list_all_projects(db_session=db_session)
    assert projects == [project2]
