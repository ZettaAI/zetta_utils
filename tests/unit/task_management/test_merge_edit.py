"""Tests for merge edit functionality"""

# pylint: disable=unused-argument,redefined-outer-name

import pytest
from datetime import datetime

from zetta_utils.task_management.merge_edit import (
    create_merge_edit,
    get_merge_edits_by_task,
    get_merge_edits_by_user,
    get_merge_edit_by_id,
)
from zetta_utils.task_management.project import create_project
from zetta_utils.task_management.task import create_task
from zetta_utils.task_management.task_type import create_task_type
from zetta_utils.task_management.user import create_user
from zetta_utils.task_management.types import Task, TaskType, User


@pytest.fixture
def test_project(clean_db, db_session):
    """Create test project"""
    project_name = "test_project"
    create_project(
        project_name=project_name,
        segmentation_path="gs://test/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=42.0,
        db_session=db_session,
    )
    return project_name


@pytest.fixture
def test_user(clean_db, db_session, test_project):
    """Create test user"""
    user_data: User = {
        "user_id": "test_user",
        "hourly_rate": 50.0,
        "active_task": "",
        "qualified_task_types": ["trace_v0"],
    }
    create_user(project_name=test_project, data=user_data, db_session=db_session)
    return user_data


@pytest.fixture
def test_task_type(clean_db, db_session, test_project):
    """Create test task type"""
    task_type_data: TaskType = {
        "task_type": "trace_v0",
        "completion_statuses": ["Done", "Can't Continue"],
    }
    create_task_type(project_name=test_project, data=task_type_data, db_session=db_session)
    return task_type_data


@pytest.fixture
def test_task(clean_db, db_session, test_project, test_task_type):
    """Create test task"""
    task_data: Task = {
        "task_id": "test_task",
        "completion_status": "",
        "assigned_user_id": "",
        "active_user_id": "",
        "completed_user_id": "",
        "ng_state": {"url": "http://example.com"},
        "ng_state_initial": {"url": "http://example.com"},
        "priority": 1,
        "batch_id": "batch_1",
        "task_type": "trace_v0",
        "is_active": True,
        "last_leased_ts": 0.0,
    }
    create_task(project_name=test_project, data=task_data, db_session=db_session)
    return task_data


@pytest.fixture
def sample_merge_data():
    """Sample merge edit data"""
    return {
        "points": [
            ["74450544596519719", 300951.5, 72310.875, 151809],
            ["74450544596525136", 300647.15625, 72035.0625, 151809],
        ],
    }


class TestMergeEdit:
    """Test cases for merge edit functionality"""

    def test_create_merge_edit_success(
        self, test_project, test_user, test_task, sample_merge_data, db_session
    ):
        """Test successful creation of merge edit"""
        edit_id = create_merge_edit(
            project_name=test_project,
            task_id=test_task["task_id"],
            user_id=test_user["user_id"],
            points=sample_merge_data["points"],
            db_session=db_session,
        )
        
        assert isinstance(edit_id, int)
        assert edit_id > 0

    def test_create_merge_edit_returns_different_ids(
        self, test_project, test_user, test_task, sample_merge_data, db_session
    ):
        """Test that multiple merge edits get different IDs"""
        edit_id_1 = create_merge_edit(
            project_name=test_project,
            task_id=test_task["task_id"],
            user_id=test_user["user_id"],
            points=sample_merge_data["points"],
            db_session=db_session,
        )
        
        edit_id_2 = create_merge_edit(
            project_name=test_project,
            task_id=test_task["task_id"],
            user_id=test_user["user_id"],
            points=sample_merge_data["points"],
            db_session=db_session,
        )
        
        assert edit_id_1 != edit_id_2

    def test_get_merge_edits_by_task(
        self, test_project, test_user, test_task, sample_merge_data, db_session
    ):
        """Test getting merge edits by task ID"""
        # Create merge edit
        edit_id = create_merge_edit(
            project_name=test_project,
            task_id=test_task["task_id"],
            user_id=test_user["user_id"],
            points=sample_merge_data["points"],
            db_session=db_session,
        )
        
        # Get by task ID
        edits = get_merge_edits_by_task(
            project_name=test_project,
            task_id=test_task["task_id"],
            db_session=db_session,
        )
        
        assert len(edits) == 1
        assert edits[0]["edit_id"] == edit_id
        assert edits[0]["task_id"] == test_task["task_id"]
        assert edits[0]["user_id"] == test_user["user_id"]
        assert edits[0]["points"] == sample_merge_data["points"]
        assert "created_at" in edits[0]

    def test_get_merge_edits_by_user(
        self, test_project, test_user, test_task, sample_merge_data, db_session
    ):
        """Test getting merge edits by user ID"""
        # Create merge edit
        edit_id = create_merge_edit(
            project_name=test_project,
            task_id=test_task["task_id"],
            user_id=test_user["user_id"],
            points=sample_merge_data["points"],
            db_session=db_session,
        )
        
        # Get by user ID
        edits = get_merge_edits_by_user(
            project_name=test_project,
            user_id=test_user["user_id"],
            db_session=db_session,
        )
        
        assert len(edits) == 1
        assert edits[0]["edit_id"] == edit_id
        assert edits[0]["user_id"] == test_user["user_id"]

    def test_get_merge_edit_by_id(
        self, test_project, test_user, test_task, sample_merge_data, db_session
    ):
        """Test getting merge edit by ID"""
        # Create merge edit
        edit_id = create_merge_edit(
            project_name=test_project,
            task_id=test_task["task_id"],
            user_id=test_user["user_id"],
            points=sample_merge_data["points"],
            db_session=db_session,
        )
        
        # Get by ID
        edit = get_merge_edit_by_id(
            project_name=test_project,
            edit_id=edit_id,
            db_session=db_session,
        )
        
        assert edit is not None
        assert edit["edit_id"] == edit_id
        assert edit["task_id"] == test_task["task_id"]
        assert edit["user_id"] == test_user["user_id"]
        assert edit["points"] == sample_merge_data["points"]

    def test_get_merge_edit_by_id_not_found(
        self, test_project, db_session
    ):
        """Test getting non-existent merge edit by ID"""
        edit = get_merge_edit_by_id(
            project_name=test_project,
            edit_id=999,
            db_session=db_session,
        )
        
        assert edit is None

    def test_get_merge_edits_empty_results(
        self, test_project, db_session
    ):
        """Test getting merge edits when none exist"""
        # Get by task ID
        edits_by_task = get_merge_edits_by_task(
            project_name=test_project,
            task_id="nonexistent_task",
            db_session=db_session,
        )
        assert len(edits_by_task) == 0
        
        # Get by user ID
        edits_by_user = get_merge_edits_by_user(
            project_name=test_project,
            user_id="nonexistent_user",
            db_session=db_session,
        )
        assert len(edits_by_user) == 0

    def test_merge_edits_ordering(
        self, test_project, test_user, test_task, sample_merge_data, db_session
    ):
        """Test that merge edits are returned in correct order (newest first)"""
        # Create first merge edit
        edit_id_1 = create_merge_edit(
            project_name=test_project,
            task_id=test_task["task_id"],
            user_id=test_user["user_id"],
            points=sample_merge_data["points"],
            db_session=db_session,
        )
        
        # Create second merge edit
        edit_id_2 = create_merge_edit(
            project_name=test_project,
            task_id=test_task["task_id"],
            user_id=test_user["user_id"],
            points=sample_merge_data["points"],
            db_session=db_session,
        )
        
        # Get by task ID
        edits = get_merge_edits_by_task(
            project_name=test_project,
            task_id=test_task["task_id"],
            db_session=db_session,
        )
        
        assert len(edits) == 2
        # Should be ordered by created_at DESC (newest first)
        assert edits[0]["edit_id"] == edit_id_2
        assert edits[1]["edit_id"] == edit_id_1

    def test_merge_edits_different_projects(
        self, test_user, test_task, sample_merge_data, db_session
    ):
        """Test that merge edits are properly isolated by project"""
        # Create two projects
        project_1 = "project_1"
        project_2 = "project_2"
        
        for project in [project_1, project_2]:
            create_project(
                project_name=project,
                segmentation_path="gs://test/segmentation",
                sv_resolution_x=4.0,
                sv_resolution_y=4.0,
                sv_resolution_z=42.0,
                db_session=db_session,
            )
        
        # Create merge edit in project 1
        edit_id_1 = create_merge_edit(
            project_name=project_1,
            task_id=test_task["task_id"],
            user_id=test_user["user_id"],
            points=sample_merge_data["points"],
            db_session=db_session,
        )
        
        # Create merge edit in project 2
        edit_id_2 = create_merge_edit(
            project_name=project_2,
            task_id=test_task["task_id"],
            user_id=test_user["user_id"],
            points=sample_merge_data["points"],
            db_session=db_session,
        )
        
        # Get edits for project 1
        edits_1 = get_merge_edits_by_task(
            project_name=project_1,
            task_id=test_task["task_id"],
            db_session=db_session,
        )
        assert len(edits_1) == 1
        assert edits_1[0]["edit_id"] == edit_id_1
        
        # Get edits for project 2
        edits_2 = get_merge_edits_by_task(
            project_name=project_2,
            task_id=test_task["task_id"],
            db_session=db_session,
        )
        assert len(edits_2) == 1
        assert edits_2[0]["edit_id"] == edit_id_2