"""Tests for task_management db models"""

# pylint: disable=unused-argument,redefined-outer-name

from datetime import datetime, timezone

import pytest

from zetta_utils.task_management.db.models import (
    DependencyModel,
    EndpointModel,
    EndpointUpdateModel,
    MergeEditModel,
    ProjectModel,
    SegmentModel,
    SegmentTypeModel,
    SplitEditModel,
    TaskModel,
    TaskTypeModel,
    TimesheetModel,
    TimesheetSubmissionModel,
    UserModel,
    _parse_datetime,
)

# Test _parse_datetime utility function


def test_parse_datetime_from_datetime():
    """Test parsing when input is already datetime"""
    dt = datetime.now(timezone.utc)
    result = _parse_datetime(dt)
    assert result == dt


def test_parse_datetime_from_string():
    """Test parsing from ISO format string"""
    dt_str = "2025-07-03T12:00:00+00:00"
    result = _parse_datetime(dt_str)
    assert isinstance(result, datetime)
    assert result.year == 2025
    assert result.month == 7
    assert result.day == 3


def test_parse_datetime_from_timestamp():
    """Test parsing from Unix timestamp"""
    timestamp = 1720008000  # 2025-07-03 12:00:00 UTC
    result = _parse_datetime(timestamp)
    assert isinstance(result, datetime)


def test_parse_datetime_from_float_timestamp():
    """Test parsing from float Unix timestamp"""
    timestamp = 1720008000.123
    result = _parse_datetime(timestamp)
    assert isinstance(result, datetime)


def test_parse_datetime_invalid_type():
    """Test parsing with invalid type raises ValueError"""
    with pytest.raises(ValueError, match="Cannot parse datetime"):
        _parse_datetime([1, 2, 3])  # type: ignore[arg-type]


# Test ProjectModel


def test_project_model_to_dict_minimal(db_session):
    """Test to_dict with minimal fields"""
    project = ProjectModel(
        project_name="test_project",
        segmentation_path="precomputed://test",
        sv_resolution_x=8.0,
        sv_resolution_y=8.0,
        sv_resolution_z=40.0,
    )
    db_session.add(project)
    db_session.commit()

    result = project.to_dict()
    assert result["project_name"] == "test_project"
    assert result["segmentation_path"] == "precomputed://test"
    assert result["sv_resolution_x"] == 8.0
    assert result["sv_resolution_y"] == 8.0
    assert result["sv_resolution_z"] == 40.0
    assert result["status"] == "active"
    assert result["created_at"] is None
    assert result["description"] is None
    # Optional fields should not be in dict
    assert "brain_mesh_path" not in result
    assert "datastack_name" not in result
    assert "synapse_table" not in result
    assert "extra_layers" not in result


def test_project_model_to_dict_full(db_session):
    """Test to_dict with all fields"""
    project = ProjectModel(
        project_name="full_project",
        created_at="2025-07-03T12:00:00",
        description="Test project",
        status="archived",
        segmentation_path="precomputed://test",
        brain_mesh_path="mesh://brain",
        datastack_name="test_stack",
        synapse_table="synapses_v1",
        sv_resolution_x=8.0,
        sv_resolution_y=8.0,
        sv_resolution_z=40.0,
        extra_layers={"layer1": {"type": "image"}},
    )
    db_session.add(project)
    db_session.commit()

    result = project.to_dict()
    assert result["project_name"] == "full_project"
    assert result["created_at"] == "2025-07-03T12:00:00"
    assert result["description"] == "Test project"
    assert result["status"] == "archived"
    assert result["brain_mesh_path"] == "mesh://brain"
    assert result["datastack_name"] == "test_stack"
    assert result["synapse_table"] == "synapses_v1"
    assert result["extra_layers"] == {"layer1": {"type": "image"}}


# TestTaskTypeModel tests


def test_to_dict_with_description_task_type_model(db_session):
    """Test to_dict with description"""
    task_type = TaskTypeModel(
        project_name="test_project",
        task_type="verify",
        completion_statuses=["Verified", "Failed"],
        description="Verification task",
    )
    db_session.add(task_type)
    db_session.commit()

    result = task_type.to_dict()
    assert result["description"] == "Verification task"


def test_from_dict_task_type_model():
    """Test creating from dict"""
    data = {
        "task_type": "custom",
        "completion_statuses": ["A", "B", "C"],
        "description": "Custom task",
    }
    model = TaskTypeModel.from_dict("test_project", data)
    assert model.project_name == "test_project"
    assert model.task_type == "custom"
    assert model.completion_statuses == ["A", "B", "C"]
    assert model.description == "Custom task"


def test_from_dict_no_description_task_type_model():
    """Test creating from dict without description"""
    data = {
        "task_type": "test_type",
        "completion_statuses": ["pending", "completed"],
    }
    model = TaskTypeModel.from_dict("test_project", data)
    assert model.description is None


# TestUserModel tests


def test_from_dict_user_model():
    """Test creating from dict"""
    data = {
        "user_id": "test_user",
        "hourly_rate": 30.0,
        "active_task": "",
        "qualified_task_types": ["type1", "type2"],
    }
    model = UserModel.from_dict("test_project", data)
    assert model.project_name == "test_project"
    assert model.user_id == "test_user"
    assert model.hourly_rate == 30.0
    assert model.active_task == ""
    assert model.qualified_task_types == ["type1", "type2"]


# TestDependencyModel tests


def test_from_dict_with_is_satisfied_dependency_model():
    """Test creating from dict with is_satisfied"""
    data = {
        "dependency_id": "dep456",
        "task_id": "taskA",
        "dependent_on_task_id": "taskB",
        "required_completion_status": "Verified",
        "is_satisfied": True,
    }
    model = DependencyModel.from_dict("test_project", data)
    assert model.project_name == "test_project"
    assert model.dependency_id == "dep456"
    assert model.task_id == "taskA"
    assert model.dependent_on_task_id == "taskB"
    assert model.is_satisfied is True
    assert model.required_completion_status == "Verified"


def test_from_dict_without_is_satisfied_dependency_model():
    """Test creating from dict without is_satisfied (default False)"""
    data = {
        "dependency_id": "dep789",
        "task_id": "taskC",
        "dependent_on_task_id": "taskD",
        "required_completion_status": "Done",
    }
    model = DependencyModel.from_dict("test_project", data)
    assert model.is_satisfied is False  # Default value


# TestTimesheetModel tests


def test_from_dict_timesheet_model():
    """Test creating from dict"""
    data = {
        "entry_id": "entry456",
        "task_id": "taskXYZ",
        "user": "userABC",
        "seconds_spent": 7200,
    }
    model = TimesheetModel.from_dict("test_project", data)
    assert model.project_name == "test_project"
    assert model.entry_id == "entry456"
    assert model.task_id == "taskXYZ"
    assert model.user == "userABC"
    assert model.seconds_spent == 7200


# TestTimesheetSubmissionModel tests


def test_to_dict_none_submitted_at_timesheet_submission_model(db_session):
    """Test to_dict when submitted_at would be None (edge case)"""
    # This is a hypothetical edge case test - in practice submitted_at is required
    submission = TimesheetSubmissionModel(
        project_name="test_project",
        user_id="user123",
        task_id="task456",
        seconds_spent=1800,
        submitted_at=datetime.now(timezone.utc),
    )
    db_session.add(submission)
    db_session.commit()

    # Simulate None case for coverage
    original_timestamp = submission.submitted_at
    submission.submitted_at = None  # type: ignore[assignment]
    result = submission.to_dict()
    assert result["submitted_at"] is None

    # Restore for cleanup
    submission.submitted_at = original_timestamp


def test_from_dict_with_string_timestamp_timesheet_submission_model():
    """Test creating from dict with string timestamp"""
    data = {
        "user_id": "userXYZ",
        "task_id": "taskABC",
        "seconds_spent": 2400,
        "submitted_at": "2025-07-03T14:30:00+00:00",
    }
    model = TimesheetSubmissionModel.from_dict("test_project", data)
    assert model.project_name == "test_project"
    assert model.user_id == "userXYZ"
    assert model.task_id == "taskABC"
    assert model.seconds_spent == 2400
    assert isinstance(model.submitted_at, datetime)


def test_from_dict_with_int_timestamp_timesheet_submission_model():
    """Test creating from dict with integer timestamp"""
    data = {
        "user_id": "userABC",
        "task_id": "taskXYZ",
        "seconds_spent": 1200,
        "submitted_at": 1720008000,
    }
    model = TimesheetSubmissionModel.from_dict("test_project", data)
    assert isinstance(model.submitted_at, datetime)


# TestSegmentTypeModel tests


def test_to_dict_full_segment_type_model(db_session):
    """Test to_dict with all fields"""
    now = datetime.now(timezone.utc)
    segment_type = SegmentTypeModel(
        type_name="glia",
        project_name="test_project",
        sample_segment_ids=["123", "456", "789"],
        description="Glial cells",
        region_mesh="mesh://region",
        seed_mask="mask://seed",
        created_at=now,
        updated_at=now,
    )
    db_session.add(segment_type)
    db_session.commit()

    result = segment_type.to_dict()
    assert result["description"] == "Glial cells"
    assert result["sample_segment_ids"] == ["123", "456", "789"]
    assert result["region_mesh"] == "mesh://region"
    assert result["seed_mask"] == "mask://seed"


def test_from_dict_segment_type_model():
    """Test creating from dict"""
    data = {
        "type_name": "axon",
        "project_name": "test_project",
        "sample_segment_ids": ["111", "222"],
        "description": "Axon segments",
        "created_at": "2025-07-03T10:00:00+00:00",
        "updated_at": "2025-07-03T11:00:00+00:00",
    }
    model = SegmentTypeModel.from_dict(data)
    assert model.type_name == "axon"
    assert model.project_name == "test_project"
    assert model.sample_segment_ids == ["111", "222"]
    assert model.description == "Axon segments"
    assert isinstance(model.created_at, datetime)
    assert isinstance(model.updated_at, datetime)


def test_from_dict_defaults_segment_type_model():
    """Test creating from dict with defaults"""
    data = {
        "type_name": "dendrite",
        "project_name": "test_project",
        "created_at": "2025-07-03T10:00:00+00:00",
        "updated_at": "2025-07-03T11:00:00+00:00",
    }
    model = SegmentTypeModel.from_dict(data)
    assert model.sample_segment_ids == []
    assert model.description is None


# TestSegmentModel tests


def test_to_dict_full_segment_model(db_session):
    """Test to_dict with all fields"""
    now = datetime.now(timezone.utc)
    segment = SegmentModel(
        project_name="test_project",
        seed_id=12345,
        seed_x=100.0,
        seed_y=200.0,
        seed_z=300.0,
        root_x=150.0,
        root_y=250.0,
        root_z=350.0,
        task_ids=["task1", "task2"],
        segment_type="neuron",
        expected_segment_type="neuron",
        batch="batch_001",
        current_segment_id=67890,
        skeleton_path_length_mm=1.5,
        pre_synapse_count=10,
        post_synapse_count=20,
        status="Completed",
        is_exported=True,
        created_at=now,
        updated_at=now,
        extra_data={"key": "value"},
    )
    db_session.add(segment)
    db_session.commit()

    result = segment.to_dict()
    assert result["seed_id"] == 12345
    assert result["segment_type"] == "neuron"
    assert result["expected_segment_type"] == "neuron"
    assert result["batch"] == "batch_001"
    assert result["root_x"] == 150.0
    assert result["root_y"] == 250.0
    assert result["root_z"] == 350.0
    assert result["current_segment_id"] == 67890
    assert result["skeleton_path_length_mm"] == 1.5
    assert result["pre_synapse_count"] == 10
    assert result["post_synapse_count"] == 20
    assert result["extra_data"] == {"key": "value"}


def test_from_dict_segment_model():
    """Test creating from dict"""
    data = {
        "project_name": "test_project",
        "seed_id": 99999,
        "seed_x": 10.0,
        "seed_y": 20.0,
        "seed_z": 30.0,
        "root_x": 15.0,
        "root_y": 25.0,
        "root_z": 35.0,
        "task_ids": ["taskA", "taskB"],
        "segment_type": "glia",
        "expected_segment_type": "glia",
        "batch": "batch_002",
        "current_segment_id": 11111,
        "skeleton_path_length_mm": 2.5,
        "pre_synapse_count": 5,
        "post_synapse_count": 15,
        "status": "Retired",
        "is_exported": True,
        "created_at": "2025-07-03T09:00:00+00:00",
        "updated_at": "2025-07-03T10:00:00+00:00",
        "extra_data": {"info": "test"},
    }
    model = SegmentModel.from_dict(data)
    assert model.project_name == "test_project"
    assert model.seed_id == 99999
    assert model.segment_type == "glia"
    assert model.status == "Retired"
    assert model.is_exported is True
    assert model.extra_data == {"info": "test"}


def test_from_dict_defaults_segment_model():
    """Test creating from dict with defaults"""
    data = {
        "project_name": "test_project",
        "seed_x": 1.0,
        "seed_y": 2.0,
        "seed_z": 3.0,
        "created_at": "2025-07-03T09:00:00+00:00",
        "updated_at": "2025-07-03T10:00:00+00:00",
    }
    model = SegmentModel.from_dict(data)
    assert model.task_ids == []
    assert model.status == "wip"  # Note: lowercase in from_dict default
    assert model.is_exported is False
    assert model.segment_type is None


# TestEndpointModel tests


def test_from_dict_endpoint_model():
    """Test creating from dict"""
    data = {
        "seed_id": 67890,
        "x": 50.0,
        "y": 60.0,
        "z": 70.0,
        "status": "UNCERTAIN",
        "user": "userXYZ",
        "created_at": "2025-07-03T08:00:00+00:00",
        "updated_at": "2025-07-03T08:30:00+00:00",
    }
    model = EndpointModel.from_dict("test_project", data)
    assert model.project_name == "test_project"
    assert model.seed_id == 67890
    assert model.status == "UNCERTAIN"
    assert model.user == "userXYZ"


def test_to_dict_endpoint_model():
    """Test converting EndpointModel to dict"""
    model = EndpointModel(
        project_name="test_project",
        endpoint_id=123,
        seed_id=67890,
        x=50.0,
        y=60.0,
        z=70.0,
        status="UNCERTAIN",
        user="userXYZ",
        created_at=datetime(2025, 7, 3, 8, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2025, 7, 3, 8, 30, 0, tzinfo=timezone.utc),
    )

    result = model.to_dict()

    assert result == {
        "endpoint_id": 123,
        "seed_id": 67890,
        "x": 50.0,
        "y": 60.0,
        "z": 70.0,
        "status": "UNCERTAIN",
        "user": "userXYZ",
        "created_at": "2025-07-03T08:00:00+00:00",
        "updated_at": "2025-07-03T08:30:00+00:00",
    }


# TestEndpointUpdateModel tests


def test_from_dict_endpoint_update_model():
    """Test creating from dict"""
    data = {
        "endpoint_id": 456,
        "user": "userABC",
        "new_status": "BREADCRUMB",
        "timestamp": "2025-07-03T07:00:00+00:00",
    }
    model = EndpointUpdateModel.from_dict("test_project", data)
    assert model.project_name == "test_project"
    assert model.endpoint_id == 456
    assert model.user == "userABC"
    assert model.new_status == "BREADCRUMB"
    assert isinstance(model.timestamp, datetime)


def test_to_dict_endpoint_update_model():
    """Test converting EndpointUpdateModel to dict"""
    model = EndpointUpdateModel(
        project_name="test_project",
        update_id=789,
        endpoint_id=456,
        user="userABC",
        new_status="BREADCRUMB",
        timestamp=datetime(2025, 7, 3, 7, 0, 0, tzinfo=timezone.utc),
    )

    result = model.to_dict()

    assert result == {
        "update_id": 789,
        "endpoint_id": 456,
        "user": "userABC",
        "new_status": "BREADCRUMB",
        "timestamp": "2025-07-03T07:00:00+00:00",
    }


# TestTaskModel tests


def test_to_dict_full_task_model(db_session):
    """Test to_dict with all fields"""
    task = TaskModel(
        project_name="test_project",
        task_id="task456",
        completion_status="Done",
        assigned_user_id="user1",
        active_user_id="user2",
        completed_user_id="user3",
        ng_state={"layers": ["layer1"]},
        ng_state_initial={"layers": ["layer1"]},
        priority=100,
        batch_id="batch2",
        last_leased_ts=1720008000.0,
        is_active=False,
        is_paused=True,
        is_checked=True,
        task_type="verify",
        id_nonunique=67890,
        extra_data={"info": "test"},
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(task)
    db_session.commit()

    result = task.to_dict()
    assert result["completion_status"] == "Done"
    assert result["assigned_user_id"] == "user1"
    assert result["active_user_id"] == "user2"
    assert result["completed_user_id"] == "user3"
    assert result["is_active"] is False
    assert result["is_paused"] is True
    assert result["is_checked"] is True
    assert result["extra_data"] == {"info": "test"}


def test_from_dict_task_model():
    """Test creating from dict"""
    data = {
        "task_id": "task789",
        "completion_status": "Faulty",
        "assigned_user_id": "userA",
        "active_user_id": "userB",
        "completed_user_id": "userC",
        "ng_state": {"test": "state"},
        "ng_state_initial": {"test": "initial"},
        "priority": 75,
        "batch_id": "batch3",
        "last_leased_ts": 1720009000.0,
        "is_active": True,
        "is_paused": False,
        "is_checked": True,
        "task_type": "custom",
        "id_nonunique": 11111,
        "extra_data": {"key": "value"},
    }
    model = TaskModel.from_dict("test_project", data)
    assert model.project_name == "test_project"
    assert model.task_id == "task789"
    assert model.completion_status == "Faulty"
    assert model.assigned_user_id == "userA"
    assert model.priority == 75
    assert model.extra_data == {"key": "value"}


def test_from_dict_defaults_task_model():
    """Test creating from dict with defaults"""
    data = {
        "task_id": "task999",
        "ng_state": {"test": "state"},
        "ng_state_initial": {"test": "initial"},
        "priority": 50,
        "batch_id": "batch",
        "task_type": "test",
    }
    model = TaskModel.from_dict("test_project", data)
    assert model.completion_status == ""
    assert model.assigned_user_id == ""
    assert model.active_user_id == ""
    assert model.completed_user_id == ""
    assert model.last_leased_ts == 0.0
    assert model.is_active is True
    assert model.is_paused is False
    assert model.is_checked is False
    assert model.extra_data is None


# Keep original backward compatibility tests
def test_task_type_model_with_description():
    """Test TaskTypeModel with description (backward compatibility)"""
    data = {
        "task_type": "test_type",
        "completion_statuses": ["pending", "completed"],
        "description": "Test description",
    }
    model = TaskTypeModel.from_dict("test_project", data)
    result_dict = model.to_dict()
    assert "description" in result_dict
    assert result_dict["description"] == "Test description"


def test_task_type_model_without_description():
    """Test TaskTypeModel without description (backward compatibility)"""
    data = {"task_type": "test_type", "completion_statuses": ["pending", "completed"]}
    model = TaskTypeModel.from_dict("test_project", data)
    result_dict = model.to_dict()
    assert "description" not in result_dict


def test_task_model_with_id_nonunique():
    """Test TaskModel with id_nonunique field (backward compatibility)"""
    data = {
        "task_id": "test_task",
        "job_id": "test_job",
        "ng_state": {"url": "http://example.com"},
        "ng_state_initial": {"url": "http://example.com/initial"},
        "priority": 1,
        "batch_id": "batch_1",
        "task_type": "segmentation",
        "id_nonunique": 1,
    }
    model = TaskModel.from_dict("test_project", data)
    assert model.id_nonunique == 1


def test_dependency_model_with_is_satisfied():
    """Test DependencyModel with is_satisfied field (backward compatibility)"""
    data = {
        "dependency_id": "dep_1",
        "task_id": "task_1",
        "dependent_on_task_id": "task_2",
        "required_completion_status": "completed",
        "is_satisfied": True,
    }
    model = DependencyModel.from_dict("test_project", data)
    assert model.is_satisfied is True


def test_all_models_to_dict():
    """Test that all models can convert to dict (backward compatibility)"""
    # Test UserModel
    user_data = {
        "user_id": "test_user",
        "hourly_rate": 50.0,
        "active_task": "",
        "qualified_task_types": ["type1"],
    }
    user_model = UserModel.from_dict("test_project", user_data)
    user_dict = user_model.to_dict()
    assert user_dict["user_id"] == "test_user"

    # Test TimesheetModel
    timesheet_data = {
        "entry_id": "entry_1",
        "task_id": "task_1",
        "user": "test_user",
        "seconds_spent": 3600,
    }
    timesheet_model = TimesheetModel.from_dict("test_project", timesheet_data)
    timesheet_dict = timesheet_model.to_dict()
    assert timesheet_dict["entry_id"] == "entry_1"


def test_task_model_note_field(clean_db, db_session):
    """Test TaskModel.to_dict() with note field to cover line 653"""
    # Simplest test - just create a task with note and call to_dict()
    task = TaskModel(
        project_name="test",
        task_id="t1",
        ng_state={},
        ng_state_initial={},
        priority=1,
        batch_id="b1",
        task_type="test",
        id_nonunique=1,
        note="test note",  # This will trigger line 653
        created_at=datetime.now(timezone.utc),
    )

    result = task.to_dict()
    assert result["note"] == "test note"


# TestSplitEditModel tests


def test_split_edit_model_from_dict():
    """Test SplitEditModel.from_dict() to cover line 770"""
    data = {
        "task_id": "task123",
        "user_id": "user456",
        "sources": [{"segment_id": 123, "x": 100, "y": 200, "z": 300}],
        "sinks": [{"segment_id": 456, "x": 400, "y": 500, "z": 600}],
        "created_at": "2025-07-03T10:00:00+00:00",
    }
    model = SplitEditModel.from_dict("test_project", data)
    assert model.project_name == "test_project"
    assert model.task_id == "task123"
    assert model.user_id == "user456"
    assert model.sources == [{"segment_id": 123, "x": 100, "y": 200, "z": 300}]
    assert model.sinks == [{"segment_id": 456, "x": 400, "y": 500, "z": 600}]
    assert isinstance(model.created_at, datetime)


def test_split_edit_model_from_dict_default_timestamp():
    """Test SplitEditModel.from_dict() with default timestamp"""
    data = {
        "task_id": "task789",
        "user_id": "user101",
        "sources": [],
        "sinks": [],
    }
    model = SplitEditModel.from_dict("test_project", data)
    assert model.project_name == "test_project"
    assert model.task_id == "task789"
    assert model.user_id == "user101"
    assert isinstance(model.created_at, datetime)


# TestMergeEditModel tests


def test_merge_edit_model_from_dict():
    """Test MergeEditModel.from_dict() to cover line 829"""
    data = {
        "task_id": "task456",
        "user_id": "user789",
        "points": [
            {"segment_id": 111, "x": 10, "y": 20, "z": 30},
            {"segment_id": 222, "x": 40, "y": 50, "z": 60},
        ],
        "created_at": "2025-07-03T12:00:00+00:00",
    }
    model = MergeEditModel.from_dict("test_project", data)
    assert model.project_name == "test_project"
    assert model.task_id == "task456"
    assert model.user_id == "user789"
    assert model.points == [
        {"segment_id": 111, "x": 10, "y": 20, "z": 30},
        {"segment_id": 222, "x": 40, "y": 50, "z": 60},
    ]
    assert isinstance(model.created_at, datetime)


def test_merge_edit_model_from_dict_default_timestamp():
    """Test MergeEditModel.from_dict() with default timestamp"""
    data = {
        "task_id": "task999",
        "user_id": "user333",
        "points": [],
    }
    model = MergeEditModel.from_dict("test_project", data)
    assert model.project_name == "test_project"
    assert model.task_id == "task999"
    assert model.user_id == "user333"
    assert isinstance(model.created_at, datetime)
