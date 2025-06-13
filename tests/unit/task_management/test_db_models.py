# pylint: disable=redefined-outer-name,unused-argument

from zetta_utils.task_management.db.models import (
    DependencyModel,
    JobModel,
    SubtaskModel,
    SubtaskTypeModel,
    TimesheetModel,
    UserModel,
)


def test_subtask_type_model_with_description():
    """Test SubtaskTypeModel with description"""
    data = {
        "subtask_type": "test_type",
        "completion_statuses": ["pending", "completed"],
        "description": "Test description",
    }

    model = SubtaskTypeModel.from_dict("test_project", data)
    result_dict = model.to_dict()

    assert "description" in result_dict
    assert result_dict["description"] == "Test description"


def test_subtask_type_model_without_description():
    """Test SubtaskTypeModel without description"""
    data = {"subtask_type": "test_type", "completion_statuses": ["pending", "completed"]}

    model = SubtaskTypeModel.from_dict("test_project", data)
    result_dict = model.to_dict()

    assert "description" not in result_dict


def test_job_model_with_id_nonunique():
    """Test JobModel with id_nonunique field"""
    data = {
        "job_id": "test_job",
        "batch_id": "batch_1",
        "status": "pending",
        "job_type": "segmentation",
        "ng_state": "http://example.com",
        "id_nonunique": 1,
    }

    model = JobModel.from_dict("test_project", data)
    assert model.id_nonunique == 1


def test_subtask_model_with_id_nonunique():
    """Test SubtaskModel with id_nonunique field"""
    data = {
        "subtask_id": "test_subtask",
        "job_id": "test_job",
        "ng_state": "http://example.com",
        "ng_state_initial": "http://example.com/initial",
        "priority": 1,
        "batch_id": "batch_1",
        "subtask_type": "segmentation",
        "id_nonunique": 1,
    }

    model = SubtaskModel.from_dict("test_project", data)
    assert model.id_nonunique == 1


def test_dependency_model_with_is_satisfied():
    """Test DependencyModel with is_satisfied field"""
    data = {
        "dependency_id": "dep_1",
        "subtask_id": "subtask_1",
        "dependent_on_subtask_id": "subtask_2",
        "required_completion_status": "completed",
        "is_satisfied": True,
    }

    model = DependencyModel.from_dict("test_project", data)
    assert model.is_satisfied is True


def test_all_models_to_dict():
    """Test that all models can convert to dict"""
    # Test UserModel
    user_data = {
        "user_id": "test_user",
        "hourly_rate": 50.0,
        "active_subtask": "",
        "qualified_subtask_types": ["type1"],
    }
    user_model = UserModel.from_dict("test_project", user_data)
    user_dict = user_model.to_dict()
    assert user_dict["user_id"] == "test_user"

    # Test TimesheetModel
    timesheet_data = {
        "entry_id": "entry_1",
        "job_id": "job_1",
        "subtask_id": "subtask_1",
        "user": "test_user",
        "seconds_spent": 3600,
    }
    timesheet_model = TimesheetModel.from_dict("test_project", timesheet_data)
    timesheet_dict = timesheet_model.to_dict()
    assert timesheet_dict["entry_id"] == "entry_1"
