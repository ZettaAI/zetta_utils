"""Tests for TaskFeedbackModel"""

from datetime import datetime, timezone

from zetta_utils.task_management.db.models import TaskFeedbackModel


def test_task_feedback_model_to_dict():
    """Test TaskFeedbackModel.to_dict() method"""
    now = datetime.now(timezone.utc)

    feedback = TaskFeedbackModel(
        project_name="test_project",
        feedback_id=123,
        task_id="task_001",
        feedback_task_id="feedback_001",
        created_at=now,
        user_id="test_user@zetta.ai",
    )

    result = feedback.to_dict()

    assert result["feedback_id"] == 123
    assert result["task_id"] == "task_001"
    assert result["feedback_task_id"] == "feedback_001"
    assert result["created_at"] == now.isoformat()
    assert result["user_id"] == "test_user@zetta.ai"


def test_task_feedback_model_to_dict_none_timestamp():
    """Test TaskFeedbackModel.to_dict() with None timestamp"""
    feedback = TaskFeedbackModel(
        project_name="test_project",
        feedback_id=456,
        task_id="task_002",
        feedback_task_id="feedback_002",
        created_at=None,
        user_id="another_user@zetta.ai",
    )

    result = feedback.to_dict()

    assert result["feedback_id"] == 456
    assert result["task_id"] == "task_002"
    assert result["feedback_task_id"] == "feedback_002"
    assert result["created_at"] is None
    assert result["user_id"] == "another_user@zetta.ai"


def test_task_feedback_model_database_fields():
    """Test TaskFeedbackModel has correct database fields"""
    feedback = TaskFeedbackModel(
        project_name="test_project",
        feedback_id=789,
        task_id="task_003",
        feedback_task_id="feedback_003",
        created_at=datetime.now(timezone.utc),
        user_id="database_user@zetta.ai",
    )

    # Check that all required fields are present
    assert feedback.project_name == "test_project"
    assert feedback.feedback_id == 789
    assert feedback.task_id == "task_003"
    assert feedback.feedback_task_id == "feedback_003"
    assert feedback.created_at is not None
    assert feedback.user_id == "database_user@zetta.ai"


def test_task_feedback_model_user_id_field():
    """Test that user_id field is properly handled"""
    feedback = TaskFeedbackModel(
        project_name="test_project",
        feedback_id=101,
        task_id="task_004",
        feedback_task_id="feedback_004",
        created_at=datetime.now(timezone.utc),
        user_id="user_test@zetta.ai",
    )

    # Test that user_id is stored and retrieved correctly
    assert feedback.user_id == "user_test@zetta.ai"
    # Test that it appears in the dict representation
    result = feedback.to_dict()
    assert "user_id" in result
    assert result["user_id"] == "user_test@zetta.ai"
