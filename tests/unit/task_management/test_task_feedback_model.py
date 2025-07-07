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
    )

    result = feedback.to_dict()

    assert result["feedback_id"] == 123
    assert result["task_id"] == "task_001"
    assert result["feedback_task_id"] == "feedback_001"
    assert result["created_at"] == now.isoformat()


def test_task_feedback_model_to_dict_none_timestamp():
    """Test TaskFeedbackModel.to_dict() with None timestamp"""
    feedback = TaskFeedbackModel(
        project_name="test_project",
        feedback_id=456,
        task_id="task_002",
        feedback_task_id="feedback_002",
        created_at=None,
    )

    result = feedback.to_dict()

    assert result["feedback_id"] == 456
    assert result["task_id"] == "task_002"
    assert result["feedback_task_id"] == "feedback_002"
    assert result["created_at"] is None
