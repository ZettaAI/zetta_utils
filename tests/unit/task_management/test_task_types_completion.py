# pylint: disable=redefined-outer-name,unused-argument
from unittest.mock import MagicMock

import pytest

from zetta_utils.task_management.task_types.completion import (
    _COMPLETION_HANDLERS,
    handle_task_completion,
    register_completion_handler,
)
from zetta_utils.task_management.types import Task


def test_register_completion_handler():
    """Test registering a completion handler"""

    @register_completion_handler("test_type")
    def test_handler(project_name: str, task: Task, completion_status: str) -> None:
        pass

    assert _COMPLETION_HANDLERS["test_type"] is test_handler


def test_handle_task_completion_with_handler():
    """Test handling task completion with a registered handler"""
    # Create a mock handler
    mock_handler = MagicMock()

    # Register the handler directly in the registry
    _COMPLETION_HANDLERS["test_task_type"] = mock_handler

    # Create a test task
    task = Task(
        task_id="test_task",
        task_type="test_task_type",
        batch_id="test_batch",
        ng_state={},
        ng_state_initial={},
        priority=1,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        is_active=True,
        completion_status="",
        last_leased_ts=0.0,
    )

    # Handle completion
    handle_task_completion("test_project", task, "Done")

    # Verify handler was called
    mock_handler.assert_called_once_with("test_project", task, "Done")


def test_handle_task_completion_no_handler():
    """Test handling task completion without a registered handler"""
    # Create a task with unregistered type
    task = Task(
        task_id="test_task",
        task_type="unregistered_type",
        batch_id="test_batch",
        ng_state={},
        ng_state_initial={},
        priority=1,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        is_active=True,
        completion_status="",
        last_leased_ts=0.0,
    )

    # Should not raise an error
    handle_task_completion("test_project", task, "Done")


def test_handle_task_completion_handler_error():
    """Test that handler errors are propagated"""

    def failing_handler(project_name: str, task: Task, completion_status: str) -> None:
        raise ValueError("Test error")

    _COMPLETION_HANDLERS["failing_type"] = failing_handler

    task = Task(
        task_id="test_task",
        task_type="failing_type",
        batch_id="test_batch",
        ng_state={},
        ng_state_initial={},
        priority=1,
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        is_active=True,
        completion_status="",
        last_leased_ts=0.0,
    )

    # Should raise the error from the handler
    with pytest.raises(ValueError, match="Test error"):
        handle_task_completion("test_project", task, "Done")
