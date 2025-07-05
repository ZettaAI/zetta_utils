# pylint: disable=redefined-outer-name,unused-argument

from zetta_utils.task_management.task_types.verification import (
    _VERIFIERS,
    VerificationResult,
    register_verifier,
    verify_task,
)
from zetta_utils.task_management.types import Task


def test_register_verifier():
    """Test registering a task verifier"""

    @register_verifier("test_type")
    def test_verifier(project_name: str, task: Task, completion_status: str) -> VerificationResult:
        return VerificationResult(passed=True, message="Test passed")

    assert _VERIFIERS["test_type"] is test_verifier


def test_verify_task_with_verifier():
    """Test verifying a task with a registered verifier"""

    def strict_verifier(
        project_name: str, task: Task, completion_status: str
    ) -> VerificationResult:
        if completion_status == "Done":
            return VerificationResult(passed=True, message="Valid completion")
        else:
            return VerificationResult(passed=False, message="Invalid status")

    _VERIFIERS["strict_type"] = strict_verifier

    task = Task(
        task_id="test_task",
        task_type="strict_type",
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

    # Test valid completion
    result = verify_task("test_project", task, "Done")
    assert result.passed is True
    assert result.message == "Valid completion"

    # Test invalid completion
    result = verify_task("test_project", task, "Wrong")
    assert result.passed is False
    assert result.message == "Invalid status"


def test_verify_task_no_verifier():
    """Test verifying a task without a registered verifier"""
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

    # Should pass by default
    result = verify_task("test_project", task, "Any Status")
    assert result.passed is True
    assert "No verifier registered" in result.message


def test_verification_result():
    """Test VerificationResult creation"""
    result = VerificationResult(passed=True, message="Success")
    assert result.passed is True
    assert result.message == "Success"

    result = VerificationResult(passed=False, message="Failure reason")
    assert result.passed is False
    assert result.message == "Failure reason"
