from typing import Callable
import attrs

from ..types import Task

# Type alias for verifier functions
VerifierFunc = Callable[[Task, str], "VerificationResult"]

# Global registry
_VERIFIERS: dict[str, VerifierFunc] = {}


@attrs.frozen
class VerificationResult:
    """Result of task verification."""
    passed: bool
    message: str


def register_verifier(task_type: str):
    """Decorator to register a verifier function for a task type."""
    def decorator(func: VerifierFunc) -> VerifierFunc:
        _VERIFIERS[task_type] = func
        return func
    return decorator


def get_verifier(task_type: str) -> VerifierFunc | None:
    """Get verifier function for a task type."""
    return _VERIFIERS.get(task_type)


def verify_task(task: Task, completion_status: str) -> VerificationResult:
    """Verify a task using its registered verifier.
    
    Args:
        task: The task to verify
        completion_status: The completion status being applied to the task
        
    Returns:
        VerificationResult with passed status and message
    """
    verifier = get_verifier(task["task_type"])
    
    if not verifier:
        return VerificationResult(
            passed=True,
            message=f"No verifier registered for task type: {task['task_type']}"
        )
    
    return verifier(task, completion_status)