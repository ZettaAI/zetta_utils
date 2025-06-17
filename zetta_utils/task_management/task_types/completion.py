from typing import Callable
import attrs

from ..types import Task

# Type alias for completion handler functions
CompletionHandlerFunc = Callable[[Task, str], "CompletionResult"]

# Global registry
_COMPLETION_HANDLERS: dict[str, CompletionHandlerFunc] = {}


@attrs.frozen
class CompletionResult:
    """Result of task completion handling."""
    success: bool
    message: str
    updated_ng_state: dict | None = None  # Optional updated neuroglancer state


def register_completion_handler(task_type: str):
    """Decorator to register a completion handler function for a task type."""
    def decorator(func: CompletionHandlerFunc) -> CompletionHandlerFunc:
        _COMPLETION_HANDLERS[task_type] = func
        return func
    return decorator


def get_completion_handler(task_type: str) -> CompletionHandlerFunc | None:
    """Get completion handler function for a task type."""
    return _COMPLETION_HANDLERS.get(task_type)


def handle_task_completion(task: Task, completion_status: str) -> CompletionResult:
    """Handle task completion using its registered handler.
    
    Args:
        task: The task being completed
        completion_status: The completion status being applied
        
    Returns:
        CompletionResult with success status, message, and optional updated ng_state
    """
    handler = get_completion_handler(task["task_type"])
    
    if not handler:
        return CompletionResult(
            success=True,
            message=f"No completion handler registered for task type: {task['task_type']}",
            updated_ng_state=None
        )
    
    return handler(task, completion_status)