from typing import Callable

from ..types import Task

# Type alias for completion handler functions
CompletionHandlerFunc = Callable[[str, Task, str], None]

# Global registry
_COMPLETION_HANDLERS: dict[str, CompletionHandlerFunc] = {}


def register_completion_handler(task_type: str):
    """Decorator to register a completion handler function for a task type."""

    def decorator(func: CompletionHandlerFunc) -> CompletionHandlerFunc:
        _COMPLETION_HANDLERS[task_type] = func
        return func

    return decorator


def get_completion_handler(task_type: str) -> CompletionHandlerFunc | None:
    """Get completion handler function for a task type."""
    return _COMPLETION_HANDLERS.get(task_type)


def handle_task_completion(project_name: str, task: Task, completion_status: str):
    """Handle task completion using its registered handler.

    Args:
        project_name: The project name
        task: The task being completed
        completion_status: The completion status being applied

    Returns:
        None
    """
    handler = get_completion_handler(task["task_type"])
    if handler is not None:
        handler(project_name, task, completion_status)
