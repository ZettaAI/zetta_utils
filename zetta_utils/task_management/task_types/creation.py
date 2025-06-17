from typing import Callable

import attrs

from zetta_utils import log

from ..db.models import SegmentModel
from ..db.session import get_session_context

logger = log.get_logger()

# Type alias for creation handler functions
CreationHandlerFunc = Callable[[str, SegmentModel, dict], str]

# Global registry
_CREATION_HANDLERS: dict[str, CreationHandlerFunc] = {}


@attrs.frozen
class CreationResult:
    """Result of task creation."""

    success: bool
    message: str
    task_id: str | None = None


def register_creation_handler(task_type: str):
    """Decorator to register a creation handler function for a task type."""

    def decorator(func: CreationHandlerFunc) -> CreationHandlerFunc:
        _CREATION_HANDLERS[task_type] = func
        return func

    return decorator


def get_creation_handler(task_type: str) -> CreationHandlerFunc | None:
    """Get creation handler function for a task type."""
    return _CREATION_HANDLERS.get(task_type)


def add_segment_task(
    project_name: str, seed_id: int, task_type: str, task_creation_kwargs: dict | None = None
) -> str:
    """Create a task for a segment.

    Args:
        project_name: The project name
        seed_id: The seed_id of the segment
        task_type: The type of task to create
        task_creation_kwargs: Optional kwargs to pass to the task creation handler

    Returns:
        The created task_id

    Raises:
        ValueError: If segment not found or handler not registered
    """
    if task_creation_kwargs is None:
        task_creation_kwargs = {}

    with get_session_context() as session:
        # Get the segment
        segment = (
            session.query(SegmentModel)
            .filter_by(project_name=project_name, seed_id=seed_id)
            .first()
        )

        if not segment:
            raise ValueError(f"Segment with seed_id {seed_id} not found in project {project_name}")

        # Get the handler
        handler = get_creation_handler(task_type)
        if not handler:
            raise ValueError(f"No creation handler registered for task type: {task_type}")

        # Call the handler
        logger.info(f"Creating {task_type} task for segment {seed_id} in project {project_name}")
        task_id = handler(project_name, segment, task_creation_kwargs)

        logger.info(f"Created task {task_id} for segment {seed_id}")
        return task_id
