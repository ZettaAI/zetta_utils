import copy
from datetime import datetime, timezone
from typing import Any, Optional

from zetta_utils import log
from ..db import SegmentModel

from ..db.models import TaskFeedbackModel, TaskModel
from ..db.session import get_session_context
from ..task import create_task, get_task
from ..types import Task
from ..utils import generate_id_nonunique
from .completion import register_completion_handler
from .creation import register_creation_handler
from .trace_v0 import verify_trace_layers
from .verification import VerificationResult, register_verifier

logger = log.get_logger()


@register_verifier("trace_feedback_v0")
def verify_trace_feedback_v0(
    project_name: str, task: Task, completion_status: str  # pylint: disable=unused-argument
) -> VerificationResult:
    """Verify trace_feedback_v0 task completion."""
    # For faulty tasks, skip validation
    if completion_status == "Faulty Task":
        return VerificationResult(
            passed=True, message="Task marked as faulty - skipping validation"
        )

    ng_state = task.get("ng_state", {})

    # Verify it has the standard trace layers (but allow multiple segments)
    layers_result = verify_trace_layers(ng_state, require_single_segment=False)
    if not layers_result.passed:
        return layers_result

    # Verify it references a valid task
    extra_data = task.get("extra_data", {})
    if not extra_data or "original_task_id" not in extra_data:
        return VerificationResult(
            passed=False, message="Feedback task must reference original task_id in extra_data"
        )

    return VerificationResult(
        passed=True, message=f"Feedback task validated for status '{completion_status}'"
    )


@register_completion_handler("trace_feedback_v0")
def handle_trace_feedback_v0_completion(
    project_name: str, task: Task, completion_status: str  # pylint: disable=unused-argument
) -> None:
    """Handle completion of trace_feedback_v0 tasks."""
    # Skip database entry for faulty tasks
    if completion_status == "Faulty Task":
        logger.info(f"Feedback task {task['task_id']} marked as faulty - skipping database entry")
        return

    # Get the original task ID from extra_data
    extra_data = task.get("extra_data", {})
    if not extra_data:
        extra_data = {}
    original_task_id = extra_data.get("original_task_id")
    if not original_task_id:
        logger.error(f"Feedback task {task['task_id']} missing original task_id in extra_data")
        return

    # Get the original task to find the user who completed it
    original_task = get_task(project_name=project_name, task_id=original_task_id)
    original_user_id = original_task.get("completed_user_id", "")

    if not original_user_id:
        logger.warning(f"Original task {original_task_id} has no completed_user_id")
        # Fall back to extra_data if available
        original_user_id = extra_data.get("original_user", "unknown")

    with get_session_context() as session:
        # Create feedback record
        feedback = TaskFeedbackModel(
            project_name=project_name,
            task_id=original_task_id,
            feedback_task_id=task["task_id"],
            user_id=original_user_id,
            created_at=datetime.now(timezone.utc),
        )
        session.add(feedback)
        session.commit()

        logger.info(
            f"Recorded feedback '{completion_status}' for task {original_task_id} "
            f"(user: {original_user_id}) via feedback task {task['task_id']}"
        )


@register_creation_handler("trace_feedback_v0")
def create_trace_feedback_v0_task(
    project_name: str, segment: SegmentModel, kwargs: dict  # pylint: disable=unused-argument
) -> str:
    """Create a trace_feedback_v0 task for reviewing a completed trace.

    Args:
        project_name: The project name
        segment: Not used for feedback tasks, pass None
        kwargs: Must include 'task_id' - the trace task to review

    Returns:
        The created feedback task_id
    """
    # Get the original task to review
    original_task_id = kwargs.get("task_id")
    if not original_task_id:
        raise ValueError("trace_feedback_v0 requires 'task_id' in kwargs")

    # Get the original task
    original_task = get_task(project_name=project_name, task_id=original_task_id, process_ng_state=False)

    # Verify it's a trace_v0 task
    if original_task["task_type"] != "trace_v0":
        raise ValueError(f"Task {original_task_id} is not a trace_v0 task")

    # Generate unique feedback task ID
    task_id = f"feedback_{original_task_id}_{generate_id_nonunique()}"

    # Copy the ng_state from the original task
    ng_state = { "seed_id": segment.seed_id }

    # Build feedback task data
    task_data = Task(
        task_id=task_id,
        task_type="trace_feedback_v0",
        ng_state=ng_state,
        ng_state_initial=copy.deepcopy(ng_state),
        completion_status="",
        assigned_user_id=kwargs.get("assigned_user_id", ""),
        active_user_id="",
        completed_user_id="",
        priority=kwargs.get("priority", 40),  # Lower priority than trace tasks
        batch_id=kwargs.get("batch_id", "feedback"),
        last_leased_ts=0.0,
        is_active=True,
        is_paused=kwargs.get("is_paused", False),
        is_checked=False,
        extra_data={"original_task_id": original_task_id},
    )

    # Create the task
    with get_session_context() as session:
        created_task_id = create_task(
            project_name=project_name, data=task_data, db_session=session
        )

    logger.info(f"Created feedback task {created_task_id} for trace task {original_task_id}")
    return created_task_id


def create_feedback_task_from_trace(
    project_name: str, trace_task_id: str, add_status_annotation: bool = True
) -> Optional[str]:
    """Create a trace_feedback_v0 task for a completed trace task.

    Args:
        project_name: The project name
        trace_task_id: The ID of the completed trace task
        add_status_annotation: Whether to add Status annotation layer

    Returns:
        The created feedback task ID, or None if creation failed
    """
    with get_session_context() as session:
        # Get the trace task
        trace_task = get_task(project_name=project_name, task_id=trace_task_id, process_ng_state=add_status_annotation)

        # Generate feedback task ID
        feedback_task_id = f"feedback_{trace_task_id}_{generate_id_nonunique()}"

        # Copy ng_state and add annotation layer if requested
        ng_state = copy.deepcopy(trace_task.ng_state)

        if add_status_annotation and ng_state:
            # Add annotation layer with completion status
            layers = ng_state.get("layers", [])

            # Create status annotation layer
            status_layer = {
                "type": "annotation",
                "source": {
                    "url": "local://annotations",
                    "transform": {
                        "outputDimensions": {
                            "x": [4e-9, "m"],
                            "y": [4e-9, "m"],
                            "z": [4.2e-8, "m"],
                        }
                    },
                },
                "tab": "annotations",
                "name": f"Status: {trace_task.completion_status}",
                "visible": True,
            }

            # Add to layers
            layers.append(status_layer)
            ng_state["layers"] = layers

        # Create feedback task with the trace task's final ng_state
        feedback_task = Task(
            task_id=feedback_task_id,
            task_type="trace_feedback_v0",
            ng_state=ng_state,
            ng_state_initial=ng_state,  # Same for initial state
            completion_status="",
            assigned_user_id="",
            active_user_id="",
            completed_user_id="",
            priority=70,  # High priority for feedback
            batch_id="feedback",
            last_leased_ts=0.0,
            is_active=True,
            is_paused=False,
            is_checked=False,
            extra_data={
                "original_task_id": trace_task.task_id,
                "original_user": trace_task.completed_user_id,
                "original_completion_status": trace_task.completion_status,
            },
        )

        # Create the task
        created_task_id = create_task(
            project_name=project_name, data=feedback_task, db_session=session
        )

        logger.info(
            f"Created feedback task {created_task_id} for trace task {trace_task_id} "
            f"(user: {trace_task.completed_user_id})"
        )

        return created_task_id
