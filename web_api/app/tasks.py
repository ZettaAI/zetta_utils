# pylint: disable=all # type: ignore
import json
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from zetta_utils import log
from zetta_utils.task_management.db.models import TaskFeedbackModel, TaskModel
from zetta_utils.task_management.db.session import get_session_context
from zetta_utils.task_management.segment_link import get_segment_link
from zetta_utils.task_management.task import (
    get_task,
    release_task,
    start_task,
    update_task,
)
from zetta_utils.task_management.task_type import create_task_type, get_task_type
from zetta_utils.task_management.task_types import handle_task_completion, verify_task
from zetta_utils.task_management.timesheet import submit_timesheet
from zetta_utils.task_management.types import Task, TaskType, TaskUpdate

from .utils import generic_exception_handler

logger = log.get_logger()

api = FastAPI()


class NgStateRequest(BaseModel):
    ng_state: str


@api.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    return generic_exception_handler(request, exc)


@api.get("/projects/{project_name}/task_types/{task_type_id}")
async def get_task_type_api(project_name: str, task_type_id: str) -> dict:
    """
    Get a task type by ID.

    :param task_type_id: The ID of the task type to get
    :return: The task type data
    """
    task_type = get_task_type(project_name=project_name, task_type=task_type_id)
    result = {k.replace("task", "task"): v for k, v in task_type.items()}
    return result


@api.get("/projects/{project_name}/tasks/{task_id}")
async def get_task_api(project_name: str, task_id: str) -> dict:
    """
    Get a task by ID.

    :param project_name: The name of the project
    :param task_id: The ID of the task to get
    :return: The task data
    """
    task = get_task(project_name=project_name, task_id=task_id)
    result = {k.replace("task", "task"): v for k, v in task.items()}
    result["ng_state"] = result["ng_state"]
    return result


@api.post("/projects/{project_name}/start_task")
async def start_task_api(
    project_name: str,
    user_id: str,
    task_id: str | None = None,
) -> str | None:
    """
    Start a task for a user.

    :param project_name: The name of the project
    :param user_id: The ID of the user starting the task
    :param task_id: Optional specific task ID to start
    :return: The ID of the started task, or None if no task available
    """
    return start_task(project_name=project_name, user_id=user_id, task_id=task_id)


@api.post("/projects/{project_name}/set_task_ng_state")
async def set_task_ng_state_api(
    project_name: str,
    task_id: str,
    request: NgStateRequest,
):
    """
    Update the neuroglancer state for a task.
    :param project_name: The name of the project
    :param task_id: The ID of the task to update
    :param request: Request body containing the neuroglancer state
    """

    update_data = TaskUpdate(ng_state=json.loads(request.ng_state))
    update_task(project_name=project_name, task_id=task_id, data=update_data)


@api.put("/projects/{project_name}/release_task")
async def release_task_api(
    project_name: str,
    task_id: str,
    user_id: str,
    completion_status: str,
    note: str | None = None,
):
    """
    Release a task with a completion status.

    :param project_name: The name of the project
    :param task_id: The ID of the task to release
    :param user_id: The ID of the user releasing the task
    :param completion_status: The completion status to set
    :param note: Optional note to save with the task
    :return: True if successful
    """
    task = get_task(project_name=project_name, task_id=task_id)

    verification_result = verify_task(
        project_name=project_name, task=task, completion_status=completion_status
    )

    if not verification_result.passed:
        raise HTTPException(
            status_code=409, detail=f"Task verification failed: {verification_result.message}"
        )

    handle_task_completion(
        project_name=project_name, task=task, completion_status=completion_status
    )

    release_success = release_task(
        project_name=project_name,
        user_id=user_id,
        task_id=task_id,
        completion_status=completion_status,
        note=note,
    )

    if not release_success:
        raise HTTPException(status_code=409, detail="Failed to release task")

    return True


@api.post("/projects/{project_name}/submit_timesheet")
async def submit_timesheet_api(
    project_name: str,
    user_id: str,
    duration_seconds: float,
    task_id: str,
) -> None:
    """
    Submit a timesheet entry for a user.

    :param project_name: The name of the project
    :param user_id: The ID of the user submitting the timesheet
    :param duration_seconds: The duration of work in seconds
    :param task_id: The ID of the task to submit timesheet for
    """
    submit_timesheet(
        project_name=project_name,
        user_id=user_id,
        duration_seconds=duration_seconds,
        task_id=task_id,
    )


@api.get("/projects/{project_name}/task_feedback")
async def get_task_feedback_api(
    project_name: str,
    user_id: str,
    limit: int = 20,
) -> list[dict]:
    """
    Get the latest task feedback entries for a project for a specific user.

    :param project_name: The name of the project
    :param user_id: The ID of the user to get feedback for
    :param limit: Maximum number of feedback entries to return (default: 20)
    :return: List of feedback entries with task and feedback data for the user
    """
    with get_session_context() as session:
        # Query feedback entries filtered by user, then get task data separately
        feedbacks = (
            session.query(TaskFeedbackModel)
            .filter(
                TaskFeedbackModel.project_name == project_name,
                TaskFeedbackModel.user_id == user_id,
            )
            .order_by(TaskFeedbackModel.created_at.desc())
            .limit(limit)
            .all()
        )

        feedback_data = []
        for feedback in feedbacks:
            # Get original task data
            original_task = (
                session.query(TaskModel)
                .filter(
                    TaskModel.project_name == feedback.project_name,
                    TaskModel.task_id == feedback.task_id,
                )
                .first()
            )

            # Get feedback task data
            feedback_task = (
                session.query(TaskModel)
                .filter(
                    TaskModel.project_name == feedback.project_name,
                    TaskModel.task_id == feedback.feedback_task_id,
                )
                .first()
            )

            # Map completion status to feedback type
            feedback_type = feedback_task.completion_status if feedback_task else "Unknown"
            feedback_color = "red"  # Default to red for unknown statuses

            if feedback_type == "Accurate":
                feedback_color = "green"
            elif feedback_type == "Fair":
                feedback_color = "yellow"
            elif feedback_type == "Inaccurate":
                feedback_color = "red"

            feedback_data.append(
                {
                    "task_id": feedback.task_id,
                    "task_link": original_task.ng_state if original_task else None,
                    "feedback_link": feedback_task.ng_state if feedback_task else None,
                    "feedback": feedback_type,
                    "feedback_color": feedback_color,
                    "note": feedback_task.note if feedback_task else None,
                    "created_at": feedback.created_at.isoformat() if feedback.created_at else None,
                    "user_id": feedback.user_id,
                    "feedback_id": feedback.feedback_id,
                    "feedback_task_id": feedback.feedback_task_id,
                }
            )

        return feedback_data


@api.get("/projects/{project_name}/segments/{seed_id}/link")
async def get_segment_link_api(
    project_name: str,
    seed_id: int,
    include_certain_ends: bool = True,
    include_uncertain_ends: bool = True,
    include_breadcrumbs: bool = True,
) -> dict:
    """
    Generate neuroglancer link for a segment by seed supervoxel ID.

    :param project_name: The name of the project
    :param seed_id: Seed supervoxel ID (primary key)
    :param include_certain_ends: Whether to include certain endpoints layer (yellow). Defaults to True.
    :param include_uncertain_ends: Whether to include uncertain endpoints layer (red). Defaults to True.
    :param include_breadcrumbs: Whether to include breadcrumbs layer (blue). Defaults to True.
    :return: Dictionary containing the neuroglancer link
    """
    try:
        link = get_segment_link(
            project_name=project_name,
            seed_id=seed_id,
            include_certain_ends=include_certain_ends,
            include_uncertain_ends=include_uncertain_ends,
            include_breadcrumbs=include_breadcrumbs,
        )
        return {"link": link}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
