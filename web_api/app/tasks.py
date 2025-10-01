# pylint: disable=all # type: ignore
import json
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from zetta_utils import log
from zetta_utils.task_management.db.models import TaskFeedbackModel, TaskModel
from zetta_utils.task_management.db.session import get_session_context
from zetta_utils.task_management.merge_edit import (
    create_merge_edit,
    get_merge_edit_by_id,
    get_merge_edits_by_task,
    get_merge_edits_by_user,
)
from zetta_utils.task_management.ng_state import (
    get_segment_link,
    get_trace_task_link,
    get_trace_task_state,
)
from zetta_utils.task_management.seg_trace_utils.ingest_segment_coordinates import (
    ingest_validated_coordinates,
)
from zetta_utils.task_management.split_edit import (
    create_split_edit,
    get_split_edit_by_id,
    get_split_edits_by_task,
    get_split_edits_by_user,
)
from zetta_utils.task_management.task import (
    get_task,
    release_task,
    start_task,
    update_task,
)
from zetta_utils.task_management.task_type import get_task_type
from zetta_utils.task_management.task_types import handle_task_completion, verify_task
from zetta_utils.task_management.timesheet import submit_timesheet
from zetta_utils.task_management.types import TaskUpdate

from .utils import generic_exception_handler

logger = log.get_logger()

api = FastAPI()


class NgStateRequest(BaseModel):
    ng_state: str


class SplitEditRequest(BaseModel):
    user_id: str
    task_id: str
    sources: list[list]
    sinks: list[list]


class MergeEditRequest(BaseModel):
    user_id: str
    task_id: str
    points: list[list]


class IngestSegmentsRequest(BaseModel):
    valid_coordinates: list[dict]
    expected_neuron_type: str
    batch_name: str


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

    try:
        release_task(
            project_name=project_name,
            user_id=user_id,
            task_id=task_id,
            completion_status=completion_status,
            note=note,
        )
    except Exception as e:
        logger.error(f"Failed to release task: {e}")
        raise HTTPException(status_code=409, detail=f"Failed to release task: {str(e)}")

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


@api.post("/projects/{project_name}/split_edit")
async def create_split_edit_api(
    project_name: str,
    request: SplitEditRequest,
) -> dict:
    """
    Create a new split edit record.

    :param project_name: The name of the project
    :param request: Request body containing user_id, task_id, sources, and sinks
    :return: Dict containing the created edit_id
    """
    try:
        edit_id = create_split_edit(
            project_name=project_name,
            task_id=request.task_id,
            user_id=request.user_id,
            sources=request.sources,
            sinks=request.sinks,
        )
        return {"edit_id": edit_id}
    except Exception as e:
        logger.error(f"Failed to create split edit: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create split edit: {str(e)}")


@api.post("/projects/{project_name}/merge_edit")
async def create_merge_edit_api(
    project_name: str,
    request: MergeEditRequest,
) -> dict:
    """
    Create a new merge edit record.

    :param project_name: The name of the project
    :param request: Request body containing user_id, task_id, and points
    :return: Dict containing the created edit_id
    """
    try:
        edit_id = create_merge_edit(
            project_name=project_name,
            task_id=request.task_id,
            user_id=request.user_id,
            points=request.points,
        )
        return {"edit_id": edit_id}
    except Exception as e:
        logger.error(f"Failed to create merge edit: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create merge edit: {str(e)}")


@api.get("/projects/{project_name}/split_edits")
async def get_split_edits_api(
    project_name: str,
    task_id: str | None = None,
    user_id: str | None = None,
) -> list[dict]:
    """
    Get split edits filtered by task_id or user_id.

    :param project_name: The name of the project
    :param task_id: Optional task ID to filter by
    :param user_id: Optional user ID to filter by
    :return: List of split edit records
    """
    if not task_id and not user_id:
        raise HTTPException(
            status_code=400, detail="Either task_id or user_id query parameter is required"
        )

    try:
        if task_id:
            return get_split_edits_by_task(project_name=project_name, task_id=task_id)
        else:
            assert user_id is not None  # Already validated above
            return get_split_edits_by_user(project_name=project_name, user_id=user_id)
    except Exception as e:
        logger.error(f"Failed to get split edits: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get split edits: {str(e)}")


@api.get("/projects/{project_name}/split_edits/{edit_id}")
async def get_split_edit_by_id_api(
    project_name: str,
    edit_id: int,
) -> dict:
    """
    Get a specific split edit by ID.

    :param project_name: The name of the project
    :param edit_id: The ID of the split edit to retrieve
    :return: Split edit record or 404 if not found
    """
    try:
        split_edit = get_split_edit_by_id(project_name=project_name, edit_id=edit_id)
        if not split_edit:
            raise HTTPException(status_code=404, detail="Split edit not found")
        return split_edit
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get split edit by ID: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get split edit: {str(e)}")


@api.get("/projects/{project_name}/merge_edits")
async def get_merge_edits_api(
    project_name: str,
    task_id: str | None = None,
    user_id: str | None = None,
) -> list[dict]:
    """
    Get merge edits filtered by task_id or user_id.

    :param project_name: The name of the project
    :param task_id: Optional task ID to filter by
    :param user_id: Optional user ID to filter by
    :return: List of merge edit records
    """
    if not task_id and not user_id:
        raise HTTPException(
            status_code=400, detail="Either task_id or user_id query parameter is required"
        )

    try:
        if task_id:
            return get_merge_edits_by_task(project_name=project_name, task_id=task_id)
        else:
            assert user_id is not None  # Already validated above
            return get_merge_edits_by_user(project_name=project_name, user_id=user_id)
    except Exception as e:
        logger.error(f"Failed to get merge edits: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get merge edits: {str(e)}")


@api.get("/projects/{project_name}/merge_edits/{edit_id}")
async def get_merge_edit_by_id_api(
    project_name: str,
    edit_id: int,
) -> dict:
    """
    Get a specific merge edit by ID.

    :param project_name: The name of the project
    :param edit_id: The ID of the merge edit to retrieve
    :return: Merge edit record or 404 if not found
    """
    try:
        merge_edit = get_merge_edit_by_id(project_name=project_name, edit_id=edit_id)
        if not merge_edit:
            raise HTTPException(status_code=404, detail="Merge edit not found")
        return merge_edit
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get merge edit by ID: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get merge edit: {str(e)}")


@api.get("/projects/{project_name}/tasks/{task_id}/trace_state")
async def get_trace_task_state_api(
    project_name: str,
    task_id: str,
    include_certain_ends: bool = True,
    include_uncertain_ends: bool = True,
    include_breadcrumbs: bool = True,
    include_segment_type_layers: bool = True,
    include_merges: bool = True,
) -> dict:
    """
    Generate neuroglancer state for a trace task with merge annotations.

    :param project_name: The name of the project
    :param task_id: ID of the trace task
    :param include_certain_ends: Whether to include certain endpoints layer (yellow). Defaults to True.
    :param include_uncertain_ends: Whether to include uncertain endpoints layer (red). Defaults to True.
    :param include_breadcrumbs: Whether to include breadcrumbs layer (blue). Defaults to True.
    :param include_segment_type_layers: Whether to include segment type layers. Defaults to True.
    :param include_merges: Whether to include merge edits as line annotations (orange). Defaults to True.
    :return: Dictionary containing the neuroglancer state
    """
    try:
        ng_state = get_trace_task_state(
            project_name=project_name,
            task_id=task_id,
            include_certain_ends=include_certain_ends,
            include_uncertain_ends=include_uncertain_ends,
            include_breadcrumbs=include_breadcrumbs,
            include_segment_type_layers=include_segment_type_layers,
            include_merges=include_merges,
        )
        return {
            "project_name": project_name,
            "task_id": task_id,
            "ng_state": ng_state,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get trace task state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trace task state: {str(e)}")


@api.get("/projects/{project_name}/tasks/{task_id}/trace_link")
async def get_trace_task_link_api(
    project_name: str,
    task_id: str,
    include_certain_ends: bool = True,
    include_uncertain_ends: bool = True,
    include_breadcrumbs: bool = True,
    include_segment_type_layers: bool = True,
    include_merges: bool = True,
) -> dict:
    """
    Generate spelunker link for a trace task with merge annotations.

    :param project_name: The name of the project
    :param task_id: ID of the trace task
    :param include_certain_ends: Whether to include certain endpoints layer (yellow). Defaults to True.
    :param include_uncertain_ends: Whether to include uncertain endpoints layer (red). Defaults to True.
    :param include_breadcrumbs: Whether to include breadcrumbs layer (blue). Defaults to True.
    :param include_segment_type_layers: Whether to include segment type layers. Defaults to True.
    :param include_merges: Whether to include merge edits as line annotations (orange). Defaults to True.
    :return: Dictionary containing the spelunker link
    """
    try:
        link = get_trace_task_link(
            project_name=project_name,
            task_id=task_id,
            include_certain_ends=include_certain_ends,
            include_uncertain_ends=include_uncertain_ends,
            include_breadcrumbs=include_breadcrumbs,
            include_segment_type_layers=include_segment_type_layers,
            include_merges=include_merges,
        )
        return {
            "project_name": project_name,
            "task_id": task_id,
            "spelunker_link": link,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get trace task link: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trace task link: {str(e)}")


@api.post("/projects/{project_name}/ingest_segments")
async def ingest_segments_api(
    project_name: str,
    request: IngestSegmentsRequest,
) -> dict:
    """
    Ingest validated segment coordinates into the database.

    :param project_name: The name of the project
    :param request: Request body containing valid_coordinates, expected_neuron_type, and batch_name
    :return: Dictionary containing ingestion results
    """
    try:
        results = ingest_validated_coordinates(
            project_name=project_name,
            valid_coordinates=request.valid_coordinates,
            expected_neuron_type=request.expected_neuron_type,
            batch_name=request.batch_name,
        )
        return results
    except Exception as e:
        logger.error(f"Failed to ingest segments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest segments: {str(e)}")
