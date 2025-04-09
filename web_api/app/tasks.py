# pylint: disable=all # type: ignore
from fastapi import FastAPI, Request

from zetta_utils.task_management.subtask import (
    get_subtask,
    release_subtask,
    start_subtask,
    update_subtask,
)
from zetta_utils.task_management.subtask_type import (
    create_subtask_type,
    get_subtask_type,
)
from zetta_utils.task_management.timesheet import submit_timesheet
from zetta_utils.task_management.types import Subtask, SubtaskType, SubtaskUpdate

from .utils import generic_exception_handler

api = FastAPI()


@api.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    return generic_exception_handler(request, exc)


@api.post("/projects/{project_name}/subtask_types/{subtask_type_id}")
async def create_subtask_type_api(project_name: str, data: SubtaskType) -> str:
    """
    Create a new subtask type.

    :param data: The subtask type data to create
    :return: The ID of the created subtask type
    """
    return create_subtask_type(project_name, data)


@api.get("/projects/{project_name}/subtask_types/{subtask_type_id}")
async def get_subtask_type_api(project_name: str, subtask_type_id: str) -> SubtaskType:
    """
    Get a subtask type by ID.

    :param subtask_type_id: The ID of the subtask type to get
    :return: The subtask type data
    """
    return get_subtask_type(project_name, subtask_type_id)


@api.get("/projects/{project_name}/subtasks/{subtask_id}")
async def get_subtask_api(project_name: str, subtask_id: str) -> Subtask:
    """
    Get a subtask by ID.

    :param project_name: The name of the project
    :param subtask_id: The ID of the subtask to get
    :return: The subtask data
    """
    return get_subtask(project_name, subtask_id)


@api.post("/projects/{project_name}/start_subtask")
async def start_subtask_api(
    project_name: str,
    user_id: str,
    subtask_id: str | None = None,
) -> str | None:
    """
    Start a subtask for a user.

    :param project_name: The name of the project
    :param user_id: The ID of the user starting the subtask
    :param subtask_id: Optional specific subtask ID to start
    :return: The ID of the started subtask, or None if no subtask available
    """
    return start_subtask(project_name, user_id, subtask_id)


@api.post("/projects/{project_name}/set_subtask_ng_state")
async def set_subtask_ng_state_api(
    project_name: str,
    subtask_id: str,
    ng_state: str,
):
    """
    Update the neuroglancer state for a subtask.
    :param project_name: The name of the project
    :param subtask_id: The ID of the subtask to update
    :param ng_state: The new neuroglancer state URL
    """

    update_data = SubtaskUpdate(ng_state=ng_state)
    update_subtask(project_name, subtask_id, update_data)


@api.put("/projects/{project_name}/release_subtask")
async def release_subtask_api(
    project_name: str,
    subtask_id: str,
    user_id: str,
    completion_status: str,
) -> bool:
    """
    Release a subtask with a completion status.

    :param project_name: The name of the project
    :param subtask_id: The ID of the subtask to release
    :param user_id: The ID of the user releasing the subtask
    :param completion_status: The completion status to set
    :return: True if successful
    """
    return release_subtask(
        project_name=project_name,
        user_id=user_id,
        subtask_id=subtask_id,
        completion_status=completion_status,
    )


@api.post("/projects/{project_name}/submit_timesheet")
async def submit_timesheet_api(
    project_name: str,
    user_id: str,
    duration_seconds: float,
    subtask_id: str,
) -> None:
    """
    Submit a timesheet entry for a user.

    :param project_name: The name of the project
    :param user_id: The ID of the user submitting the timesheet
    :param duration_seconds: The duration of work in seconds
    :param subtask_id: The ID of the subtask to submit timesheet for
    """
    submit_timesheet(
        project_name=project_name,
        user_id=user_id,
        duration_seconds=duration_seconds,
        subtask_id=subtask_id,
    )
