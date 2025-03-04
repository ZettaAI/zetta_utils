# pylint: disable=all # type: ignore
from fastapi import FastAPI, Request

from zetta_utils.task_management.subtask import (
    get_subtask,
    release_subtask,
    start_subtask,
)

from .utils import generic_exception_handler

api = FastAPI()


@api.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    return generic_exception_handler(request, exc)


@api.get("/subtask/{subtask_id}")
async def get_subtask_api(subtask_id: str, project_name: str):
    """
    Get a subtask by ID.

    :param subtask_id: The ID of the subtask to get
    :param project_name: The name of the project
    :return: The subtask data
    """
    return get_subtask(project_name, subtask_id)


@api.post("/subtask/start")
async def start_subtask_api(project_name: str, user_id: str, subtask_id: str | None = None):
    """
    Start a subtask for a user.

    :param project_name: The name of the project
    :param user_id: The ID of the user starting the subtask
    :param subtask_id: Optional specific subtask ID to start
    :return: The ID of the started subtask, or None if no subtask available
    """
    return start_subtask(project_name, user_id, subtask_id)


@api.put("/subtask/release")
async def release_task_api(
    subtask_id: str,
    project_name: str,
    user_id: str,
    completion_status: str,
):
    """
    Release a subtask with a completion status.

    :param subtask_id: The ID of the subtask to release
    :param project_name: The name of the project
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
