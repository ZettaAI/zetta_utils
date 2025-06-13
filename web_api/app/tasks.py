# pylint: disable=all # type: ignore
from fastapi import FastAPI, Request

from zetta_utils.task_management.task import (
    get_task,
    release_task,
    start_task,
    update_task,
)
from zetta_utils.task_management.task_type import (
    create_task_type,
    get_task_type,
)
from zetta_utils.task_management.timesheet import submit_timesheet
from zetta_utils.task_management.types import Task, TaskType, TaskUpdate

from .utils import generic_exception_handler

api = FastAPI()


@api.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    return generic_exception_handler(request, exc)


@api.post("/projects/{project_name}/task_types/{task_type_id}")
async def create_task_type_api(project_name: str, data: TaskType) -> str:
    """
    Create a new task type.

    :param data: The task type data to create
    :return: The ID of the created task type
    """
    return create_task_type(project_name, data)


@api.get("/projects/{project_name}/task_types/{task_type_id}")
async def get_task_type_api(project_name: str, task_type_id: str) -> TaskType:
    """
    Get a task type by ID.

    :param task_type_id: The ID of the task type to get
    :return: The task type data
    """
    return get_task_type(project_name, task_type_id)


@api.get("/projects/{project_name}/tasks/{task_id}")
async def get_task_api(project_name: str, task_id: str) -> Task:
    """
    Get a task by ID.

    :param project_name: The name of the project
    :param task_id: The ID of the task to get
    :return: The task data
    """
    return get_task(project_name, task_id)


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
    return start_task(project_name, user_id, task_id)


@api.post("/projects/{project_name}/set_task_ng_state")
async def set_task_ng_state_api(
    project_name: str,
    task_id: str,
    ng_state: str,
):
    """
    Update the neuroglancer state for a task.
    :param project_name: The name of the project
    :param task_id: The ID of the task to update
    :param ng_state: The new neuroglancer state URL
    """

    update_data = TaskUpdate(ng_state=ng_state)
    update_task(project_name, task_id, update_data)


@api.put("/projects/{project_name}/release_task")
async def release_task_api(
    project_name: str,
    task_id: str,
    user_id: str,
    completion_status: str,
) -> bool:
    """
    Release a task with a completion status.

    :param project_name: The name of the project
    :param task_id: The ID of the task to release
    :param user_id: The ID of the user releasing the task
    :param completion_status: The completion status to set
    :return: True if successful
    """
    return release_task(
        project_name=project_name,
        user_id=user_id,
        task_id=task_id,
        completion_status=completion_status,
    )


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
