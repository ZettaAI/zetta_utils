from datetime import datetime
from typing import NotRequired, TypedDict


class User(TypedDict):
    user_id: str
    hourly_rate: float
    active_task: str  # Empty string when no active task
    qualified_task_types: list[str]  # List of task types user can work on


class UserUpdate(TypedDict, total=False):
    hourly_rate: float
    active_task: str
    qualified_task_types: list[str]


class Task(TypedDict):
    job_id: str
    task_id: str
    completion_status: str  # Changed from status
    assigned_user_id: str  # Empty string when unassigned
    active_user_id: str  # Empty string when inactive
    completed_user_id: str  # Empty string when not completed
    ng_state: dict
    ng_state_initial: dict
    priority: int
    batch_id: str  # Changed from batch_name
    last_leased_ts: float  # 0 for never leased
    is_active: bool  # Whether the task can be worked on
    is_paused: NotRequired[bool]  # Whether the task is paused (not auto-selectable)
    task_type: str  # Reference to TaskType.task_type
    extra_data: NotRequired[dict | None]  # Additional task-specific data


class TaskUpdate(TypedDict, total=False):
    job_id: str
    task_id: str
    completion_status: str
    assigned_user_id: str
    active_user_id: str
    completed_user_id: str
    ng_state: dict
    ng_state_initial: dict
    priority: int
    batch_id: str
    last_leased_ts: float
    is_active: bool
    is_paused: bool
    task_type: str
    extra_data: dict | None


class Timesheet(TypedDict):
    entry_id: str
    job_id: str
    task_id: str
    user: str
    seconds_spent: int


class TimesheetUpdate(TypedDict, total=False):
    seconds_spent: int


class Dependency(TypedDict):
    dependency_id: str
    task_id: str
    dependent_on_task_id: str
    is_satisfied: bool
    required_completion_status: str


class DependencyUpdate(TypedDict, total=False):
    is_satisfied: bool
    required_completion_status: str
    task_id: str
    dependent_on_task_id: str


class TaskType(TypedDict):
    """A type of task and its allowed completion statuses."""

    task_type: str
    completion_statuses: list[str]
    description: NotRequired[str]


class TaskTypeUpdate(TypedDict, total=False):
    """Update type for task types."""

    completion_statuses: list[str]
    description: str


class Job(TypedDict):
    """A job that contains tasks."""

    job_id: str
    batch_id: str
    status: str
    job_type: str
    ng_state: dict


class JobUpdate(TypedDict, total=False):
    """Update type for jobs."""

    status: str
    batch_id: str
    ng_state: dict
    job_type: str


class TimesheetEntry(TypedDict):
    """A timesheet entry for work done on a task"""

    start_time: datetime
    duration_seconds: float
    description: str
