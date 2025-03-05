from datetime import datetime
from typing import NotRequired, TypedDict


class User(TypedDict):
    user_id: str
    hourly_rate: float
    active_subtask: str  # Empty string when no active task
    qualified_subtask_types: list[str]  # List of subtask types user can work on


class UserUpdate(TypedDict, total=False):
    hourly_rate: float
    active_subtask: str
    qualified_subtask_types: list[str]


class Subtask(TypedDict):
    task_id: str
    subtask_id: str
    completion_status: str  # Changed from status
    assigned_user_id: str  # Empty string when unassigned
    active_user_id: str  # Empty string when inactive
    completed_user_id: str  # Empty string when not completed
    ng_state: str
    priority: int
    batch_id: str  # Changed from batch_name
    last_leased_ts: float  # 0 for never leased
    is_active: bool  # Whether the subtask can be worked on
    subtask_type: str  # Reference to SubtaskType.subtask_type


class SubtaskUpdate(TypedDict, total=False):
    task_id: str
    subtask_id: str
    completion_status: str
    assigned_user_id: str
    active_user_id: str
    completed_user_id: str
    ng_state: str
    priority: int
    batch_id: str
    last_leased_ts: float
    is_active: bool
    subtask_type: str


class Timesheet(TypedDict):
    entry_id: str
    task_id: str
    subtask_id: str
    user: str
    seconds_spent: int


class TimesheetUpdate(TypedDict, total=False):
    seconds_spent: int


class Dependency(TypedDict):
    dependency_id: str
    subtask_id: str
    dependent_on_subtask_id: str
    is_satisfied: bool
    required_completion_status: str


class DependencyUpdate(TypedDict, total=False):
    is_satisfied: bool
    required_completion_status: str
    subtask_id: str
    dependent_on_subtask_id: str


class SubtaskType(TypedDict):
    """A type of subtask and its allowed completion statuses."""

    subtask_type: str
    completion_statuses: list[str]
    description: NotRequired[str]


class SubtaskTypeUpdate(TypedDict, total=False):
    """Update type for subtask types."""

    completion_statuses: list[str]
    description: str


class Task(TypedDict):
    """A task that contains subtasks."""

    task_id: str
    batch_id: str
    status: str
    task_type: str
    ng_state: str


class TaskUpdate(TypedDict, total=False):
    """Update type for tasks."""

    status: str
    batch_id: str
    ng_state: str
    task_type: str


class TimesheetEntry(TypedDict):
    """A timesheet entry for work done on a subtask"""

    start_time: datetime
    duration_seconds: float
    description: str
