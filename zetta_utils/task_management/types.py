from datetime import datetime
from typing import NotRequired, TypedDict


class Project(TypedDict):
    """A project with segmentation configuration."""
    
    project_name: str
    segmentation_link: str
    sv_resolution_x: float
    sv_resolution_y: float
    sv_resolution_z: float
    created_at: NotRequired[str | None]
    description: NotRequired[str | None]
    status: str


class ProjectUpdate(TypedDict, total=False):
    """Update type for projects."""
    
    segmentation_link: str
    sv_resolution_x: float
    sv_resolution_y: float
    sv_resolution_z: float
    description: str | None
    status: str


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
    is_tagged: NotRequired[bool]  # Whether the task is tagged for special attention
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
    is_tagged: bool
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


class SegmentType(TypedDict):
    """A segment type that can be assigned to segments."""

    id: int
    type_name: str
    created_at: float
    description: NotRequired[str | None]


class SegmentTypeUpdate(TypedDict, total=False):
    """Update type for segment types."""

    description: str | None


class Segment(TypedDict):
    """A segment with seed location and synapse counts."""

    internal_segment_id: int
    seed_x: float
    seed_y: float
    seed_z: float
    seed_sv_id: int  # BigInteger for uint64, non-nullable
    current_segment_id: int  # BigInteger for uint64, non-nullable
    task_ids: list[str]  # List of task IDs that worked on this segment
    created_at: float
    updated_at: float
    segment_type_id: NotRequired[int | None]  # Reference to segment_types.id
    expected_segment_type_id: NotRequired[int | None]  # Reference to segment_types.id
    skeleton_length: NotRequired[float | None]
    pre_synapse_count: NotRequired[int | None]
    post_synapse_count: NotRequired[int | None]
    extra_data: NotRequired[dict | None]


class SegmentUpdate(TypedDict, total=False):
    """Update type for segments."""

    seed_x: float
    seed_y: float
    seed_z: float
    seed_sv_id: int
    current_segment_id: int
    task_ids: list[str]
    segment_type_id: int | None
    expected_segment_type_id: int | None
    skeleton_length: float | None
    pre_synapse_count: int | None
    post_synapse_count: int | None
    updated_at: float
    extra_data: dict | None
