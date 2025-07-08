from datetime import datetime
from typing import Literal, NotRequired, TypedDict


class Project(TypedDict):
    """A project with segmentation configuration."""

    project_name: str
    segmentation_path: str
    sv_resolution_x: float
    sv_resolution_y: float
    sv_resolution_z: float
    created_at: NotRequired[str | None]
    description: NotRequired[str | None]
    status: str
    brain_mesh_path: NotRequired[str | None]
    datastack_name: NotRequired[str | None]
    extra_layers: NotRequired[dict | None]


class ProjectUpdate(TypedDict, total=False):
    """Update type for projects."""

    segmentation_path: str
    sv_resolution_x: float
    sv_resolution_y: float
    sv_resolution_z: float
    description: str | None
    status: str
    brain_mesh_path: str | None
    datastack_name: str | None
    extra_layers: dict | None


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
    first_start_ts: NotRequired[float | None]  # First start timestamp
    is_active: bool  # Whether the task can be worked on
    is_paused: NotRequired[bool]  # Whether the task is paused (not auto-selectable)
    is_checked: NotRequired[bool]  # Whether the task has been checked/reviewed
    task_type: str  # Reference to TaskType.task_type
    extra_data: NotRequired[dict | None]  # Additional task-specific data
    note: NotRequired[str | None]  # Optional note field


class TaskUpdate(TypedDict, total=False):
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
    first_start_ts: float | None
    is_active: bool
    is_paused: bool
    is_checked: bool
    task_type: str
    extra_data: dict | None
    note: str | None


class Timesheet(TypedDict):
    entry_id: str
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


class TimesheetEntry(TypedDict):
    """A timesheet entry for work done on a task"""

    start_time: datetime
    duration_seconds: float
    description: str


class TimesheetSubmission(TypedDict):
    """An individual timesheet submission for audit trail"""

    submission_id: int
    user_id: str
    task_id: str
    seconds_spent: int
    submitted_at: str  # ISO format timestamp


class SegmentType(TypedDict):
    """A segment type that can be assigned to segments."""

    type_name: str
    project_name: str
    reference_segment_ids: list[int]
    created_at: str  # ISO format timestamp
    updated_at: str  # ISO format timestamp
    description: NotRequired[str | None]


class SegmentTypeUpdate(TypedDict, total=False):
    """Update type for segment types."""

    reference_segment_ids: list[int]
    description: str | None
    updated_at: str  # ISO format timestamp


class Segment(TypedDict):
    """A segment with seed location and synapse counts."""

    project_name: str
    seed_x: float
    seed_y: float
    seed_z: float
    root_x: NotRequired[float | None]  # Root location x coordinate
    root_y: NotRequired[float | None]  # Root location y coordinate
    root_z: NotRequired[float | None]  # Root location z coordinate
    seed_id: int  # BigInteger for uint64 - Primary key
    current_segment_id: NotRequired[int | None]  # BigInteger for uint64
    task_ids: list[str]  # List of task IDs that worked on this segment
    created_at: str  # ISO format timestamp
    updated_at: str  # ISO format timestamp
    segment_type: NotRequired[str | None]  # Reference to segment_types.type_name
    expected_segment_type: NotRequired[str | None]  # Reference to segment_types.type_name
    batch: NotRequired[str | None]  # Batch identifier
    skeleton_path_length_mm: NotRequired[float | None]
    pre_synapse_count: NotRequired[int | None]
    post_synapse_count: NotRequired[int | None]
    status: Literal["WIP", "Completed", "Retired", "Abandoned"]  # Segment status
    is_exported: bool  # Whether the segment has been exported
    extra_data: NotRequired[dict | None]


class Endpoint(TypedDict):
    """An endpoint location for a segment."""

    endpoint_id: int
    seed_id: int
    x: float
    y: float
    z: float
    status: Literal["CERTAIN", "UNCERTAIN", "CONTINUED", "BREADCRUMB"]
    user: str
    created_at: str  # ISO format timestamp
    updated_at: str  # ISO format timestamp


class EndpointUpdate(TypedDict):
    """A record of an endpoint status change."""

    update_id: int
    endpoint_id: int
    user: str
    new_status: Literal["CERTAIN", "UNCERTAIN", "CONTINUED", "BREADCRUMB"]
    timestamp: str  # ISO format timestamp


class TaskFeedback(TypedDict):
    """A feedback record linking a trace task to its review."""

    feedback_id: int
    task_id: str  # Original trace task
    feedback_task_id: str  # The feedback task that reviewed it
    user_id: str  # User who completed the original trace task
    created_at: str  # ISO format timestamp
