# pylint: disable=singleton-comparison
from datetime import datetime

from sqlalchemy import (
    ARRAY,
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _parse_datetime(value: str | int | float | datetime) -> datetime:
    """Convert various datetime formats to datetime object."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    raise ValueError(f"Cannot parse datetime from {type(value)}")


class Base(DeclarativeBase):
    """Modern SQLAlchemy 2.0 declarative base with proper typing support."""


class ProjectModel(Base):
    """
    SQLAlchemy model for the projects table.

    This table maintains a registry of all projects for O(1) lookups.
    """

    __tablename__ = "projects"

    project_name: Mapped[str] = mapped_column(String, primary_key=True)

    created_at: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")

    segmentation_path: Mapped[str] = mapped_column(String, nullable=False)
    brain_mesh_path: Mapped[str | None] = mapped_column(String, nullable=True)
    datastack_name: Mapped[str | None] = mapped_column(String, nullable=True)
    synapse_table: Mapped[str | None] = mapped_column(String, nullable=True)
    sv_resolution_x: Mapped[float] = mapped_column(Float, nullable=False)
    sv_resolution_y: Mapped[float] = mapped_column(Float, nullable=False)
    sv_resolution_z: Mapped[float] = mapped_column(Float, nullable=False)
    extra_layers: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result: dict[str, float | str | dict | None] = {
            "project_name": self.project_name,
            "created_at": self.created_at,
            "description": self.description,
            "status": self.status,
            "segmentation_path": self.segmentation_path,
            "sv_resolution_x": self.sv_resolution_x,
            "sv_resolution_y": self.sv_resolution_y,
            "sv_resolution_z": self.sv_resolution_z,
        }

        if self.brain_mesh_path is not None:
            result["brain_mesh_path"] = self.brain_mesh_path
        if self.datastack_name is not None:
            result["datastack_name"] = self.datastack_name
        if self.synapse_table is not None:
            result["synapse_table"] = self.synapse_table
        if self.extra_layers is not None:
            result["extra_layers"] = self.extra_layers

        return result


class TaskTypeModel(Base):
    """
    SQLAlchemy model for the task_types table.
    """

    __tablename__ = "task_types"

    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    task_type: Mapped[str] = mapped_column(String, primary_key=True)

    completion_statuses: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)

    def to_dict(self) -> dict:
        """Convert the model to a dictionary matching the TypedDict structure"""
        result = {
            "task_type": self.task_type,
            "completion_statuses": self.completion_statuses,
        }

        if self.description:
            result["description"] = self.description

        return result

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "TaskTypeModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            task_type=data["task_type"],
            completion_statuses=data["completion_statuses"],
            description=data.get("description"),
        )


class UserModel(Base):
    """
    SQLAlchemy model for the users table.
    """

    __tablename__ = "users"

    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, primary_key=True)

    hourly_rate: Mapped[float] = mapped_column(Float, nullable=False)
    active_task: Mapped[str] = mapped_column(String, nullable=False, default="")
    qualified_task_types: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)

    def to_dict(self) -> dict:
        """Convert the model to a dictionary matching the User TypedDict structure"""
        return {
            "user_id": self.user_id,
            "hourly_rate": self.hourly_rate,
            "active_task": self.active_task,
            "qualified_task_types": self.qualified_task_types,
        }

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "UserModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            user_id=data["user_id"],
            hourly_rate=data["hourly_rate"],
            active_task=data["active_task"],
            qualified_task_types=data["qualified_task_types"],
        )


class DependencyModel(Base):
    """
    SQLAlchemy model for the dependencies table.
    """

    __tablename__ = "dependencies"

    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    dependency_id: Mapped[str] = mapped_column(String, primary_key=True)

    task_id: Mapped[str] = mapped_column(String, nullable=False)
    dependent_on_task_id: Mapped[str] = mapped_column(String, nullable=False)
    is_satisfied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    required_completion_status: Mapped[str] = mapped_column(String, nullable=False)

    __table_args__ = (
        Index("idx_dependencies_project_task", "project_name", "task_id"),
        Index("idx_dependencies_project_dependent_on", "project_name", "dependent_on_task_id"),
        Index(
            "idx_dependencies_project_task_unsatisfied",
            "project_name",
            "task_id",
            "is_satisfied",
        ),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary matching the Dependency TypedDict structure"""
        return {
            "dependency_id": self.dependency_id,
            "task_id": self.task_id,
            "dependent_on_task_id": self.dependent_on_task_id,
            "is_satisfied": self.is_satisfied,
            "required_completion_status": self.required_completion_status,
        }

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "DependencyModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            dependency_id=data["dependency_id"],
            task_id=data["task_id"],
            dependent_on_task_id=data["dependent_on_task_id"],
            is_satisfied=data.get("is_satisfied", False),
            required_completion_status=data["required_completion_status"],
        )


class TimesheetModel(Base):
    """
    SQLAlchemy model for the timesheet table.
    """

    __tablename__ = "timesheet"

    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    entry_id: Mapped[str] = mapped_column(String, primary_key=True)

    task_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    user: Mapped[str] = mapped_column(String, nullable=False)
    seconds_spent: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("idx_timesheet_project_task", "project_name", "task_id"),
        Index("idx_timesheet_project_user", "project_name", "user"),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary matching the Timesheet TypedDict structure"""
        return {
            "entry_id": self.entry_id,
            "task_id": self.task_id,
            "user": self.user,
            "seconds_spent": self.seconds_spent,
        }

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "TimesheetModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            entry_id=data["entry_id"],
            task_id=data["task_id"],
            user=data["user"],
            seconds_spent=data["seconds_spent"],
        )


class TimesheetSubmissionModel(Base):
    """
    SQLAlchemy model for the timesheet_submissions table.

    This table tracks individual timesheet submissions for audit trail.
    """

    __tablename__ = "timesheet_submissions"

    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    submission_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    user_id: Mapped[str] = mapped_column(String, nullable=False)
    task_id: Mapped[str] = mapped_column(String, nullable=False)
    seconds_spent: Mapped[int] = mapped_column(Integer, nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_timesheet_submissions_user_time", "project_name", "user_id", "submitted_at"),
        Index("idx_timesheet_submissions_task_time", "project_name", "task_id", "submitted_at"),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary"""
        return {
            "submission_id": self.submission_id,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "seconds_spent": self.seconds_spent,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
        }

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "TimesheetSubmissionModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            user_id=data["user_id"],
            task_id=data["task_id"],
            seconds_spent=data["seconds_spent"],
            submitted_at=_parse_datetime(data["submitted_at"]),
        )


class SegmentTypeModel(Base):
    """
    SQLAlchemy model for the segment_types table.

    This table stores the available segment types with string keys and reference segments.
    """

    __tablename__ = "segment_types"

    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    type_name: Mapped[str] = mapped_column(String, primary_key=True)

    sample_segment_ids: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False, default=[]
    )

    description: Mapped[str | None] = mapped_column(String, nullable=True)
    region_mesh: Mapped[str | None] = mapped_column(String, nullable=True)
    seed_mask: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def to_dict(self) -> dict:
        """Convert the model to a dictionary"""
        result = {
            "type_name": self.type_name,
            "project_name": self.project_name,
            "sample_segment_ids": self.sample_segment_ids,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

        if self.description is not None:
            result["description"] = self.description
        if self.region_mesh is not None:
            result["region_mesh"] = self.region_mesh
        if self.seed_mask is not None:
            result["seed_mask"] = self.seed_mask

        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SegmentTypeModel":
        """Create a model instance from a dictionary"""
        return cls(
            type_name=data["type_name"],
            project_name=data["project_name"],
            sample_segment_ids=data.get("sample_segment_ids", []),
            description=data.get("description"),
            region_mesh=data.get("region_mesh"),
            seed_mask=data.get("seed_mask"),
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
        )


class SegmentModel(Base):
    """
    SQLAlchemy model for the segments table.

    This table stores segment data with root locations and synapse counts.
    """

    __tablename__ = "segments"

    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    seed_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    seed_x: Mapped[float] = mapped_column(Float, nullable=False)
    seed_y: Mapped[float] = mapped_column(Float, nullable=False)
    seed_z: Mapped[float] = mapped_column(Float, nullable=False)

    root_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    root_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    root_z: Mapped[float | None] = mapped_column(Float, nullable=True)

    task_ids: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=[])

    segment_type: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    expected_segment_type: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    batch: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    current_segment_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)

    skeleton_path_length_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    pre_synapse_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    post_synapse_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(
        Enum("Raw", "Proofread", "Duplicate", "Wrong type", name="segment_status"),
        nullable=False,
        default="Raw",
    )
    is_exported: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_modified: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    extra_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_segments_segment_type", "segment_type"),
        Index("idx_segments_seed_location", "seed_x", "seed_y", "seed_z"),
        Index("idx_segments_last_modified", "last_modified"),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary"""
        result = {
            "seed_id": self.seed_id,
            "project_name": self.project_name,
            "seed_x": self.seed_x,
            "seed_y": self.seed_y,
            "seed_z": self.seed_z,
            "task_ids": self.task_ids,
            "status": self.status,
            "is_exported": self.is_exported,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
        }

        if self.segment_type is not None:
            result["segment_type"] = self.segment_type
        if self.expected_segment_type is not None:
            result["expected_segment_type"] = self.expected_segment_type
        if self.batch is not None:
            result["batch"] = self.batch

        if self.root_x is not None:
            result["root_x"] = self.root_x
        if self.root_y is not None:
            result["root_y"] = self.root_y
        if self.root_z is not None:
            result["root_z"] = self.root_z

        if self.current_segment_id is not None:
            result["current_segment_id"] = self.current_segment_id

        if self.seed_id is not None:
            result["seed_id"] = self.seed_id

        if self.skeleton_path_length_mm is not None:
            result["skeleton_path_length_mm"] = self.skeleton_path_length_mm

        if self.pre_synapse_count is not None:
            result["pre_synapse_count"] = self.pre_synapse_count

        if self.post_synapse_count is not None:
            result["post_synapse_count"] = self.post_synapse_count

        if self.extra_data is not None:
            result["extra_data"] = self.extra_data

        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SegmentModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=data["project_name"],
            seed_x=data["seed_x"],
            seed_y=data["seed_y"],
            seed_z=data["seed_z"],
            root_x=data.get("root_x"),
            root_y=data.get("root_y"),
            root_z=data.get("root_z"),
            task_ids=data.get("task_ids", []),
            segment_type=data.get("segment_type"),
            expected_segment_type=data.get("expected_segment_type"),
            batch=data.get("batch"),
            current_segment_id=data.get("current_segment_id"),
            seed_id=data.get("seed_id"),
            skeleton_path_length_mm=data.get("skeleton_path_length_mm"),
            pre_synapse_count=data.get("pre_synapse_count"),
            post_synapse_count=data.get("post_synapse_count"),
            status=data.get("status", "Raw"),
            is_exported=data.get("is_exported", False),
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
            last_modified=(
                _parse_datetime(data["last_modified"]) if data.get("last_modified") else None
            ),
            extra_data=data.get("extra_data"),
        )


class EndpointModel(Base):
    """
    SQLAlchemy model for the endpoints table.

    This table stores endpoint locations for segments.
    """

    __tablename__ = "endpoints"

    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    endpoint_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    seed_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    z: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum("CERTAIN", "UNCERTAIN", "CONTINUED", "BREADCRUMB", name="endpoint_status"),
        nullable=False,
    )
    user: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_endpoints_project_segment", "project_name", "seed_id"),
        Index("idx_endpoints_project_location", "project_name", "x", "y", "z"),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary"""
        return {
            "endpoint_id": self.endpoint_id,
            "seed_id": self.seed_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "status": self.status,
            "user": self.user,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "EndpointModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            seed_id=data["seed_id"],
            x=data["x"],
            y=data["y"],
            z=data["z"],
            status=data["status"],
            user=data["user"],
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
        )


class EndpointUpdateModel(Base):
    """
    SQLAlchemy model for the endpoint_updates table.

    This table tracks changes to endpoint statuses.
    """

    __tablename__ = "endpoint_updates"

    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    update_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    endpoint_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    user: Mapped[str] = mapped_column(String, nullable=False)
    new_status: Mapped[str] = mapped_column(
        Enum("CERTAIN", "UNCERTAIN", "CONTINUED", "BREADCRUMB", name="endpoint_status"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_endpoint_updates_project_endpoint", "project_name", "endpoint_id"),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary"""
        return {
            "update_id": self.update_id,
            "endpoint_id": self.endpoint_id,
            "user": self.user,
            "new_status": self.new_status,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "EndpointUpdateModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            endpoint_id=data["endpoint_id"],
            user=data["user"],
            new_status=data["new_status"],
            timestamp=_parse_datetime(data["timestamp"]),
        )


class TaskModel(Base):
    """
    SQLAlchemy model for the tasks table.
    """

    __tablename__ = "tasks"

    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    task_id: Mapped[str] = mapped_column(String, primary_key=True)

    completion_status: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    assigned_user_id: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    active_user_id: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    completed_user_id: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    ng_state: Mapped[dict] = mapped_column(JSON, nullable=False)
    ng_state_initial: Mapped[dict] = mapped_column(JSON, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    batch_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    last_leased_ts: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, index=True)
    first_start_ts: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    is_paused: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    is_checked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    task_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    id_nonunique: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    segment_seed_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    extra_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    note: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_tasks_project_assigned_user", "project_name", "assigned_user_id"),
        Index("idx_tasks_project_active_user", "project_name", "active_user_id"),
        Index("idx_tasks_project_type_priority", "project_name", "task_type", "priority"),
        Index("idx_tasks_project_active_status", "project_name", "is_active", "completion_status"),
        Index("idx_tasks_project_lease_time", "project_name", "last_leased_ts"),
        Index("idx_tasks_project_segment", "project_name", "segment_seed_id"),
        Index(
            "idx_tasks_assigned_search",
            "project_name",
            "is_active",
            "assigned_user_id",
            "active_user_id",
            "completion_status",
            "task_type",
            "priority",
        ),
        Index(
            "idx_tasks_unassigned_search",
            "project_name",
            "is_active",
            "assigned_user_id",
            "active_user_id",
            "completion_status",
            "task_type",
            "priority",
        ),
        Index(
            "idx_tasks_idle_search",
            "project_name",
            "is_active",
            "completion_status",
            "task_type",
            "last_leased_ts",
        ),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary matching the Task TypedDict structure"""
        result = {
            "task_id": self.task_id,
            "completion_status": self.completion_status,
            "assigned_user_id": self.assigned_user_id,
            "active_user_id": self.active_user_id,
            "completed_user_id": self.completed_user_id,
            "ng_state": self.ng_state,
            "ng_state_initial": self.ng_state_initial,
            "priority": self.priority,
            "batch_id": self.batch_id,
            "last_leased_ts": self.last_leased_ts,
            "first_start_ts": self.first_start_ts,
            "is_active": self.is_active,
            "is_paused": self.is_paused,
            "is_checked": self.is_checked,
            "task_type": self.task_type,
            "segment_seed_id": self.segment_seed_id,
            "note": self.note,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

        if self.extra_data is not None:
            result["extra_data"] = self.extra_data

        return result

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "TaskModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            task_id=data["task_id"],
            completion_status=data.get("completion_status", ""),
            assigned_user_id=data.get("assigned_user_id", ""),
            active_user_id=data.get("active_user_id", ""),
            completed_user_id=data.get("completed_user_id", ""),
            ng_state=data["ng_state"],
            ng_state_initial=data["ng_state_initial"],
            priority=data["priority"],
            batch_id=data["batch_id"],
            last_leased_ts=data.get("last_leased_ts", 0.0),
            first_start_ts=data.get("first_start_ts"),
            is_active=data.get("is_active", True),
            is_paused=data.get("is_paused", False),
            is_checked=data.get("is_checked", False),
            task_type=data["task_type"],
            id_nonunique=data.get("id_nonunique"),
            segment_seed_id=data.get("segment_seed_id"),
            extra_data=data.get("extra_data"),
            note=data.get("note"),
            created_at=_parse_datetime(data.get("created_at", datetime.now())),
        )


class TaskFeedbackModel(Base):
    """
    SQLAlchemy model for the task_feedback table.

    Records feedback task completions (excluding Faulty Task status).
    Links original trace tasks to their feedback review tasks.
    """

    __tablename__ = "task_feedback"

    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    feedback_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    task_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    feedback_task_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_task_feedback_project_task", "project_name", "task_id"),
        Index("idx_task_feedback_project_feedback_task", "project_name", "feedback_task_id"),
        Index("idx_task_feedback_project_user", "project_name", "user_id"),
        Index("idx_task_feedback_user_created", "project_name", "user_id", "created_at"),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary"""
        return {
            "feedback_id": self.feedback_id,
            "task_id": self.task_id,
            "feedback_task_id": self.feedback_task_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SplitEditModel(Base):
    """
    SQLAlchemy model for the split_edits table.

    Records split edit operations with sources and sinks coordinates.
    Each edit contains sources and sinks groups with segment IDs and
    coordinates in nanometers.
    """

    __tablename__ = "split_edits"

    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    edit_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    task_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    sources: Mapped[list] = mapped_column(JSON, nullable=False)
    sinks: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_split_edits_project_task", "project_name", "task_id"),
        Index("idx_split_edits_project_user", "project_name", "user_id"),
        Index(
            "idx_split_edits_user_created",
            "project_name",
            "user_id",
            "created_at",
        ),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary"""
        return {
            "edit_id": self.edit_id,
            "task_id": self.task_id,
            "user_id": self.user_id,
            "sources": self.sources,
            "sinks": self.sinks,
            "created_at": (self.created_at.isoformat() if self.created_at else None),
        }

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "SplitEditModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            task_id=data["task_id"],
            user_id=data["user_id"],
            sources=data["sources"],
            sinks=data["sinks"],
            created_at=_parse_datetime(data.get("created_at", datetime.now())),
        )


class MergeEditModel(Base):
    """
    SQLAlchemy model for the merge_edits table.

    Records merge edit operations with two points to be merged.
    Each edit contains a list of two points with segment IDs and
    coordinates in nanometers.
    """

    __tablename__ = "merge_edits"

    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    edit_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    task_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    points: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_merge_edits_project_task", "project_name", "task_id"),
        Index("idx_merge_edits_project_user", "project_name", "user_id"),
        Index(
            "idx_merge_edits_user_created",
            "project_name",
            "user_id",
            "created_at",
        ),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary"""
        return {
            "edit_id": self.edit_id,
            "task_id": self.task_id,
            "user_id": self.user_id,
            "points": self.points,
            "created_at": (self.created_at.isoformat() if self.created_at else None),
        }

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "MergeEditModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            task_id=data["task_id"],
            user_id=data["user_id"],
            points=data["points"],
            created_at=_parse_datetime(data.get("created_at", datetime.now())),
        )


class LockedSegmentModel(Base):
    """
    SQLAlchemy model for the locked_segments table.

    Tracks segments that are currently locked to prevent merging.
    """

    __tablename__ = "locked_segments"

    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    segment_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("idx_locked_segments_created_at", "created_at"),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary"""
        return {
            "project_name": self.project_name,
            "segment_id": self.segment_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LockedSegmentModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=data["project_name"],
            segment_id=data["segment_id"],
            created_at=_parse_datetime(data.get("created_at", datetime.now())),
        )
