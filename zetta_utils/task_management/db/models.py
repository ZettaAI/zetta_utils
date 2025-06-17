# pylint: disable=singleton-comparison
from typing import Any, Optional

from sqlalchemy import ARRAY, JSON, BigInteger, Boolean, Float, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Modern SQLAlchemy 2.0 declarative base with proper typing support."""


class ProjectModel(Base):
    """
    SQLAlchemy model for the projects table.

    This table maintains a registry of all projects for O(1) lookups.
    """

    __tablename__ = "projects"

    # Primary key - just the project name
    project_name: Mapped[str] = mapped_column(String, primary_key=True)

    # Segmentation configuration (mandatory)
    segmentation_link: Mapped[str] = mapped_column(String, nullable=False)
    sv_resolution_x: Mapped[float] = mapped_column(Float, nullable=False)
    sv_resolution_y: Mapped[float] = mapped_column(Float, nullable=False)
    sv_resolution_z: Mapped[float] = mapped_column(Float, nullable=False)

    # Metadata columns
    created_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "project_name": self.project_name,
            "segmentation_link": self.segmentation_link,
            "sv_resolution_x": self.sv_resolution_x,
            "sv_resolution_y": self.sv_resolution_y,
            "sv_resolution_z": self.sv_resolution_z,
            "created_at": self.created_at,
            "description": self.description,
            "status": self.status,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ProjectModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=data["project_name"],
            segmentation_link=data["segmentation_link"],
            sv_resolution_x=data["sv_resolution_x"],
            sv_resolution_y=data["sv_resolution_y"],
            sv_resolution_z=data["sv_resolution_z"],
            created_at=data.get("created_at"),
            description=data.get("description"),
            status=data.get("status", "active"),
        )


class TaskTypeModel(Base):
    """
    SQLAlchemy model for the task_types table.
    """

    __tablename__ = "task_types"

    # Composite primary key of project_name and task_type
    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    task_type: Mapped[str] = mapped_column(String, primary_key=True)

    # Columns
    completion_statuses: Mapped[list[Any]] = mapped_column(ARRAY(String), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)

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

    # Composite primary key of project_name and user_id
    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, primary_key=True)

    # Columns
    hourly_rate: Mapped[float] = mapped_column(Float, nullable=False)
    active_task: Mapped[str] = mapped_column(String, nullable=False, default="")
    qualified_task_types: Mapped[list[Any]] = mapped_column(ARRAY(String), nullable=False)

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


class JobModel(Base):
    """
    SQLAlchemy model for the jobs table.
    """

    __tablename__ = "jobs"

    # Composite primary key of project_name and job_id
    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    job_id: Mapped[str] = mapped_column(String, primary_key=True)

    # Columns
    batch_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    job_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    ng_state: Mapped[dict] = mapped_column(JSON, nullable=False)
    id_nonunique: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, index=True)

    # Additional indexes for performance
    __table_args__ = (
        Index("idx_jobs_project_batch", "project_name", "batch_id"),
        Index("idx_jobs_project_status", "project_name", "status"),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary matching the Job TypedDict structure"""
        return {
            "job_id": self.job_id,
            "batch_id": self.batch_id,
            "status": self.status,
            "job_type": self.job_type,
            "ng_state": self.ng_state,
        }

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "JobModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            job_id=data["job_id"],
            batch_id=data["batch_id"],
            status=data["status"],
            job_type=data["job_type"],
            ng_state=data["ng_state"],
            id_nonunique=data.get("id_nonunique"),
        )


class DependencyModel(Base):
    """
    SQLAlchemy model for the dependencies table.
    """

    __tablename__ = "dependencies"

    # Composite primary key of project_name and dependency_id
    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    dependency_id: Mapped[str] = mapped_column(String, primary_key=True)

    # Columns
    task_id: Mapped[str] = mapped_column(String, nullable=False)
    dependent_on_task_id: Mapped[str] = mapped_column(String, nullable=False)
    is_satisfied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    required_completion_status: Mapped[str] = mapped_column(String, nullable=False)

    # Performance indexes for dependency lookups
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

    # Composite primary key of project_name and entry_id
    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    entry_id: Mapped[str] = mapped_column(String, primary_key=True)

    # Columns
    job_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    user: Mapped[str] = mapped_column(
        String, nullable=False
    )  # Using 'user' to match the TypedDict
    seconds_spent: Mapped[int] = mapped_column(Integer, nullable=False)

    # Performance indexes for timesheet queries
    __table_args__ = (
        Index("idx_timesheet_project_task", "project_name", "task_id"),
        Index("idx_timesheet_project_user", "project_name", "user"),
        Index("idx_timesheet_project_job", "project_name", "job_id"),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary matching the Timesheet TypedDict structure"""
        return {
            "entry_id": self.entry_id,
            "job_id": self.job_id,
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
            job_id=data["job_id"],
            task_id=data["task_id"],
            user=data["user"],
            seconds_spent=data["seconds_spent"],
        )


class SegmentTypeModel(Base):
    """
    SQLAlchemy model for the segment_types table.
    
    This table stores the available segment types that can be assigned to segments.
    """

    __tablename__ = "segment_types"

    # Composite primary key - project_name and auto-incrementing id
    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    type_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Unique constraint on project_name + type_name
    __table_args__ = (
        Index("idx_segment_types_project_name", "project_name", "type_name", unique=True),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary"""
        result = {
            "id": self.id,
            "type_name": self.type_name,
            "created_at": self.created_at,
        }
        
        if self.description is not None:
            result["description"] = self.description
        
        return result

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "SegmentTypeModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            type_name=data["type_name"],
            description=data.get("description"),
            created_at=data["created_at"],
        )


class SegmentModel(Base):
    """
    SQLAlchemy model for the segments table.
    
    This table stores segment data with root locations and synapse counts.
    """

    __tablename__ = "segments"

    # Composite primary key - project_name and auto-incrementing segment_id
    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    internal_segment_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Seed location coordinates
    seed_x: Mapped[float] = mapped_column(Float, nullable=False)
    seed_y: Mapped[float] = mapped_column(Float, nullable=False)
    seed_z: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Segment type references (foreign keys to segment_types table)
    segment_type_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    expected_segment_type_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    
    # Segment IDs (BigInteger to handle uint64)
    current_segment_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    seed_sv_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    
    # Skeleton and synapse data
    skeleton_length: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pre_synapse_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    post_synapse_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Task tracking
    task_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    
    # Timestamps
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    
    # Optional extra data
    extra_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Performance indexes
    __table_args__ = (
        Index("idx_segments_project_seed_location", "project_name", "seed_x", "seed_y", "seed_z"),
        Index("idx_segments_project_type_ids", "project_name", "segment_type_id", "expected_segment_type_id"),
        Index("idx_segments_project_current_segment", "project_name", "current_segment_id"),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary"""
        result = {
            "internal_segment_id": self.internal_segment_id,
            "seed_x": self.seed_x,
            "seed_y": self.seed_y,
            "seed_z": self.seed_z,
            "seed_sv_id": self.seed_sv_id,
            "current_segment_id": self.current_segment_id,
            "task_ids": self.task_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        
        if self.segment_type_id is not None:
            result["segment_type_id"] = self.segment_type_id
            
        if self.expected_segment_type_id is not None:
            result["expected_segment_type_id"] = self.expected_segment_type_id
        
        if self.skeleton_length is not None:
            result["skeleton_length"] = self.skeleton_length
        
        if self.pre_synapse_count is not None:
            result["pre_synapse_count"] = self.pre_synapse_count
            
        if self.post_synapse_count is not None:
            result["post_synapse_count"] = self.post_synapse_count
            
        if self.extra_data is not None:
            result["extra_data"] = self.extra_data
        
        return result

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "SegmentModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            seed_x=data["seed_x"],
            seed_y=data["seed_y"],
            seed_z=data["seed_z"],
            seed_sv_id=data["seed_sv_id"],
            current_segment_id=data["current_segment_id"],
            segment_type_id=data.get("segment_type_id"),
            expected_segment_type_id=data.get("expected_segment_type_id"),
            skeleton_length=data.get("skeleton_length"),
            pre_synapse_count=data.get("pre_synapse_count"),
            post_synapse_count=data.get("post_synapse_count"),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            extra_data=data.get("extra_data"),
        )


class TaskModel(Base):
    """
    SQLAlchemy model for the tasks table.
    """

    __tablename__ = "tasks"

    # Composite primary key of project_name and task_id
    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    task_id: Mapped[str] = mapped_column(String, primary_key=True)

    # Columns
    job_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    completion_status: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    assigned_user_id: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    active_user_id: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    completed_user_id: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    ng_state: Mapped[dict] = mapped_column(JSON, nullable=False)
    ng_state_initial: Mapped[dict] = mapped_column(JSON, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    batch_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    last_leased_ts: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    is_paused: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    is_tagged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    task_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    id_nonunique: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    extra_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Performance indexes for common query patterns
    __table_args__ = (
        # Basic lookup indexes
        Index("idx_tasks_project_job", "project_name", "job_id"),
        Index("idx_tasks_project_assigned_user", "project_name", "assigned_user_id"),
        Index("idx_tasks_project_active_user", "project_name", "active_user_id"),
        Index("idx_tasks_project_type_priority", "project_name", "task_type", "priority"),
        Index(
            "idx_tasks_project_active_status", "project_name", "is_active", "completion_status"
        ),
        Index("idx_tasks_project_lease_time", "project_name", "last_leased_ts"),
        # Composite indexes for auto-select queries (most critical for performance)
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
            "job_id": self.job_id,
            "completion_status": self.completion_status,
            "assigned_user_id": self.assigned_user_id,
            "active_user_id": self.active_user_id,
            "completed_user_id": self.completed_user_id,
            "ng_state": self.ng_state,
            "ng_state_initial": self.ng_state_initial,
            "priority": self.priority,
            "batch_id": self.batch_id,
            "last_leased_ts": self.last_leased_ts,
            "is_active": self.is_active,
            "is_paused": self.is_paused,
            "is_tagged": self.is_tagged,
            "task_type": self.task_type,
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
            job_id=data["job_id"],
            completion_status=data.get("completion_status", ""),
            assigned_user_id=data.get("assigned_user_id", ""),
            active_user_id=data.get("active_user_id", ""),
            completed_user_id=data.get("completed_user_id", ""),
            ng_state=data["ng_state"],
            ng_state_initial=data["ng_state_initial"],
            priority=data["priority"],
            batch_id=data["batch_id"],
            last_leased_ts=data.get("last_leased_ts", 0.0),
            is_active=data.get("is_active", True),
            is_paused=data.get("is_paused", False),
            is_tagged=data.get("is_tagged", False),
            task_type=data["task_type"],
            id_nonunique=data.get("id_nonunique"),
            extra_data=data.get("extra_data"),
        )
