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

    # Metadata columns
    created_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "project_name": self.project_name,
            "created_at": self.created_at,
            "description": self.description,
            "status": self.status,
        }


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
    id_nonunique: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

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
            task_type=data["task_type"],
            id_nonunique=data.get("id_nonunique"),
            extra_data=data.get("extra_data"),
        )
