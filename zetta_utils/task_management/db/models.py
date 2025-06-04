# pylint: disable=singleton-comparison
from typing import Any, Optional

from sqlalchemy import ARRAY, BigInteger, Boolean, Float, Index, Integer, String
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


class SubtaskTypeModel(Base):
    """
    SQLAlchemy model for the subtask_types table.
    """

    __tablename__ = "subtask_types"

    # Composite primary key of project_name and subtask_type
    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    subtask_type: Mapped[str] = mapped_column(String, primary_key=True)

    # Columns
    completion_statuses: Mapped[list[Any]] = mapped_column(ARRAY(String), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    def to_dict(self) -> dict:
        """Convert the model to a dictionary matching the TypedDict structure"""
        result = {
            "subtask_type": self.subtask_type,
            "completion_statuses": self.completion_statuses,
        }

        if self.description:
            result["description"] = self.description

        return result

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "SubtaskTypeModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            subtask_type=data["subtask_type"],
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
    active_subtask: Mapped[str] = mapped_column(String, nullable=False, default="")
    qualified_subtask_types: Mapped[list[Any]] = mapped_column(ARRAY(String), nullable=False)

    def to_dict(self) -> dict:
        """Convert the model to a dictionary matching the User TypedDict structure"""
        return {
            "user_id": self.user_id,
            "hourly_rate": self.hourly_rate,
            "active_subtask": self.active_subtask,
            "qualified_subtask_types": self.qualified_subtask_types,
        }

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "UserModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            user_id=data["user_id"],
            hourly_rate=data["hourly_rate"],
            active_subtask=data["active_subtask"],
            qualified_subtask_types=data["qualified_subtask_types"],
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
    batch_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    task_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    ng_state: Mapped[str] = mapped_column(String, nullable=False)
    id_nonunique: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    # Additional indexes for performance
    __table_args__ = (
        Index("idx_tasks_project_batch", "project_name", "batch_id"),
        Index("idx_tasks_project_status", "project_name", "status"),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary matching the Task TypedDict structure"""
        return {
            "task_id": self.task_id,
            "batch_id": self.batch_id,
            "status": self.status,
            "task_type": self.task_type,
            "ng_state": self.ng_state,
        }

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "TaskModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            task_id=data["task_id"],
            batch_id=data["batch_id"],
            status=data["status"],
            task_type=data["task_type"],
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
    subtask_id: Mapped[str] = mapped_column(String, nullable=False)
    dependent_on_subtask_id: Mapped[str] = mapped_column(String, nullable=False)
    is_satisfied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    required_completion_status: Mapped[str] = mapped_column(String, nullable=False)

    # Performance indexes for dependency lookups
    __table_args__ = (
        Index("idx_dependencies_project_subtask", "project_name", "subtask_id"),
        Index("idx_dependencies_project_dependent_on", "project_name", "dependent_on_subtask_id"),
        Index(
            "idx_dependencies_project_subtask_unsatisfied",
            "project_name",
            "subtask_id",
            "is_satisfied",
        ),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary matching the Dependency TypedDict structure"""
        return {
            "dependency_id": self.dependency_id,
            "subtask_id": self.subtask_id,
            "dependent_on_subtask_id": self.dependent_on_subtask_id,
            "is_satisfied": self.is_satisfied,
            "required_completion_status": self.required_completion_status,
        }

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "DependencyModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            dependency_id=data["dependency_id"],
            subtask_id=data["subtask_id"],
            dependent_on_subtask_id=data["dependent_on_subtask_id"],
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
    task_id: Mapped[str] = mapped_column(String, nullable=False)
    subtask_id: Mapped[str] = mapped_column(String, nullable=False)
    user: Mapped[str] = mapped_column(
        String, nullable=False
    )  # Using 'user' to match the TypedDict
    seconds_spent: Mapped[int] = mapped_column(Integer, nullable=False)

    # Performance indexes for timesheet queries
    __table_args__ = (
        Index("idx_timesheet_project_subtask", "project_name", "subtask_id"),
        Index("idx_timesheet_project_user", "project_name", "user"),
        Index("idx_timesheet_project_task", "project_name", "task_id"),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary matching the Timesheet TypedDict structure"""
        return {
            "entry_id": self.entry_id,
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
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
            subtask_id=data["subtask_id"],
            user=data["user"],
            seconds_spent=data["seconds_spent"],
        )


class SubtaskModel(Base):
    """
    SQLAlchemy model for the subtasks table.
    """

    __tablename__ = "subtasks"

    # Composite primary key of project_name and subtask_id
    project_name: Mapped[str] = mapped_column(String, primary_key=True)
    subtask_id: Mapped[str] = mapped_column(String, primary_key=True)

    # Columns
    task_id: Mapped[str] = mapped_column(String, nullable=False)
    completion_status: Mapped[str] = mapped_column(String, nullable=False, default="")
    assigned_user_id: Mapped[str] = mapped_column(String, nullable=False, default="")
    active_user_id: Mapped[str] = mapped_column(String, nullable=False, default="")
    completed_user_id: Mapped[str] = mapped_column(String, nullable=False, default="")
    ng_state: Mapped[str] = mapped_column(String, nullable=False)
    ng_state_initial: Mapped[str] = mapped_column(String, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False)
    batch_id: Mapped[str] = mapped_column(String, nullable=False)
    last_leased_ts: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_paused: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    subtask_type: Mapped[str] = mapped_column(String, nullable=False)
    id_nonunique: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    # Performance indexes for common query patterns
    __table_args__ = (
        # Basic lookup indexes
        Index("idx_subtasks_project_task", "project_name", "task_id"),
        Index("idx_subtasks_project_assigned_user", "project_name", "assigned_user_id"),
        Index("idx_subtasks_project_active_user", "project_name", "active_user_id"),
        Index("idx_subtasks_project_type_priority", "project_name", "subtask_type", "priority"),
        Index(
            "idx_subtasks_project_active_status", "project_name", "is_active", "completion_status"
        ),
        Index("idx_subtasks_project_lease_time", "project_name", "last_leased_ts"),
        # Composite indexes for auto-select queries (most critical for performance)
        Index(
            "idx_subtasks_assigned_search",
            "project_name",
            "is_active",
            "assigned_user_id",
            "active_user_id",
            "completion_status",
            "subtask_type",
            "priority",
        ),
        Index(
            "idx_subtasks_unassigned_search",
            "project_name",
            "is_active",
            "assigned_user_id",
            "active_user_id",
            "completion_status",
            "subtask_type",
            "priority",
        ),
        Index(
            "idx_subtasks_idle_search",
            "project_name",
            "is_active",
            "completion_status",
            "subtask_type",
            "last_leased_ts",
        ),
    )

    def to_dict(self) -> dict:
        """Convert the model to a dictionary matching the Subtask TypedDict structure"""
        return {
            "subtask_id": self.subtask_id,
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
            "is_active": self.is_active,
            "is_paused": self.is_paused,
            "subtask_type": self.subtask_type,
        }

    @classmethod
    def from_dict(cls, project_name: str, data: dict) -> "SubtaskModel":
        """Create a model instance from a dictionary"""
        return cls(
            project_name=project_name,
            subtask_id=data["subtask_id"],
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
            is_active=data.get("is_active", True),
            is_paused=data.get("is_paused", False),
            subtask_type=data["subtask_type"],
            id_nonunique=data.get("id_nonunique"),
        )
