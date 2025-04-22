import json
from typing import Any, List, Optional, Type, cast

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    TypeDecorator,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ArrayType(TypeDecorator):
    """
    SQLAlchemy type that stores lists as JSON in SQLite and as arrays in PostgreSQL.
    This makes our models compatible with both database types.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Optional[List[Any]], dialect: Any) -> Optional[str]:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value  # PostgreSQL can handle arrays natively
        else:
            return json.dumps(value)  # SQLite needs JSON string

    def process_result_value(self, value: Optional[str], dialect: Any) -> Optional[List[Any]]:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value  # PostgreSQL returns arrays
        else:
            return cast(List[Any], json.loads(value))  # SQLite returns JSON string


class SubtaskTypeModel(Base):
    """
    SQLAlchemy model for the subtask_types table.
    """

    __tablename__ = "subtask_types"

    # Composite primary key of project_name and subtask_type
    project_name = Column(String, primary_key=True)
    subtask_type = Column(String, primary_key=True)

    # Columns
    completion_statuses = Column(ArrayType, nullable=False)
    description = Column(String, nullable=True)

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
    project_name = Column(String, primary_key=True)
    user_id = Column(String, primary_key=True)

    # Columns
    hourly_rate = Column(Float, nullable=False)
    active_subtask = Column(String, nullable=False, default="")
    qualified_subtask_types = Column(ArrayType, nullable=False)

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
    project_name = Column(String, primary_key=True)
    task_id = Column(String, primary_key=True)

    # Columns
    batch_id = Column(String, nullable=False)
    status = Column(String, nullable=False)
    task_type = Column(String, nullable=False)
    ng_state = Column(String, nullable=False)
    id_nonunique = Column(String, nullable=True)  # For compatibility with Firestore

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
            id_nonunique=data.get("_id_nonunique"),
        )
