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
