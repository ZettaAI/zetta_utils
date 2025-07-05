"""Functions to clear all data from a project."""

from sqlalchemy import delete
from sqlalchemy.orm import Session

from zetta_utils.log import get_logger
from zetta_utils.task_management.db.models import (
    DependencyModel,
    TaskModel,
    TaskTypeModel,
    TimesheetModel,
    UserModel,
)
from zetta_utils.task_management.db.session import get_session_context

logger = get_logger(__name__)


def clear_project_data(
    *,
    project_name: str,
    db_session: Session | None = None,
) -> dict[str, int]:
    """
    Clear all data for a project including tasks, dependencies, and timesheets.

    WARNING: This is a destructive operation that cannot be undone!

    Args:
        project_name: Name of the project to clear
        db_session: Optional database session

    Returns:
        Dictionary with counts of deleted records
    """
    with get_session_context(db_session) as session:
        deleted_counts = {}

        try:
            # Delete in order to respect foreign key constraints

            # 1. Delete timesheets first (references tasks)
            timesheet_result = session.execute(
                delete(TimesheetModel).where(TimesheetModel.project_name == project_name)
            )
            deleted_counts["timesheets"] = timesheet_result.rowcount

            # 2. Delete dependencies (references tasks)
            dependency_result = session.execute(
                delete(DependencyModel).where(DependencyModel.project_name == project_name)
            )
            deleted_counts["dependencies"] = dependency_result.rowcount

            # 3. Delete tasks
            task_result = session.execute(
                delete(TaskModel).where(TaskModel.project_name == project_name)
            )
            deleted_counts["tasks"] = task_result.rowcount

            # Commit all deletions
            session.commit()

            logger.info(f"Successfully cleared all data for project '{project_name}'")
            logger.info(f"Deleted counts: {deleted_counts}")

            return deleted_counts

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to clear project data: {e}")
            raise


def clear_project_users(
    *,
    project_name: str,
    db_session: Session | None = None,
) -> int:
    """
    Clear all users for a project.

    Args:
        project_name: Name of the project
        db_session: Optional database session

    Returns:
        Number of users deleted
    """
    with get_session_context(db_session) as session:
        try:
            result = session.execute(
                delete(UserModel).where(UserModel.project_name == project_name)
            )
            count = result.rowcount
            session.commit()

            logger.info(f"Deleted {count} users from project '{project_name}'")
            return count

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to clear users: {e}")
            raise


def clear_project_task_types(
    *,
    project_name: str,
    db_session: Session | None = None,
) -> int:
    """
    Clear all task types for a project.

    Args:
        project_name: Name of the project
        db_session: Optional database session

    Returns:
        Number of task types deleted
    """
    with get_session_context(db_session) as session:
        try:
            result = session.execute(
                delete(TaskTypeModel).where(TaskTypeModel.project_name == project_name)
            )
            count = result.rowcount
            session.commit()

            logger.info(f"Deleted {count} task types from project '{project_name}'")
            return count

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to clear task types: {e}")
            raise
