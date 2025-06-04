from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from .db.session import create_tables, get_session_context


def create_project_tables(project_name: str, db_session: Session | None = None) -> None:
    """Create any missing SQL tables and setup for a project.

    In SQL, we don't need per-project collections like Firestore,
    but we use this function to ensure all tables exist and are properly set up.

    :param project_name: The name of the project (used for validation/logging)
    :param db_session: Optional database session to use
    """
    with get_session_context(db_session) as session:
        try:
            # Create all tables if they don't exist
            create_tables(session.bind)

            # For SQL, we don't need a separate project record since project_name
            # is embedded in each table record. But we could add a projects table
            # later if needed.

        except Exception as exc:
            session.rollback()
            raise RuntimeError(
                f"Failed to create project tables for {project_name}: {exc}"
            ) from exc


def get_project(project_name: str, db_session: Session | None = None) -> dict[str, Any]:
    """
    Get a project by name.

    For now, this returns basic project info since we don't have a dedicated
    projects table. In the future, we could add a ProjectModel to track metadata.

    :param project_name: The name of the project.
    :param db_session: Optional database session to use
    :return: The project data.
    :raises KeyError: If the project does not exist (has no data).
    """
    with get_session_context(db_session) as session:
        try:
            # Check if project has any data by looking for any records with this
            # project_name. We'll check the users table as a proxy for existence
            query = text("SELECT EXISTS(SELECT 1 FROM users WHERE project_name = :project_name)")
            result = session.execute(query, {"project_name": project_name}).scalar()

            if not result:
                # Check other tables to see if project exists anywhere
                tables_to_check = ["tasks", "subtask_types"]
                project_exists = False

                for table in tables_to_check:
                    try:
                        query = text(
                            f"SELECT EXISTS(SELECT 1 FROM {table} "
                            "WHERE project_name = :project_name)"
                        )
                        result = session.execute(query, {"project_name": project_name}).scalar()
                        if result:
                            project_exists = True
                            break
                    except Exception:  # pylint: disable=broad-exception-caught
                        # Table might not exist yet, skip
                        continue

                if not project_exists:
                    raise KeyError(f"Project {project_name} not found")

            # Return basic project information
            return {
                "project_name": project_name,
                "created_ts": datetime.now(),
                "database_type": "sql",
            }

        except Exception as exc:
            if isinstance(exc, KeyError):
                raise
            raise RuntimeError(f"Failed to get project {project_name}: {exc}") from exc
