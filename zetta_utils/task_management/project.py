from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session
from typeguard import typechecked

from .db.models import ProjectModel
from .db.session import create_tables, get_session_context


@typechecked
def create_project(
    *,
    project_name: str,
    description: str = "",
    db_session: Session | None = None,
) -> str:
    """
    :param project_name: The name of the project
    :param description: Optional project description
    :param db_session: Optional database session
    :return: The project name
    :raises ValueError: If project already exists
    """

    with get_session_context(db_session) as session:
        try:
            existing = session.execute(
                select(ProjectModel).where(ProjectModel.project_name == project_name)
            ).first()

            if existing:
                raise ValueError(f"Project '{project_name}' already exists")

            project = ProjectModel(
                project_name=project_name,
                description=description,
                created_at=datetime.now().isoformat(),
                status="active",
            )

            session.add(project)
            session.commit()

            return project_name

        except Exception:  # pylint: disable=broad-exception-caught # pragma: no cover
            session.rollback()
            raise


@typechecked
def list_all_projects(
    *,
    db_session: Session | None = None,
) -> list[str]:
    """
    :param db_session: Optional database session
    :return: List of project names sorted alphabetically
    """

    with get_session_context(db_session) as session:
        result = session.execute(
            select(ProjectModel.project_name)
            .where(ProjectModel.status == "active")
            .order_by(ProjectModel.project_name)
        ).fetchall()

        return [row[0] for row in result]


@typechecked
def get_project(
    *,
    project_name: str,
    db_session: Session | None = None,
) -> dict:
    """
    :param project_name: The name of the project
    :param db_session: Optional database session
    :return: Project details dictionary
    :raises KeyError: If project doesn't exist
    """

    with get_session_context(db_session) as session:
        result = session.execute(
            select(ProjectModel).where(ProjectModel.project_name == project_name)
        ).first()

        if not result:
            raise KeyError(f"Project '{project_name}' not found")

        return result[0].to_dict()


@typechecked
def delete_project(
    *,
    project_name: str,
    db_session: Session | None = None,
) -> bool:
    """
    Note: This only removes from projects registry.
    Project data in other tables remains.

    :param project_name: The name of the project to delete
    :param db_session: Optional database session
    :return: True if deleted, False if didn't exist
    """

    with get_session_context(db_session) as session:
        try:
            result = session.execute(
                select(ProjectModel).where(ProjectModel.project_name == project_name)
            ).first()

            if not result:
                return False

            session.delete(result[0])
            session.commit()

            return True

        except Exception:  # pylint: disable=broad-exception-caught # pragma: no cover
            session.rollback()
            raise


@typechecked
def project_exists(
    *,
    project_name: str,
    db_session: Session | None = None,
) -> bool:
    """
    :param project_name: The name of the project
    :param db_session: Optional database session
    :return: True if project exists, False otherwise
    """

    with get_session_context(db_session) as session:
        result = session.execute(
            select(ProjectModel.project_name).where(ProjectModel.project_name == project_name)
        ).first()

        return result is not None


@typechecked
def create_project_tables(
    *,
    project_name: str,
    db_session: Session | None = None,
) -> None:
    """
    Create all project-specific tables for the given project.

    :param project_name: The name of the project
    :param db_session: Optional database session (uses default session if None)
    """
    with get_session_context(db_session) as session:
        try:
            create_project(project_name=project_name)
        except ValueError as e:
            if "already exists" in str(e):
                # Project already exists, that's fine - just ensure tables exist
                pass
            else:
                raise
        try:
            create_tables(session.bind)
        except Exception:  # pylint: disable=broad-exception-caught # pragma: no cover
            session.rollback()
            raise
