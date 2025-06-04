import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def get_engine(engine_url: str | None = None) -> Engine:
    """
    Get a SQLAlchemy engine for the database.
    Args:
        engine_url: Optional URL for the database
    Returns:
        SQLAlchemy engine
    """
    if engine_url is None:
        user = "postgres"
        host = "35.237.17.67"
        port = "5432"
        database = "postgres"
        password = os.getenv("DB_PASSWORD")

        if not password:
            raise ValueError("DB_PASSWORD environment variable not set")

        engine_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

    engine = create_engine(engine_url)
    return engine


def get_session_factory(engine: Engine) -> sessionmaker:
    """
    Create a sessionmaker for the given engine.
    Args:
        engine: SQLAlchemy engine
    Returns:
        SQLAlchemy sessionmaker
    """
    session_factory = sessionmaker(bind=engine)
    return session_factory


def create_tables(engine: Engine) -> None:
    """
    Create all tables in the database.
    Args:
        engine: SQLAlchemy engine
    """
    Base.metadata.create_all(engine)


def is_test_environment() -> bool:
    """
    Check if we're running in a test environment.
    Returns:
        True if running in pytest, False otherwise
    """
    return os.environ.get("PYTEST_CURRENT_TEST") is not None


def get_db_session(engine_url: str | None = None) -> Session:
    """
    Get a SQLAlchemy session for the database.
    Args:
        engine_url: Optional URL for the database
    Returns:
        SQLAlchemy session
    """
    engine = get_engine(engine_url)
    create_tables(engine)
    session_factory = get_session_factory(engine)
    return session_factory()


@contextmanager
def get_session_context(db_session: Session | None = None):
    """
    Context manager for database sessions.

    If db_session is provided, uses it without closing.
    If db_session is None, creates a new session and closes it when done.

    Args:
        db_session: Optional existing session to use

    Yields:
        Session: Database session to use
    """
    if db_session is not None:
        # Use provided session, don't close it
        yield db_session
    else:
        # Create new session and manage its lifecycle
        session = get_db_session()
        try:
            yield session
        finally:
            session.close()
