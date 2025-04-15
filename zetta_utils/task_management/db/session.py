import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def get_engine(engine_url: Optional[str] = None, use_sqlite: bool = False) -> Engine:
    """
    Get a SQLAlchemy engine for the database.

    Args:
        engine_url: Optional URL for the database
        use_sqlite: If True, use SQLite instead of PostgreSQL

    Returns:
        SQLAlchemy engine
    """
    if use_sqlite:
        # Use in-memory SQLite for testing
        engine_url = "sqlite:///:memory:"
    elif engine_url is None:
        # Database connection settings for PostgreSQL
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


def get_db_session(engine_url: Optional[str] = None, use_sqlite: bool = False) -> Session:
    """
    Get a SQLAlchemy session for the database.

    Args:
        engine_url: Optional URL for the database
        use_sqlite: If True, use SQLite instead of PostgreSQL

    Returns:
        SQLAlchemy session
    """
    engine = get_engine(engine_url, use_sqlite)
    create_tables(engine)
    session_factory = get_session_factory(engine)
    return session_factory()
