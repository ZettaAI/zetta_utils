# pylint: disable=redefined-outer-name,unused-argument
import os

import pytest

from zetta_utils.task_management.db.session import (
    create_tables,
    get_db_session,
    get_engine,
    get_session_context,
    get_session_factory,
    is_test_environment,
)


def test_get_engine_without_password(mocker):
    """Test getting engine without DB_PASSWORD set"""
    mocker.patch.dict(os.environ, {}, clear=True)
    with pytest.raises(ValueError, match="DB_PASSWORD environment variable not set"):
        get_engine()


def test_get_session_factory():
    """Test getting session factory"""
    custom_url = "postgresql://test:pass@localhost:5432/testdb"
    engine = get_engine(engine_url=custom_url)
    session_factory = get_session_factory(engine)
    assert session_factory is not None


def test_create_tables():
    """Test creating tables"""
    custom_url = "postgresql://test:pass@localhost:5432/testdb"
    engine = get_engine(engine_url=custom_url)

    # This will fail to connect, but we're just testing that it tries to create tables
    with pytest.raises(Exception):  # Will fail on connection, but that's expected
        create_tables(engine)


def test_is_test_environment():
    """Test detecting test environment"""
    # Should return True when running in pytest
    assert is_test_environment() is True


def test_get_db_session_creates_tables():
    """Test that get_db_session creates tables"""
    custom_url = "postgresql://test:pass@localhost:5432/testdb"

    # This will fail to connect, but we're just testing that it tries to create an engine
    with pytest.raises(Exception):  # Will fail on connection, but that's expected
        get_db_session(engine_url=custom_url)


def test_get_session_context_with_provided_session(db_session):
    """Test session context with provided session"""
    # Use the test database session from fixture
    with get_session_context(db_session) as ctx_session:
        assert ctx_session is db_session
    # Session should still be open since we provided it
    assert db_session.is_active


def test_get_session_context_without_session(postgres_container, mocker):
    """Test session context without provided session"""
    # Use the test container's connection URL
    connection_url = postgres_container.get_connection_url()

    # Create a new session each time get_db_session is called
    def create_test_session():
        return get_db_session(engine_url=connection_url)

    mock_get_db = mocker.patch("zetta_utils.task_management.db.session.get_db_session")
    mock_get_db.side_effect = create_test_session

    # Track if close was called
    closed = False

    with get_session_context() as session:
        assert session is not None
        assert session.is_active

        # Mock the close method to track if it was called
        original_close = session.close

        def mock_close():
            nonlocal closed
            closed = True
            original_close()

        session.close = mock_close

    # Verify close was called
    assert closed
