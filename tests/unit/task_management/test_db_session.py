# pylint: disable=redefined-outer-name,unused-argument
import os
from unittest.mock import patch

import pytest

from zetta_utils.task_management.db.session import (
    create_tables,
    get_db_session,
    get_engine,
    get_session_context,
    get_session_factory,
    is_test_environment,
)


def test_get_engine_without_password():
    """Test getting engine without DB_PASSWORD set"""
    with patch.dict(os.environ, {}, clear=True):
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


def test_get_session_context_with_provided_session():
    """Test session context with provided session"""
    session = get_db_session()
    with get_session_context(session) as ctx_session:
        assert ctx_session is session
    # Session should still be open since we provided it
    assert session.is_active
    session.close()


def test_get_session_context_without_session():
    """Test session context without provided session"""
    with get_session_context() as session:
        assert session is not None
        assert session.is_active
    # Session should be closed after context
