# pylint: disable=redefined-outer-name,unused-argument
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from testcontainers.postgres import PostgresContainer

from zetta_utils.task_management.db.models import Base
from zetta_utils.task_management.db.session import create_tables, get_session_factory
from zetta_utils.task_management.task import create_task
from zetta_utils.task_management.task_type import create_task_type
from zetta_utils.task_management.types import Task, TaskType, User
from zetta_utils.task_management.user import create_user


@pytest.fixture(scope="session")
def postgres_container():
    """PostgreSQL container for testing"""
    container = PostgresContainer("postgres:15")
    # Configure with more connections for concurrent tests
    container.with_command("-c max_connections=100")
    container.start()
    try:
        yield container
    finally:
        try:
            container.stop()
        except Exception:  # pylint: disable=broad-exception-caught
            pass  # Ignore cleanup errors


@pytest.fixture(scope="session")
def db_engine(postgres_container):
    """Shared database engine for all tests"""
    connection_url = postgres_container.get_connection_url()
    # Use NullPool to avoid connection pooling issues in tests
    engine = create_engine(connection_url, poolclass=NullPool)
    # Don't create tables here - let each test manage its own schema
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(autouse=True)
def ensure_tables(db_engine):
    """Ensure database tables exist before each test"""
    create_tables(db_engine)
    yield


@pytest.fixture
def db_session(db_engine):
    """
    Create a PostgreSQL database session for testing.
    """
    session_factory = get_session_factory(db_engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


# Alias for compatibility with existing tests
@pytest.fixture
def postgres_session(db_engine):
    """PostgreSQL database session - alias for db_session"""
    session_factory = get_session_factory(db_engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def clean_db(db_session):
    """
    Clean the database before and after each test.
    """
    # Clear existing data
    Base.metadata.drop_all(db_session.bind)
    Base.metadata.create_all(db_session.bind)
    db_session.commit()

    yield

    # Clean up after test
    try:
        db_session.rollback()
        Base.metadata.drop_all(db_session.bind)
        db_session.commit()
    except Exception:  # pylint: disable=broad-exception-caught
        # If cleanup fails, just continue
        pass


@pytest.fixture
def project_name():
    """
    Return a test project name.
    """
    return "test_project"


@pytest.fixture
def sample_user() -> User:
    return {
        "user_id": "test_user",
        "hourly_rate": 50.0,
        "active_task": "",
        "qualified_task_types": ["segmentation_proofread"],
    }


@pytest.fixture
def existing_user(clean_db, db_session, project_name, sample_user):
    create_user(project_name=project_name, data=sample_user, db_session=db_session)
    yield sample_user


@pytest.fixture
def sample_task_type() -> TaskType:
    return {
        "task_type": "segmentation_proofread",
        "completion_statuses": ["done", "need_help"],
    }


@pytest.fixture
def existing_task_type(clean_db, db_session, project_name, sample_task_type):
    create_task_type(project_name=project_name, data=sample_task_type, db_session=db_session)
    yield sample_task_type


@pytest.fixture
def sample_tasks() -> list[Task]:
    return [
        {
            "task_id": f"task_{i}",
            "completion_status": "",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": {"url": f"http://example.com/{i}"},
            "ng_state_initial": {"url": f"http://example.com/{i}"},
            "priority": i,
            "batch_id": "batch_1",
            "last_leased_ts": 0.0,
            "is_active": True,
            "task_type": "segmentation_proofread",
        }
        for i in range(1, 4)
    ]


@pytest.fixture
def existing_tasks(clean_db, db_session, project_name, sample_tasks, existing_task_type):
    for task in sample_tasks:
        create_task(project_name=project_name, data=task, db_session=db_session)
    yield sample_tasks


@pytest.fixture
def sample_task(existing_task_type) -> Task:
    return {
        "task_id": "task_1",
        "completion_status": "",
        "assigned_user_id": "",
        "active_user_id": "",
        "completed_user_id": "",
        "ng_state": {"url": "http://example.com"},
        "ng_state_initial": {"url": "http://example.com"},
        "priority": 1,
        "batch_id": "batch_1",
        "task_type": existing_task_type["task_type"],
        "is_active": True,
        "last_leased_ts": 0.0,
    }


@pytest.fixture
def existing_task(clean_db, db_session, project_name, existing_task_type, sample_task):
    create_task(project_name=project_name, data=sample_task, db_session=db_session)
    yield sample_task


@pytest.fixture
def task_factory(db_session, project_name, existing_task_type):
    """Factory fixture to create tasks with custom IDs"""

    def _create_task(task_id: str, **kwargs):
        task_data = Task(
            **{
                "task_id": task_id,
                "completion_status": "",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "ng_state": {"url": f"http://example.com/{task_id}"},
                "ng_state_initial": {"url": f"http://example.com/{task_id}"},
                "priority": 1,
                "batch_id": "batch_1",
                "task_type": existing_task_type["task_type"],
                "is_active": True,
                "last_leased_ts": 0.0,
                **kwargs,  # type: ignore
            }
        )
        create_task(project_name=project_name, data=task_data, db_session=db_session)
        return task_data

    return _create_task
