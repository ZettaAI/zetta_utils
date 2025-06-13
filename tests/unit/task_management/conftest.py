# pylint: disable=redefined-outer-name,unused-argument
import pytest
from testcontainers.postgres import PostgresContainer

from zetta_utils.task_management.db import get_db_session
from zetta_utils.task_management.db.models import Base
from zetta_utils.task_management.job import create_job
from zetta_utils.task_management.subtask import create_subtask
from zetta_utils.task_management.subtask_type import create_subtask_type
from zetta_utils.task_management.types import Job, Subtask, SubtaskType, User
from zetta_utils.task_management.user import create_user


@pytest.fixture(scope="session")
def postgres_container():
    """PostgreSQL container for testing"""
    container = PostgresContainer("postgres:15")
    container.start()
    try:
        yield container
    finally:
        try:
            container.stop()
        except Exception:  # pylint: disable=broad-exception-caught
            pass  # Ignore cleanup errors


@pytest.fixture
def db_session(postgres_container):
    """
    Create a PostgreSQL database session for testing.
    """
    connection_url = postgres_container.get_connection_url()
    session = get_db_session(engine_url=connection_url)
    try:
        yield session
    finally:
        session.close()


# Alias for compatibility with existing tests
@pytest.fixture
def postgres_session(postgres_container):
    """PostgreSQL database session - alias for db_session"""
    connection_url = postgres_container.get_connection_url()
    session = get_db_session(engine_url=connection_url)
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
        "active_subtask": "",
        "qualified_subtask_types": ["segmentation_proofread"],
    }


@pytest.fixture
def existing_user(clean_db, db_session, project_name, sample_user):
    create_user(project_name=project_name, data=sample_user, db_session=db_session)
    yield sample_user


@pytest.fixture
def sample_job() -> Job:
    return Job(
        **{
            "job_id": "job_1",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "job_type": "segmentation",
            "ng_state": "http://example.com/job_1",
        }
    )


@pytest.fixture
def existing_job(db_session, project_name, sample_job):
    """Fixture that creates a job in the database"""
    create_job(project_name=project_name, data=sample_job, db_session=db_session)
    return sample_job


@pytest.fixture
def sample_subtask_type() -> SubtaskType:
    return {
        "subtask_type": "segmentation_proofread",
        "completion_statuses": ["done", "need_help"],
    }


@pytest.fixture
def existing_subtask_type(clean_db, db_session, project_name, sample_subtask_type):
    create_subtask_type(project_name=project_name, data=sample_subtask_type, db_session=db_session)
    yield sample_subtask_type


@pytest.fixture
def sample_subtasks() -> list[Subtask]:
    return [
        {
            "job_id": "job_1",
            "subtask_id": f"subtask_{i}",
            "completion_status": "",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": f"http://example.com/{i}",
            "ng_state_initial": f"http://example.com/{i}",
            "priority": i,
            "batch_id": "batch_1",
            "last_leased_ts": 0.0,
            "is_active": True,
            "subtask_type": "segmentation_proofread",
        }
        for i in range(1, 4)
    ]


@pytest.fixture
def existing_subtasks(
    clean_db, db_session, project_name, sample_subtasks, existing_subtask_type, existing_job
):
    # Job is already created by existing_job fixture
    for subtask in sample_subtasks:
        create_subtask(project_name=project_name, data=subtask, db_session=db_session)
    yield sample_subtasks


@pytest.fixture
def sample_subtask(existing_subtask_type) -> Subtask:
    return {
        "job_id": "job_1",
        "subtask_id": "subtask_1",
        "completion_status": "",
        "assigned_user_id": "",
        "active_user_id": "",
        "completed_user_id": "",
        "ng_state": "http://example.com",
        "ng_state_initial": "http://example.com",
        "priority": 1,
        "batch_id": "batch_1",
        "subtask_type": existing_subtask_type["subtask_type"],
        "is_active": True,
        "last_leased_ts": 0.0,
    }


@pytest.fixture
def existing_subtask(
    clean_db, db_session, project_name, existing_subtask_type, sample_subtask, existing_job
):
    # Job is already created by existing_job fixture
    create_subtask(project_name=project_name, data=sample_subtask, db_session=db_session)
    yield sample_subtask


@pytest.fixture
def job_factory(db_session, project_name):
    """Factory fixture to create jobs with custom IDs"""

    def _create_job(job_id: str, batch_id: str | None = None, status: str = "ingested"):
        if batch_id is None:
            batch_id = job_id.replace("job_", "batch_")

        job_data = Job(
            **{
                "job_id": job_id,
                "batch_id": batch_id,
                "status": status,
                "job_type": "segmentation",
                "ng_state": f"http://example.com/{job_id}",
            }
        )
        create_job(project_name=project_name, data=job_data, db_session=db_session)
        return job_data

    return _create_job


@pytest.fixture
def subtask_factory(db_session, project_name, existing_subtask_type):
    """Factory fixture to create subtasks with custom IDs"""

    def _create_subtask(job_id: str, subtask_id: str, **kwargs):
        subtask_data = Subtask(
            **{
                "job_id": job_id,
                "subtask_id": subtask_id,
                "completion_status": "",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "ng_state": f"http://example.com/{subtask_id}",
                "ng_state_initial": f"http://example.com/{subtask_id}",
                "priority": 1,
                "batch_id": job_id.replace("job_", "batch_"),
                "subtask_type": existing_subtask_type["subtask_type"],
                "is_active": True,
                "last_leased_ts": 0.0,
                **kwargs,  # type: ignore
            }
        )
        create_subtask(project_name=project_name, data=subtask_data, db_session=db_session)
        return subtask_data

    return _create_subtask
