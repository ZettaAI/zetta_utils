# pylint: disable=redefined-outer-name,unused-argument
import pytest

from zetta_utils.task_management.ingestion import ingest_batch, ingest_job
from zetta_utils.task_management.job import create_job, get_job, update_job
from zetta_utils.task_management.task import create_task
from zetta_utils.task_management.types import Job, Task


def sample_tasks(existing_task_type) -> list[Task]:
    return [
        Task(
            **{
                "job_id": f"job_{i}",
                "task_id": f"task_{i}",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "ng_state": {"url": f"http://example.com/{i}"},
                "ng_state_initial": {"url": f"http://example.com/{i}"},
                "priority": i,
                "batch_id": "batch_1",
                "task_type": existing_task_type["task_type"],
                "is_active": True,
                "last_leased_ts": 0.0,
                "completion_status": "",
            }
        )
        for i in range(1, 4)
    ]


@pytest.fixture
def existing_tasks(clean_db, project_name, db_session, sample_tasks):
    for task in sample_tasks:
        create_task(db_session=db_session, project_name=project_name, data=task)
    yield sample_tasks


def test_ingest_job(clean_db, project_name, db_session, existing_job, existing_task_type):
    """Test ingesting a task"""
    result = ingest_job(
        project_name=project_name,
        job_id="job_1",
        task_structure="segmentation_proofread_1pass",
        priority=2,
        task_structure_kwargs={},
        db_session=db_session,
    )
    assert result is True

    job = get_job(project_name=project_name, job_id="job_1", db_session=db_session)
    assert job["status"] == "ingested"


def test_ingest_job_already_ingested(
    clean_db, project_name, db_session, existing_job, existing_task_type
):
    """Test that ingesting an already ingested task returns False"""
    ingest_job(
        project_name=project_name,
        job_id="job_1",
        task_structure="segmentation_proofread_1pass",
        priority=2,
        task_structure_kwargs={},
        db_session=db_session,
    )

    result = ingest_job(
        project_name=project_name,
        job_id="job_1",
        task_structure="segmentation_proofread_1pass",
        priority=2,
        task_structure_kwargs={},
        db_session=db_session,
    )
    assert result is False


def test_ingest_job_re_ingest(
    clean_db, project_name, db_session, existing_job, existing_task_type
):
    """Test re-ingesting a task"""
    ingest_job(
        project_name=project_name,
        job_id="job_1",
        task_structure="segmentation_proofread_1pass",
        priority=2,
        task_structure_kwargs={},
        db_session=db_session,
    )

    result = ingest_job(
        project_name=project_name,
        job_id="job_1",
        task_structure="segmentation_proofread_1pass",
        re_ingest="not_processed",
        priority=2,
        task_structure_kwargs={},
        db_session=db_session,
    )
    assert result is True


def test_ingest_batch(clean_db, project_name, db_session, existing_task_type):
    """Test ingesting a batch of jobs"""
    jobs = [
        Job(
            **{
                "job_id": f"job_{i}",
                "batch_id": "batch_1",
                "status": "pending_ingestion",
                "job_type": "segmentation",
                "ng_state": {"url": f"http://example.com/job_{i}"},
            }
        )
        for i in range(1, 4)
    ]

    for job in jobs:
        create_job(project_name=project_name, data=job, db_session=db_session)

    result = ingest_batch(
        project_name=project_name,
        batch_id="batch_1",
        task_structure="segmentation_proofread_1pass",
        task_structure_kwargs={},
        db_session=db_session,
    )
    assert result is True

    for i in range(1, 4):
        job = get_job(project_name=project_name, job_id=f"job_{i}", db_session=db_session)
        assert job["status"] == "ingested"


def test_ingest_batch_re_ingest(clean_db, project_name, db_session, existing_task_type):
    jobs = [
        Job(
            **{
                "job_id": f"job_{i}",
                "batch_id": "batch_1",
                "status": "pending_ingestion",
                "job_type": "segmentation",
                "ng_state": {"url": f"http://example.com/job_{i}"},
            }
        )
        for i in range(1, 4)
    ]

    for job in jobs:
        create_job(project_name=project_name, data=job, db_session=db_session)

    ingest_batch(
        project_name=project_name,
        batch_id="batch_1",
        task_structure="segmentation_proofread_1pass",
        task_structure_kwargs={},
        db_session=db_session,
    )

    update_job(
        project_name=project_name,
        job_id="job_3",
        data={"status": "fully_processed"},
        db_session=db_session,
    )

    result = ingest_batch(
        project_name=project_name,
        batch_id="batch_1",
        task_structure="segmentation_proofread_1pass",
        re_ingest="not_processed",
        task_structure_kwargs={},
        db_session=db_session,
    )
    assert result is True
    # Check that job_1 and job_2 were re-ingested (they were in 'ingested' state)
    job_1 = get_job(project_name=project_name, job_id="job_1", db_session=db_session)
    job_2 = get_job(project_name=project_name, job_id="job_2", db_session=db_session)
    assert job_1["status"] == "ingested"
    assert job_2["status"] == "ingested"

    # Check that job_3 was not re-ingested (it was in 'fully_processed' state)
    job_3 = get_job(project_name=project_name, job_id="job_3", db_session=db_session)
    assert job_3["status"] == "fully_processed"

    # Test re-ingesting all jobs including fully_processed
    result = ingest_batch(
        project_name=project_name,
        batch_id="batch_1",
        task_structure="segmentation_proofread_1pass",
        re_ingest="all",
        task_structure_kwargs={},
        db_session=db_session,
    )
    assert result is True

    # Verify all jobs were re-ingested including job_3
    job_3 = get_job(project_name=project_name, job_id="job_3", db_session=db_session)
    assert job_3["status"] == "ingested"


def test_ingest_job_nonexistent(clean_db, project_name, db_session):
    """Test that ingesting a nonexistent job raises KeyError"""
    with pytest.raises(KeyError, match="Jobs not found: job_nonexistent"):
        ingest_job(
            project_name=project_name,
            job_id="job_nonexistent",
            task_structure="segmentation_proofread_1pass",
            priority=2,
            task_structure_kwargs={},
            db_session=db_session,
        )


def test_ingest_job_fully_processed_no_reingest(
    clean_db, project_name, db_session, existing_task_type
):
    """Test that ingesting a fully processed job without re_ingest returns False"""
    job_data = Job(
        **{
            "job_id": "fully_processed_job",
            "batch_id": "batch_1",
            "status": "fully_processed",
            "job_type": "segmentation",
            "ng_state": {"url": "http://example.com/fully_processed_job"},
        }
    )
    create_job(project_name=project_name, data=job_data, db_session=db_session)

    result = ingest_job(
        project_name=project_name,
        job_id="fully_processed_job",
        task_structure="segmentation_proofread_1pass",
        priority=2,
        task_structure_kwargs={},
        db_session=db_session,
    )
    assert result is False

    result = ingest_job(
        project_name=project_name,
        job_id="fully_processed_job",
        task_structure="segmentation_proofread_1pass",
        re_ingest="not_processed",
        priority=2,
        task_structure_kwargs={},
        db_session=db_session,
    )
    assert result is False

    result = ingest_job(
        project_name=project_name,
        job_id="fully_processed_job",
        task_structure="segmentation_proofread_1pass",
        re_ingest="all",
        priority=2,
        task_structure_kwargs={},
        db_session=db_session,
    )
    assert result is True


def test_ingest_batch_no_tasks(clean_db, project_name, db_session):
    """Test that ingesting a batch with no jobs returns False"""
    result = ingest_batch(
        project_name=project_name,
        batch_id="nonexistent_batch",
        task_structure="segmentation_proofread_1pass",
        priority=2,
        task_structure_kwargs={},
        db_session=db_session,
    )
    assert result is False


def test_ingest_batch_no_matching_jobs(clean_db, project_name, db_session):
    """Test that ingesting a batch with no matching jobs returns False"""
    for i in range(1, 4):
        job_data = Job(
            **{
                "job_id": f"job_{i}",
                "batch_id": "batch_1",
                "status": "pending_ingestion",
                "job_type": "segmentation",
                "ng_state": {"url": f"http://example.com/job_{i}"},
            }
        )
        create_job(project_name=project_name, data=job_data, db_session=db_session)

    result = ingest_batch(
        project_name=project_name,
        batch_id="batch_2",
        task_structure="segmentation_proofread_1pass",
        priority=2,
        task_structure_kwargs={},
        db_session=db_session,
    )
    assert result is False


def test_ingest_jobs_database_failure(
    clean_db, project_name, db_session, existing_job, existing_task_type, mocker
):
    """Test that ingestion handles database failures gracefully"""
    # Mock the commit to raise an exception
    mock_commit = mocker.patch.object(db_session, "commit")
    mock_rollback = mocker.patch.object(db_session, "rollback")
    mock_commit.side_effect = Exception("Database connection lost")

    with pytest.raises(
        RuntimeError, match="Failed to ingest job bundle: Database connection lost"
    ):
        ingest_job(
            project_name=project_name,
            job_id="job_1",
            task_structure="segmentation_proofread_1pass",
            priority=2,
            task_structure_kwargs={},
            db_session=db_session,
        )

    # Verify rollback was called
    mock_rollback.assert_called_once()
