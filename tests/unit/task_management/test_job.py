# pylint: disable=redefined-outer-name,unused-argument
import pytest

from zetta_utils.task_management.job import (
    create_job,
    create_jobs_batch,
    get_job,
    update_job,
)
from zetta_utils.task_management.types import Job, JobUpdate


def test_create_job_success(project_name, sample_job, clean_db, db_session):
    result = create_job(project_name=project_name, data=sample_job, db_session=db_session)
    assert result == "job_1"

    job = get_job(project_name=project_name, job_id="job_1", db_session=db_session)
    assert job["batch_id"] == "batch_1"


def test_get_job_success(existing_job, project_name, db_session):
    result = get_job(project_name=project_name, job_id="job_1", db_session=db_session)
    assert result["job_id"] == "job_1"


def test_get_job_not_found(project_name, clean_db, db_session):
    with pytest.raises(KeyError, match="Job job_1 not found"):
        get_job(project_name=project_name, job_id="job_1", db_session=db_session)


def test_update_job_success(clean_db, existing_job, project_name, db_session):
    update_data = JobUpdate(**{"status": "ingested"})

    result = update_job(
        project_name=project_name, job_id="job_1", data=update_data, db_session=db_session
    )
    assert result is True

    job = get_job(project_name=project_name, job_id="job_1", db_session=db_session)
    assert job["status"] == "ingested"


def test_update_job_invalid_status(existing_job, project_name, db_session):
    """Test update_job with invalid status - covers the status validation line"""
    # Create an update with an invalid status value
    update_data = {"status": "invalid_status_value"}

    with pytest.raises(ValueError, match="Invalid status value"):
        update_job(
            project_name=project_name,
            job_id="job_1",
            data=update_data,  # type: ignore
            db_session=db_session,
        )


def test_create_job_validation(project_name, clean_db, db_session):
    # Test with invalid job data that doesn't match the Job type
    invalid_job = {"invalid_field": "value"}
    with pytest.raises(Exception):
        create_job(
            project_name=project_name, data=invalid_job, db_session=db_session  # type: ignore
        )


def test_get_job_not_found_different_project(existing_job, clean_db, db_session):
    with pytest.raises(KeyError, match="Job job_1 not found"):
        get_job(project_name="different_project", job_id="job_1", db_session=db_session)


def test_create_job_duplicate(project_name, clean_db, db_session):
    job_data = Job(
        **{
            "job_id": "duplicate_job",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "job_type": "segmentation",
            "ng_state": "http://example.com/duplicate_job",
        }
    )

    result = create_job(project_name=project_name, data=job_data, db_session=db_session)
    assert result == "duplicate_job"

    # Creating the same job again should succeed (return the job_id)
    result2 = create_job(project_name=project_name, data=job_data, db_session=db_session)
    assert result2 == "duplicate_job"

    job = get_job(project_name=project_name, job_id="duplicate_job", db_session=db_session)
    assert job["batch_id"] == "batch_1"
    assert job["status"] == "pending_ingestion"
    assert job["ng_state"] == "http://example.com/duplicate_job"


def test_get_job_error_case(project_name, clean_db, db_session):
    """Test error case for get_job"""
    with pytest.raises(KeyError, match="Job non_existent_job not found"):
        get_job(project_name=project_name, job_id="non_existent_job", db_session=db_session)


def test_create_job(project_name, clean_db, db_session):
    job_data = Job(
        **{
            "job_id": "job_1",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "job_type": "segmentation",
            "ng_state": "http://example.com/job_1",
        }
    )

    result = create_job(project_name=project_name, data=job_data, db_session=db_session)
    assert result == "job_1"

    job = get_job(project_name=project_name, job_id="job_1", db_session=db_session)
    assert job["batch_id"] == "batch_1"
    assert job["status"] == "pending_ingestion"
    assert job["ng_state"] == "http://example.com/job_1"


def test_update_job(project_name, clean_db, db_session):
    job_data = Job(
        **{
            "job_id": "job_2",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "job_type": "segmentation",
            "ng_state": "http://example.com/job_2",
        }
    )
    create_job(project_name=project_name, data=job_data, db_session=db_session)

    # Update with JobUpdate type
    update_data = JobUpdate(status="ingested", ng_state="http://updated.com")
    update_job(project_name=project_name, job_id="job_2", data=update_data, db_session=db_session)

    updated_job = get_job(project_name=project_name, job_id="job_2", db_session=db_session)
    assert updated_job["status"] == "ingested"
    assert updated_job["ng_state"] == "http://updated.com"


def test_create_jobs_batch_success(project_name, clean_db, db_session):
    """Test creating multiple jobs in batch"""
    jobs = [
        Job(
            **{
                "job_id": "batch_job_1",
                "batch_id": "batch_1",
                "status": "pending_ingestion",
                "job_type": "segmentation",
                "ng_state": "http://example.com/batch_job_1",
            }
        ),
        Job(
            **{
                "job_id": "batch_job_2",
                "batch_id": "batch_1",
                "status": "pending_ingestion",
                "job_type": "segmentation",
                "ng_state": "http://example.com/batch_job_2",
            }
        ),
    ]

    result = create_jobs_batch(project_name=project_name, jobs=jobs, db_session=db_session)
    assert len(result) == 2
    assert "batch_job_1" in result
    assert "batch_job_2" in result

    # Verify jobs were created
    job1 = get_job(project_name=project_name, job_id="batch_job_1", db_session=db_session)
    job2 = get_job(project_name=project_name, job_id="batch_job_2", db_session=db_session)
    assert job1["batch_id"] == "batch_1"
    assert job2["batch_id"] == "batch_1"


def test_create_jobs_batch_conflicting_content(project_name, clean_db, db_session):
    existing_job = Job(
        **{
            "job_id": "job_1",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "job_type": "segmentation",
            "ng_state": "http://example.com/job_1",
        }
    )

    # Create the existing job first
    create_job(project_name=project_name, data=existing_job, db_session=db_session)

    # Try to create a job with the same ID but different content
    conflicting_job = Job(
        **{
            "job_id": "job_1",
            "batch_id": "batch_1",
            "status": "ingested",  # Different status
            "job_type": "segmentation",
            "ng_state": "http://example.com/job_1_different",  # Different ng_state
        }
    )

    with pytest.raises(ValueError, match="Job job_1 already exists with different content"):
        create_jobs_batch(project_name=project_name, jobs=[conflicting_job], db_session=db_session)


def test_update_job_with_different_data_types(project_name, clean_db, db_session):
    """Test updating a job with different field data types"""
    job_data = Job(
        **{
            "job_id": "job_1",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "job_type": "segmentation",
            "ng_state": "http://example.com/job_1",
        }
    )
    create_job(project_name=project_name, data=job_data, db_session=db_session)

    # Update with JobUpdate type
    update_data = JobUpdate(status="ingested", ng_state="http://updated.com")
    update_job(project_name=project_name, job_id="job_1", data=update_data, db_session=db_session)

    updated_job = get_job(project_name=project_name, job_id="job_1", db_session=db_session)
    assert updated_job["status"] == "ingested"
    assert updated_job["ng_state"] == "http://updated.com"


def test_update_job_nonexistent_job(project_name, clean_db, db_session):
    """Test updating a job that doesn't exist"""
    update_data = JobUpdate(status="ingested")

    with pytest.raises(KeyError, match="Job job_1 not found"):
        update_job(
            project_name=project_name, job_id="job_1", data=update_data, db_session=db_session
        )
