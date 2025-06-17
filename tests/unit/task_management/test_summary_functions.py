# pylint: disable=redefined-outer-name,unused-argument
"""Tests for summary functions in task and job modules."""

import pytest

from zetta_utils.task_management.job import create_job, list_jobs_summary
from zetta_utils.task_management.task import create_task, list_tasks_summary
from zetta_utils.task_management.task_type import create_task_type
from zetta_utils.task_management.types import Job, Task, TaskType


def test_list_tasks_summary(clean_db, postgres_session, project_name):
    """Test the list_tasks_summary function."""
    # Create a task type
    create_task_type(
        project_name=project_name,
        data=TaskType(task_type="test_type", completion_statuses=["done", "failed"]),
        db_session=postgres_session,
    )
    
    # Create a job
    create_job(
        project_name=project_name,
        data=Job(
            job_id="test_job",
            batch_id="test_batch",
            status="ingested",
            job_type="test",
            ng_state={"test": "state"},
        ),
        db_session=postgres_session,
    )
    
    # Create some active incomplete tasks
    for i in range(3):
        create_task(
            project_name=project_name,
            data=Task(
                task_id=f"active_incomplete_{i}",
                job_id="test_job",
                task_type="test_type",
                priority=i,
                batch_id="test_batch",
                ng_state={"test": "state"},
                ng_state_initial={"test": "state"},
                completion_status="",
                assigned_user_id="",
                active_user_id="",
                completed_user_id="",
                last_leased_ts=0.0,
                is_active=True,
                is_paused=False,
            ),
            db_session=postgres_session,
        )
    
    # Create some active completed tasks
    for i in range(2):
        create_task(
            project_name=project_name,
            data=Task(
                task_id=f"active_complete_{i}",
                job_id="test_job",
                task_type="test_type",
                priority=i,
                batch_id="test_batch",
                ng_state={"test": "state"},
                ng_state_initial={"test": "state"},
                completion_status="done",
                assigned_user_id="",
                active_user_id="",
                completed_user_id="user1",
                last_leased_ts=0.0,
                is_active=True,
                is_paused=False,
            ),
            db_session=postgres_session,
        )
    
    # Create some paused tasks
    for i in range(1):
        create_task(
            project_name=project_name,
            data=Task(
                task_id=f"paused_{i}",
                job_id="test_job",
                task_type="test_type",
                priority=i,
                batch_id="test_batch",
                ng_state={"test": "state"},
                ng_state_initial={"test": "state"},
                completion_status="",
                assigned_user_id="",
                active_user_id="",
                completed_user_id="",
                last_leased_ts=0.0,
                is_active=True,
                is_paused=True,
            ),
            db_session=postgres_session,
        )
    
    # Get summary
    summary = list_tasks_summary(project_name=project_name, db_session=postgres_session)
    
    # Check counts
    assert summary["active_count"] == 3
    assert summary["completed_count"] == 2
    assert summary["paused_count"] == 1
    
    # Check sample IDs
    assert len(summary["active_unpaused_ids"]) == 3
    assert len(summary["active_paused_ids"]) == 1
    
    # Verify IDs are correct
    assert all(task_id.startswith("active_incomplete_") for task_id in summary["active_unpaused_ids"])
    assert all(task_id.startswith("paused_") for task_id in summary["active_paused_ids"])


def test_list_jobs_summary(clean_db, postgres_session, project_name):
    """Test the list_jobs_summary function."""
    # Create jobs with different statuses
    jobs_data = [
        ("pending_1", "pending_ingestion"),
        ("pending_2", "pending_ingestion"),
        ("pending_3", "pending_ingestion"),
        ("ingested_1", "ingested"),
        ("ingested_2", "ingested"),
        ("completed_1", "fully_processed"),
        ("completed_2", "fully_processed"),
    ]
    
    for job_id, status in jobs_data:
        create_job(
            project_name=project_name,
            data=Job(
                job_id=job_id,
                batch_id="test_batch",
                status=status,
                job_type="test",
                ng_state={"test": "state"},
            ),
            db_session=postgres_session,
        )
    
    # Get summary
    summary = list_jobs_summary(project_name=project_name, db_session=postgres_session)
    
    # Check counts
    assert summary["pending_ingestion_count"] == 3
    assert summary["ingested_count"] == 2
    assert summary["completed_count"] == 2
    
    # Check sample IDs (should return up to 5)
    assert len(summary["pending_ingestion_ids"]) == 3
    assert len(summary["ingested_ids"]) == 2
    
    # Verify IDs are correct
    assert all(job_id.startswith("pending_") for job_id in summary["pending_ingestion_ids"])
    assert all(job_id.startswith("ingested_") for job_id in summary["ingested_ids"])


def test_list_tasks_summary_empty_project(clean_db, postgres_session, project_name):
    """Test list_tasks_summary with no tasks."""
    # Get summary for empty project
    summary = list_tasks_summary(project_name=project_name, db_session=postgres_session)
    
    # All counts should be 0
    assert summary["active_count"] == 0
    assert summary["completed_count"] == 0 
    assert summary["paused_count"] == 0
    
    # ID lists should be empty
    assert summary["active_unpaused_ids"] == []
    assert summary["active_paused_ids"] == []


def test_list_jobs_summary_empty_project(clean_db, postgres_session, project_name):
    """Test list_jobs_summary with no jobs."""
    # Get summary for empty project
    summary = list_jobs_summary(project_name=project_name, db_session=postgres_session)
    
    # All counts should be 0
    assert summary["pending_ingestion_count"] == 0
    assert summary["ingested_count"] == 0
    assert summary["completed_count"] == 0
    
    # ID lists should be empty
    assert summary["pending_ingestion_ids"] == []
    assert summary["ingested_ids"] == []