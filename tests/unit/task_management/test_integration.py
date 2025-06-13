# pylint: disable=redefined-outer-name,unused-argument
import time

import pytest
from sqlalchemy import select

from zetta_utils.task_management import task
from zetta_utils.task_management.db.models import TaskModel
from zetta_utils.task_management.ingestion import ingest_batch
from zetta_utils.task_management.job import create_jobs_batch
from zetta_utils.task_management.project import create_project_tables
from zetta_utils.task_management.task_type import create_task_type
from zetta_utils.task_management.timesheet import submit_timesheet
from zetta_utils.task_management.types import Job, TaskType, User
from zetta_utils.task_management.user import create_user


@pytest.fixture
def short_idle_timeout():
    """Set a short idle timeout for testing takeover functionality"""
    original_timeout = task._MAX_IDLE_SECONDS  # pylint: disable=protected-access
    task._MAX_IDLE_SECONDS = 5  # pylint: disable=protected-access
    yield
    task._MAX_IDLE_SECONDS = original_timeout  # pylint: disable=protected-access


@pytest.fixture
def test_environment(clean_db, db_session):
    """Set up test environment with required task types and structures"""
    project_name = "test_project"

    print("Setting up SQL test environment")

    # Create project tables
    create_project_tables(project_name=project_name, db_session=db_session)
    print("Created project tables")

    # Create required task types
    proofread_type = TaskType(
        task_type="segmentation_proofread",
        completion_statuses=["done", "need_help"],
        description="Proofread segmentation",
    )
    verify_type = TaskType(
        task_type="segmentation_verify",
        completion_statuses=["done", "reject"],
        description="Verify segmentation",
    )
    expert_type = TaskType(
        task_type="segmentation_proofread_expert",
        completion_statuses=["done"],
        description="Expert proofread",
    )

    for task_type in [proofread_type, verify_type, expert_type]:
        create_task_type(project_name=project_name, data=task_type, db_session=db_session)

    yield project_name, db_session


def create_test_jobs(project_name: str, num_jobs: int, db_session) -> list[str]:
    """Create test jobs and return their IDs"""
    jobs = []
    for i in range(num_jobs):
        job_data = Job(
            job_id=f"test_job_{i}",
            batch_id="test_batch",
            status="pending_ingestion",
            job_type="test",
            ng_state={"url": f"http://test.com/job_{i}"},
        )
        jobs.append(job_data)

    return create_jobs_batch(
        project_name=project_name, jobs=jobs, batch_size=100, db_session=db_session
    )


def test_basic_job_workflow(test_environment, short_idle_timeout):
    """Test basic job workflow: create jobs -> ingest -> assign -> complete"""
    project_name, db_session = test_environment
    num_jobs = 5

    # Create a test user
    user_data = User(
        user_id="test_user",
        hourly_rate=20.0,
        active_task="",
        qualified_task_types=["segmentation_proofread"],
    )
    create_user(project_name=project_name, data=user_data, db_session=db_session)
    print("Created test user")

    # Create test jobs
    create_test_jobs(project_name, num_jobs, db_session)
    success = ingest_batch(
        project_name=project_name,
        batch_id="test_batch",
        task_structure="segmentation_proofread_simple_1pass",
        priority=1,
        task_structure_kwargs={},
        db_session=db_session,
    )
    assert success, "Failed to ingest jobs"
    print(f"Ingested {num_jobs} jobs")

    # Verify jobs were created using SQL queries
    query = (
        select(TaskModel)
        .where(TaskModel.project_name == project_name)
        .where(TaskModel.task_type == "segmentation_proofread")
    )
    all_tasks = db_session.execute(query).scalars().all()

    assert len(all_tasks) == num_jobs, f"Expected {num_jobs} tasks, got {len(all_tasks)}"
    print(f"Verified {len(all_tasks)} tasks were created")

    # Test task assignment and completion
    completed_tasks = []
    for i in range(num_jobs):
        # Start a task
        task_id = task.start_task(
            project_name=project_name, user_id="test_user", db_session=db_session
        )
        assert task_id is not None, f"Failed to get task {i}"
        print(f"Started task {task_id}")

        # Submit a timesheet
        submit_timesheet(
            project_name=project_name,
            user_id="test_user",
            duration_seconds=10.0,
            task_id=task_id,
            db_session=db_session,
        )
        print(f"Submitted timesheet for {task_id}")

        # Complete the task
        success = task.release_task(
            project_name=project_name,
            user_id="test_user",
            task_id=task_id,
            completion_status="done",
            db_session=db_session,
        )
        assert success, f"Failed to complete task {task_id}"
        print(f"Completed task {task_id}")
        completed_tasks.append(task_id)

    # Verify all tasks are completed
    completed_query = (
        select(TaskModel)
        .where(TaskModel.project_name == project_name)
        .where(TaskModel.completion_status == "done")
    )
    completed_in_db = db_session.execute(completed_query).scalars().all()

    assert (
        len(completed_in_db) == num_jobs
    ), f"Expected {num_jobs} completed tasks, got {len(completed_in_db)}"
    print(f"Verified all {num_jobs} tasks were completed")


def test_task_idle_takeover(test_environment, short_idle_timeout):
    """Test that idle tasks can be taken over by another user"""
    project_name, db_session = test_environment

    # Create two users
    user1_data = User(
        user_id="user1",
        hourly_rate=20.0,
        active_task="",
        qualified_task_types=["segmentation_proofread"],
    )
    user2_data = User(
        user_id="user2",
        hourly_rate=20.0,
        active_task="",
        qualified_task_types=["segmentation_proofread"],
    )
    create_user(project_name=project_name, data=user1_data, db_session=db_session)
    create_user(project_name=project_name, data=user2_data, db_session=db_session)

    # Create and ingest a job
    create_test_jobs(project_name, 1, db_session)
    ingest_batch(
        project_name=project_name,
        batch_id="test_batch",
        task_structure="segmentation_proofread_simple_1pass",
        priority=1,
        task_structure_kwargs={},
        db_session=db_session,
    )

    # User1 starts the task
    task_id = task.start_task(
        project_name=project_name, user_id="user1", db_session=db_session
    )
    assert task_id is not None, "User1 should be able to start the task"
    print(f"User1 started task {task_id}")

    # Wait for it to become idle
    time.sleep(6)  # Wait longer than the 5-second idle timeout

    # User2 should be able to take over the idle task
    taken_task_id = task.start_task(
        project_name=project_name, user_id="user2", db_session=db_session
    )
    assert taken_task_id == task_id, "User2 should take over the idle task"
    print(f"User2 took over task {taken_task_id}")

    # Verify the task is now assigned to user2
    task_data = task.get_task(
        project_name=project_name, task_id=task_id, db_session=db_session
    )
    assert task_data["active_user_id"] == "user2", "Task should be assigned to user2"
    print("Verified task takeover worked correctly")


def test_multiple_users_concurrent_workflow(test_environment):
    """Test multiple users working on different tasks concurrently"""
    project_name, db_session = test_environment
    num_users = 3
    num_jobs = 6

    # Create multiple users
    for i in range(num_users):
        user_data = User(
            user_id=f"worker_{i}",
            hourly_rate=25.0,
            active_task="",
            qualified_task_types=["segmentation_proofread"],
        )
        create_user(project_name=project_name, data=user_data, db_session=db_session)

    # Create and ingest jobs
    create_test_jobs(project_name, num_jobs, db_session)
    ingest_batch(
        project_name=project_name,
        batch_id="test_batch",
        task_structure="segmentation_proofread_simple_1pass",
        priority=1,
        task_structure_kwargs={},
        db_session=db_session,
    )

    # Each user picks up and completes tasks
    completed_by_user: dict[str, list[str]] = {}
    for user_idx in range(num_users):
        user_id = f"worker_{user_idx}"
        completed_by_user[user_id] = []

        # Each user completes 2 jobs
        for _ in range(2):
            task_id = task.start_task(
                project_name=project_name, user_id=user_id, db_session=db_session
            )
            if task_id:
                print(f"{user_id} got task {task_id}")

                # Submit timesheet and complete
                submit_timesheet(
                    project_name=project_name,
                    user_id=user_id,
                    duration_seconds=15.0,
                    task_id=task_id,
                    db_session=db_session,
                )

                task.release_task(
                    project_name=project_name,
                    user_id=user_id,
                    task_id=task_id,
                    completion_status="done",
                    db_session=db_session,
                )

                completed_by_user[user_id].append(task_id)
                print(f"{user_id} completed task {task_id}")

    # Verify all jobs are completed
    completed_query = (
        select(TaskModel)
        .where(TaskModel.project_name == project_name)
        .where(TaskModel.completion_status == "done")
    )
    completed_tasks = db_session.execute(completed_query).scalars().all()

    assert (
        len(completed_tasks) == num_jobs
    ), f"Expected {num_jobs} completed tasks, got {len(completed_tasks)}"

    # Verify each task was completed by exactly one user
    all_completed = []
    for user_completed in completed_by_user.values():
        all_completed.extend(user_completed)

    assert len(all_completed) == num_jobs, "All jobs should be completed"
    assert len(set(all_completed)) == num_jobs, "No job should be completed more than once"

    print(f"Successfully completed {num_jobs} jobs across {num_users} users")
    for user_id, jobs in completed_by_user.items():
        print(f"  {user_id}: {len(jobs)} jobs")
