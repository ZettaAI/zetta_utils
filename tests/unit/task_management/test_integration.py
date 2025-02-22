# pylint: disable=redefined-outer-name,unused-argument
import multiprocessing as mp
import time

import pytest

from zetta_utils.task_management import project, subtask
from zetta_utils.task_management.ingestion import ingest_batch
from zetta_utils.task_management.project import (
    create_project_tables,
    get_firestore_client,
)
from zetta_utils.task_management.subtask_type import create_subtask_type
from zetta_utils.task_management.task import create_tasks_batch
from zetta_utils.task_management.timesheet import submit_timesheet
from zetta_utils.task_management.types import SubtaskType, Task, TimesheetEntry, User
from zetta_utils.task_management.user import create_user


def get_test_results():
    return {"completed_tasks": {}, "failed_tasks": {}, "stolen_tasks": {}}


@pytest.fixture
def short_idle_timeout():
    original_timeout = subtask._MAX_IDLE_SECONDS  # pylint: disable=protected-access
    subtask._MAX_IDLE_SECONDS = 2  # pylint: disable=protected-access
    yield
    subtask._MAX_IDLE_SECONDS = original_timeout  # pylint: disable=protected-access


@pytest.fixture
def test_environment():
    """Set up test environment with required subtask types and structures"""
    project_name = "test_project"
    project.DEFAULT_CLIENT_CONFIG["project"] = "zetta-research"
    project.DEFAULT_CLIENT_CONFIG["database"] = "zetta-utils"
    client = get_firestore_client()
    print(f"Client config: {project.DEFAULT_CLIENT_CONFIG}")
    # Clear ALL collections at the start
    collections = client.collections()
    for collection in collections:
        if collection.id in [
            "projects",
            "subtask_types",
            f"{project_name}_tasks",
            f"{project_name}_subtasks",
            f"{project_name}_timesheets",
            f"{project_name}_dependencies",
            f"{project_name}_users",
        ]:
            docs = collection.stream()
            for doc in docs:
                doc.reference.delete()
    print("Cleared collections")

    # Create project tables
    create_project_tables(project_name)
    print("Created project tables")

    # Create required subtask types
    proofread_type = SubtaskType(
        subtask_type="segmentation_proofread",
        completion_statuses=["done", "need_help"],
        description="Proofread segmentation",
    )
    verify_type = SubtaskType(
        subtask_type="segmentation_verify",
        completion_statuses=["done", "reject"],
        description="Verify segmentation",
    )
    expert_type = SubtaskType(
        subtask_type="segmentation_proofread_expert",
        completion_statuses=["done"],
        description="Expert proofread",
    )

    for subtask_type in [proofread_type, verify_type, expert_type]:
        create_subtask_type(subtask_type)

    yield project_name, client  # Return both project name and client

    # Clean up after test
    for collection in collections:
        docs = collection.stream()
        for doc in docs:
            doc.reference.delete()

    del project.DEFAULT_CLIENT_CONFIG["project"]
    del project.DEFAULT_CLIENT_CONFIG["database"]


def create_test_tasks(project_name: str, num_tasks: int) -> list[str]:
    """Create test tasks and return their IDs"""
    tasks = []
    for i in range(num_tasks):
        task_data = Task(
            task_id=f"test_task_{i}",
            batch_id="test_batch",
            status="pending_ingestion",
            task_type="test",
            link=f"http://test.com/task_{i}",
        )
        tasks.append(task_data)

    return create_tasks_batch(project_name, tasks, batch_size=100)


def good_worker(project_name: str, user_id: str, num_tasks: int, results_dict, lock):
    """Worker that properly completes tasks with timesheets"""
    try:
        print(f"Good worker {user_id} starting")
        # Create user with proper qualifications
        user_data = User(
            user_id=user_id,
            hourly_rate=20.0,
            active_subtask="",
            qualified_subtask_types=["segmentation_proofread"],
        )
        create_user(project_name, user_data)
        print(f"Good worker {user_id} created user")

        my_completed = []
        while True:
            # Check if all tasks are completed
            total_completed = sum(len(tasks) for tasks in results_dict["completed_tasks"].values())
            if total_completed >= num_tasks:
                break

            # Try to get a task
            subtask_id = subtask.start_subtask(project_name, user_id)
            if subtask_id:
                print(f"Good worker {user_id} got subtask {subtask_id}")
                for _ in range(1):
                    time.sleep(0.1)
                    timesheet_entry = TimesheetEntry(
                        duration_seconds=1.0,
                        description="Test work",
                    )
                    submit_timesheet(project_name, user_id, timesheet_entry)
                    print(f"Good worker {user_id} submitted timesheet for {subtask_id}")

                # Complete the task
                subtask.release_subtask(project_name, user_id, "done")
                print(f"Good worker {user_id} completed subtask {subtask_id}")

                # Update the shared dictionary
                my_completed.append(subtask_id)
                with lock:
                    completed_tasks = dict(results_dict["completed_tasks"])
                    completed_tasks[user_id] = my_completed.copy()
                    results_dict["completed_tasks"] = completed_tasks
            else:
                time.sleep(0.01)

    except Exception as e:
        print(f"Error in good worker {user_id}: {e}")
        raise e


def bad_worker(project_name: str, user_id: str, results_dict):
    """Worker that takes tasks but doesn't submit timesheets"""
    try:
        # Create user with proper qualifications
        user_data = User(
            user_id=user_id,
            hourly_rate=20.0,
            active_subtask="",
            qualified_subtask_types=["segmentation_proofread"],
        )
        create_user(project_name, user_data)

        while True:
            subtask_id = subtask.start_subtask(project_name, user_id)
            if subtask_id:
                print(f"Bad worker {user_id}    got subtask {subtask_id}")
                time.sleep(4)
            else:
                time.sleep(0.01)
    except Exception as e:
        print(f"Error in bad worker {user_id}: {e}")
        raise e


def concurrent_task_processing(test_environment, short_idle_timeout):
    """Test concurrent task processing with good and bad workers"""
    project_name, client = test_environment
    num_tasks = 200
    num_good_workers = 20
    num_bad_workers = 1

    with mp.Manager() as manager:
        results_dict = manager.dict(get_test_results())
        lock = manager.Lock()

        task_ids = create_test_tasks(project_name, num_tasks)
        success = ingest_batch(
            project_name=project_name,
            batch_id="test_batch",
            subtask_structure="segmentation_proofread_simple",
            priority=1,
        )
        assert success, "Failed to ingest tasks"
        print(f"Ingested {num_tasks} tasks")

        processes = []
        for i in range(num_good_workers):
            p = mp.Process(
                target=good_worker,
                args=(project_name, f"good_worker_{i}", num_tasks, results_dict, lock),
            )
            processes.append(p)
            print(f"About to start good worker {i}")
            p.start()
        print(f"Started {num_good_workers} good workers")

        # Start bad workers
        for i in range(num_bad_workers):
            p = mp.Process(target=bad_worker, args=(project_name, f"bad_worker_{i}", results_dict))
            processes.append(p)
            p.start()
        print(f"Started {num_bad_workers} bad workers")

        # Wait for good workers to finish
        for p in processes[:num_good_workers]:
            p.join(timeout=120)
        print("Good workers finished")

        # Terminate bad workers
        for p in processes[num_good_workers:]:
            p.terminate()
            p.join()
        print("Bad workers finished")

        # Get all completed subtask IDs
        all_completed_subtasks = set()
        for completed_tasks in results_dict["completed_tasks"].values():
            all_completed_subtasks.update(completed_tasks)

        # Verify each task was completed exactly once
        for task_id in task_ids:
            # Find all subtasks for this task
            task_subtasks = (
                client.collection(f"{project_name}_subtasks")
                .where("task_id", "==", task_id)
                .where("subtask_type", "==", "segmentation_proofread")
                .stream()
            )

            # Each task should have exactly one proofread subtask
            task_subtask_ids = [s.id for s in task_subtasks]
            assert len(task_subtask_ids) == 1, f"Task {task_id} has wrong number of subtasks"

            # That subtask should be in our completed set
            assert (
                task_subtask_ids[0] in all_completed_subtasks
            ), f"Task {task_id}'s subtask {task_subtask_ids[0]} was not completed"

        # Verify no subtask was completed more than once
        assert (
            len(all_completed_subtasks) == num_tasks
        ), "Some subtasks were completed multiple times or not all tasks were completed"

        print("All tasks were completed exactly once")
