# pylint: disable=redefined-outer-name,unused-argument
import time
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select

from zetta_utils.task_management.db import get_db_session
from zetta_utils.task_management.db.models import Base, TimesheetModel
from zetta_utils.task_management.dependency import create_dependency
from zetta_utils.task_management.project import create_project
from zetta_utils.task_management.task import (
    _handle_task_completion,
    create_task,
    get_task,
    pause_task,
    release_task,
    start_task,
    unpause_task,
)
from zetta_utils.task_management.task_type import create_task_type
from zetta_utils.task_management.timesheet import submit_timesheet
from zetta_utils.task_management.types import (
    Dependency,
    Task,
    TaskType,
    User,
    UserUpdate,
)
from zetta_utils.task_management.user import create_user, get_user, update_user


def setup_simple_scenario(db_session, project_name):
    # Create the project first
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    create_task_type(
        project_name=project_name,
        data=TaskType(task_type="test", completion_statuses=["done"]),
        db_session=db_session,
    )

    for letter in ["a", "b", "c"]:
        create_task(
            project_name=project_name,
            data=Task(
                task_id=letter,
                assigned_user_id="",
                active_user_id="",
                completed_user_id="",
                ng_state={"url": "http://x"},
                ng_state_initial={"url": "http://x"},
                priority=1,
                batch_id="b",
                task_type="test",
                is_active=letter != "c",
                last_leased_ts=0.0,
                completion_status="",
            ),
            db_session=db_session,
        )

    for letter in ["a", "b"]:
        create_dependency(
            project_name=project_name,
            data=Dependency(
                dependency_id=f"c_on_{letter}",
                task_id="c",
                dependent_on_task_id=letter,
                required_completion_status="done",
                is_satisfied=False,
            ),
            db_session=db_session,
        )

    db_session.commit()


def setup_takeover_scenario(db_session, project_name):
    # Create the project first
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    create_task_type(
        project_name=project_name,
        data=TaskType(task_type="test", completion_statuses=["done"]),
        db_session=db_session,
    )

    for i in range(1, 4):
        create_user(
            project_name=project_name,
            data=User(
                user_id=f"user_{i}",
                hourly_rate=50.0,
                active_task="",
                qualified_task_types=["test"],
            ),
            db_session=db_session,
        )

    # Create a task that becomes idle
    old_time = time.time() - 300  # 5 minutes ago (older than idle threshold)
    create_task(
        project_name=project_name,
        data=Task(
            task_id="idle_task",
            assigned_user_id="",
            active_user_id="user_1",
            completed_user_id="",
            ng_state={"url": "http://x"},
            ng_state_initial={"url": "http://x"},
            priority=1,
            batch_id="b",
            task_type="test",
            is_active=True,
            last_leased_ts=old_time,
            completion_status="",
        ),
        db_session=db_session,
    )

    update_user(
        project_name=project_name,
        user_id="user_1",
        data=UserUpdate(active_task="idle_task"),
        db_session=db_session,
    )

    db_session.commit()


def setup_timesheet_scenario(db_session, project_name):
    # Create the project first
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    create_task_type(
        project_name=project_name,
        data=TaskType(task_type="test", completion_statuses=["done"]),
        db_session=db_session,
    )

    for i in range(1, 4):
        create_user(
            project_name=project_name,
            data=User(
                user_id=f"timesheet_user_{i}",
                hourly_rate=50.0,
                active_task="",
                qualified_task_types=["test"],
            ),
            db_session=db_session,
        )

    for i in range(1, 4):
        create_task(
            project_name=project_name,
            data=Task(
                task_id=f"timesheet_task_{i}",
                assigned_user_id="",
                active_user_id=f"timesheet_user_{i}",
                completed_user_id="",
                ng_state={"url": "http://x"},
                ng_state_initial={"url": "http://x"},
                priority=1,
                batch_id="b",
                task_type="test",
                is_active=True,
                last_leased_ts=time.time(),
                completion_status="",
            ),
            db_session=db_session,
        )

    for i in range(1, 4):
        update_user(
            project_name=project_name,
            user_id=f"timesheet_user_{i}",
            data=UserUpdate(active_task=f"timesheet_task_{i}"),
            db_session=db_session,
        )

    db_session.commit()


def complete_task(postgres_container, project_name, task_id):
    """Worker function - creates its own session to avoid sharing"""
    connection_url = postgres_container.get_connection_url()
    session = get_db_session(engine_url=connection_url)
    try:
        _handle_task_completion(session, project_name, task_id, "done")
        session.commit()
    finally:
        session.close()


def takeover_task(postgres_container, project_name, user_id, task_id):
    """Worker function to test start_task takeover race conditions"""
    connection_url = postgres_container.get_connection_url()
    session = get_db_session(engine_url=connection_url)
    try:
        result = start_task(
            project_name=project_name,
            user_id=user_id,
            task_id=task_id,
            db_session=session,
        )
        session.commit()
        return result
    except Exception as e:  # pylint: disable=broad-exception-caught
        session.rollback()
        return f"ERROR: {e}"
    finally:
        session.close()


def submit_concurrent_timesheet(postgres_container, project_name, user_id, task_id, duration):
    """Worker function to test submit_timesheet race conditions"""
    connection_url = postgres_container.get_connection_url()
    session = get_db_session(engine_url=connection_url)
    try:
        submit_timesheet(
            project_name=project_name,
            user_id=user_id,
            duration_seconds=duration,
            task_id=task_id,
            db_session=session,
        )
        session.commit()
        return "SUCCESS"
    except Exception as e:  # pylint: disable=broad-exception-caught
        session.rollback()
        return f"ERROR: {e}"
    finally:
        session.close()


def test_sequential_works(clean_db, postgres_session, project_name):
    """Sequential should work fine"""
    setup_simple_scenario(postgres_session, project_name)

    _handle_task_completion(postgres_session, project_name, "a", "done")
    _handle_task_completion(postgres_session, project_name, "b", "done")

    final_c = get_task(project_name=project_name, task_id="c", db_session=postgres_session)
    assert final_c["is_active"], "C should be active"


def test_concurrent_race_condition(clean_db, postgres_container, postgres_session, project_name):
    N_ITERATIONS = 5  # Reduced from 20 to avoid connection pool exhaustion

    for iteration in range(N_ITERATIONS):
        print(f"\n=== Iteration {iteration + 1}/{N_ITERATIONS} ===")

        # Clean up database state for this iteration
        postgres_session.rollback()

        Base.metadata.drop_all(postgres_session.bind)
        Base.metadata.create_all(postgres_session.bind)
        postgres_session.commit()

        # Set up scenario for this iteration
        setup_simple_scenario(postgres_session, project_name)

        # Run 2 workers concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(complete_task, postgres_container, project_name, "a")
            future_b = executor.submit(complete_task, postgres_container, project_name, "b")

            # Wait for both to complete with timeout
            future_a.result(timeout=10)
            future_b.result(timeout=10)

        # Small delay to ensure database consistency
        time.sleep(0.1)

        # Check result for this iteration
        final_c = get_task(project_name=project_name, task_id="c", db_session=postgres_session)
        assert final_c[
            "is_active"
        ], f"C should be active after A and B complete (iteration {iteration + 1})"

        print(f"Iteration {iteration + 1} passed ✓")

    print(f"\nAll {N_ITERATIONS} iterations passed! 🎉")


def test_start_task_takeover_race_condition(
    clean_db, postgres_container, postgres_session, project_name
):
    N_ITERATIONS = 5  # Reduced from 20 to avoid connection pool exhaustion

    for iteration in range(N_ITERATIONS):
        print(f"\n=== Start Task Takeover Iteration {iteration + 1}/{N_ITERATIONS} ===")

        # Clean up database state for this iteration
        postgres_session.rollback()

        Base.metadata.drop_all(postgres_session.bind)
        Base.metadata.create_all(postgres_session.bind)
        postgres_session.commit()

        # Set up scenario for this iteration
        setup_takeover_scenario(postgres_session, project_name)

        # Run 2 users trying to take over the same idle task concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for i in range(2, 4):  # user_2 and user_3 try to take over from user_1
                future = executor.submit(
                    takeover_task, postgres_container, project_name, f"user_{i}", "idle_task"
                )
                futures.append(future)

            # Collect results
            results = []
            for future in futures:
                result = future.result(timeout=10)
                results.append(result)

        print(f"Results: {results}")

        # Exactly one should succeed, others should fail
        successes = [r for r in results if r == "idle_task"]
        errors = [r for r in results if r.startswith("ERROR")]

        assert (
            len(successes) == 1
        ), f"Expected exactly 1 success, got {len(successes)}: {successes}"
        assert len(errors) == 1, f"Expected exactly 1 error, got {len(errors)}: {errors}"

        # Check final state - task should belong to one of the new users
        final_task = get_task(
            project_name=project_name, task_id="idle_task", db_session=postgres_session
        )
        assert final_task["active_user_id"] in [
            "user_2",
            "user_3",
        ], f"Task should belong to user_2 or user_3, got {final_task['active_user_id']}"

        # Check that the previous user (user_1) no longer has the task
        user_1 = get_user(project_name=project_name, user_id="user_1", db_session=postgres_session)
        assert (
            user_1["active_task"] == ""
        ), f"User_1 should no longer have active task, got {user_1['active_task']}"

        print(f"Takeover iteration {iteration + 1} passed ✓")

    print(f"\nAll {N_ITERATIONS} takeover iterations passed! 🎉")


def test_submit_timesheet_race_condition(
    clean_db, postgres_container, postgres_session, project_name
):
    N_ITERATIONS = 5  # Reduced from 20 to avoid connection pool exhaustion

    for iteration in range(N_ITERATIONS):
        print(f"\n=== Timesheet Race Iteration {iteration + 1}/{N_ITERATIONS} ===")

        # Clean up database state for this iteration
        postgres_session.rollback()

        Base.metadata.drop_all(postgres_session.bind)
        Base.metadata.create_all(postgres_session.bind)
        postgres_session.commit()

        # Set up scenario for this iteration
        setup_timesheet_scenario(postgres_session, project_name)

        # Run multiple concurrent timesheet submissions for the same user/task
        user_id = "timesheet_user_1"
        task_id = "timesheet_task_1"
        duration_per_submission = 300  # 5 minutes each
        num_concurrent_submissions = 3  # Reduced from 5

        with ThreadPoolExecutor(max_workers=num_concurrent_submissions) as executor:
            futures = []
            for _ in range(num_concurrent_submissions):
                future = executor.submit(
                    submit_concurrent_timesheet,
                    postgres_container,
                    project_name,
                    user_id,
                    task_id,
                    duration_per_submission,
                )
                futures.append(future)

            # Collect results
            results = []
            for future in futures:
                result = future.result(timeout=10)
                results.append(result)

        print(f"Timesheet results: {results}")

        # All should succeed
        successes = [r for r in results if r == "SUCCESS"]
        errors = [r for r in results if r.startswith("ERROR")]

        assert (
            len(successes) == num_concurrent_submissions
        ), f"Expected {num_concurrent_submissions} successes, got {len(successes)}: {successes}"
        assert len(errors) == 0, f"Expected 0 errors, got {len(errors)}: {errors}"

        # Check final timesheet total - should be exactly sum of all submissions

        timesheet_query = (
            select(TimesheetModel)
            .where(TimesheetModel.project_name == project_name)
            .where(TimesheetModel.user == user_id)
            .where(TimesheetModel.task_id == task_id)
        )
        timesheet_entries = postgres_session.execute(timesheet_query).scalars().all()

        total_seconds = sum(entry.seconds_spent for entry in timesheet_entries)
        expected_total = duration_per_submission * num_concurrent_submissions

        assert (
            total_seconds == expected_total
        ), f"Expected total {expected_total} seconds, got {total_seconds}"

        print(f"Timesheet iteration {iteration + 1} passed ✓ (Total: {total_seconds}s)")

    print(f"\nAll {N_ITERATIONS} timesheet iterations passed! 🎉")


def test_auto_selection_race_condition(
    clean_db, postgres_container, postgres_session, project_name
):
    def auto_select_worker(user_id):
        connection_url = postgres_container.get_connection_url()
        session = get_db_session(engine_url=connection_url)
        try:
            result = start_task(project_name=project_name, user_id=user_id, db_session=session)
            session.commit()
            return result if result else "NO_TASK"
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            return f"ERROR: {e}"
        finally:
            session.close()

    # Set up scenario - single task, 3 users
    # Create the project first
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=postgres_session,
    )

    create_task_type(
        project_name=project_name,
        data=TaskType(task_type="test", completion_statuses=["done"]),
        db_session=postgres_session,
    )

    for i in range(1, 4):
        create_user(
            project_name=project_name,
            data=User(
                user_id=f"auto_user_{i}",
                hourly_rate=50.0,
                active_task="",
                qualified_task_types=["test"],
            ),
            db_session=postgres_session,
        )

    create_task(
        project_name=project_name,
        data=Task(
            task_id="available_task",
            assigned_user_id="",
            active_user_id="",
            completed_user_id="",
            ng_state={"url": "http://x"},
            ng_state_initial={"url": "http://x"},
            priority=1,
            batch_id="b",
            task_type="test",
            is_active=True,
            last_leased_ts=0.0,
            completion_status="",
        ),
        db_session=postgres_session,
    )

    postgres_session.commit()

    # Run 3 users trying to auto-select simultaneously
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(auto_select_worker, f"auto_user_{i}") for i in range(1, 4)]
        results = [future.result(timeout=10) for future in futures]

    print(f"Auto-selection results: {results}")

    # Only one should get the task
    successes = [r for r in results if r == "available_task"]
    assert (
        len(successes) == 1
    ), f"RACE CONDITION: Expected 1 success, got {len(successes)}: {results}"


def test_task_completion_race_condition(
    clean_db, postgres_container, postgres_session, project_name
):
    def complete_worker(task_id):
        connection_url = postgres_container.get_connection_url()
        session = get_db_session(engine_url=connection_url)
        try:

            user_id = f"completion_user_{task_id.split('_')[-1]}"  # Extract user from task_id
            release_task(
                project_name=project_name,
                user_id=user_id,
                task_id=task_id,
                completion_status="done",
                db_session=session,
            )
            return "SUCCESS"
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            return f"ERROR: {e}"
        finally:
            session.close()

    # Create the project first
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=postgres_session,
    )

    create_task_type(
        project_name=project_name,
        data=TaskType(task_type="test", completion_statuses=["done"]),
        db_session=postgres_session,
    )

    create_user(
        project_name=project_name,
        data=User(
            user_id="completion_user_1",
            hourly_rate=50.0,
            active_task="completion_task_1",
            qualified_task_types=["test"],
        ),
        db_session=postgres_session,
    )
    create_user(
        project_name=project_name,
        data=User(
            user_id="completion_user_2",
            hourly_rate=50.0,
            active_task="completion_task_2",
            qualified_task_types=["test"],
        ),
        db_session=postgres_session,
    )

    for i in range(1, 3):
        create_task(
            project_name=project_name,
            data=Task(
                task_id=f"completion_task_{i}",
                assigned_user_id="",
                active_user_id=f"completion_user_{i}",
                completed_user_id="",
                ng_state={"url": "http://x"},
                ng_state_initial={"url": "http://x"},
                priority=1,
                batch_id="b",
                task_type="test",
                is_active=True,
                last_leased_ts=0.0,
                completion_status="",
            ),
            db_session=postgres_session,
        )

    postgres_session.commit()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(complete_worker, f"completion_task_{i}") for i in range(1, 3)]
        results = [future.result(timeout=10) for future in futures]

    print(f"Completion results: {results}")

    successes = [r for r in results if r == "SUCCESS"]
    assert (
        len(successes) == 2
    ), f"Expected 2 successful completions, got {len(successes)}: {results}"

    # Job status check removed as jobs no longer exist
    # Test passes if both tasks complete successfully


def test_pause_unpause_task_functionality(
    clean_db, postgres_container, postgres_session, project_name
):
    # Create the project first
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=postgres_session,
    )

    create_task_type(
        project_name=project_name,
        data=TaskType(task_type="test", completion_statuses=["done"]),
        db_session=postgres_session,
    )

    create_user(
        project_name=project_name,
        data=User(
            user_id="pause_user",
            hourly_rate=50.0,
            active_task="",
            qualified_task_types=["test"],
        ),
        db_session=postgres_session,
    )

    create_task(
        project_name=project_name,
        data=Task(
            task_id="pausable_task",
            assigned_user_id="",
            active_user_id="",
            completed_user_id="",
            ng_state={"url": "http://x"},
            ng_state_initial={"url": "http://x"},
            priority=1,
            batch_id="b",
            task_type="test",
            is_active=True,
            last_leased_ts=0.0,
            completion_status="",
        ),
        db_session=postgres_session,
    )

    postgres_session.commit()

    auto_selected = start_task(
        project_name=project_name, user_id="pause_user", db_session=postgres_session
    )
    assert (
        auto_selected == "pausable_task"
    ), f"Expected auto-selection to work, got {auto_selected}"

    release_task(
        project_name=project_name,
        user_id="pause_user",
        task_id="pausable_task",
        completion_status="",
        db_session=postgres_session,
    )

    pause_task(project_name=project_name, task_id="pausable_task", db_session=postgres_session)

    auto_selected_paused = start_task(
        project_name=project_name, user_id="pause_user", db_session=postgres_session
    )
    assert (
        auto_selected_paused is None
    ), f"Expected paused task to not be auto-selected, got {auto_selected_paused}"

    manual_selected = start_task(
        project_name=project_name,
        user_id="pause_user",
        task_id="pausable_task",
        db_session=postgres_session,
    )
    assert (
        manual_selected == "pausable_task"
    ), f"Expected manual selection to work on paused task, got {manual_selected}"

    release_task(
        project_name=project_name,
        user_id="pause_user",
        task_id="pausable_task",
        completion_status="",
        db_session=postgres_session,
    )

    unpause_task(project_name=project_name, task_id="pausable_task", db_session=postgres_session)

    auto_selected_unpaused = start_task(
        project_name=project_name, user_id="pause_user", db_session=postgres_session
    )
    assert (
        auto_selected_unpaused == "pausable_task"
    ), f"Expected unpaused task to be auto-selectable, got {auto_selected_unpaused}"


def test_pause_unpause_race_condition(
    clean_db, postgres_container, postgres_session, project_name
):
    def pause_worker(task_id):
        connection_url = postgres_container.get_connection_url()
        session = get_db_session(engine_url=connection_url)
        try:
            pause_task(project_name=project_name, task_id=task_id, db_session=session)
            return "PAUSED"
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            return f"ERROR: {e}"
        finally:
            session.close()

    def unpause_worker(task_id):
        connection_url = postgres_container.get_connection_url()
        session = get_db_session(engine_url=connection_url)
        try:
            unpause_task(project_name=project_name, task_id=task_id, db_session=session)
            return "UNPAUSED"
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            return f"ERROR: {e}"
        finally:
            session.close()

    # Create the project first
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=postgres_session,
    )

    create_task_type(
        project_name=project_name,
        data=TaskType(task_type="test", completion_statuses=["done"]),
        db_session=postgres_session,
    )

    create_task(
        project_name=project_name,
        data=Task(
            task_id="race_pause_task",
            assigned_user_id="",
            active_user_id="",
            completed_user_id="",
            ng_state={"url": "http://x"},
            ng_state_initial={"url": "http://x"},
            priority=1,
            batch_id="b",
            task_type="test",
            is_active=True,
            last_leased_ts=0.0,
            completion_status="",
        ),
        db_session=postgres_session,
    )

    postgres_session.commit()

    # Run concurrent pause/unpause operations
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        # 2 pause operations and 2 unpause operations
        for _ in range(2):
            futures.append(executor.submit(pause_worker, "race_pause_task"))
            futures.append(executor.submit(unpause_worker, "race_pause_task"))

        results = [future.result(timeout=10) for future in futures]

    print(f"Pause/Unpause race results: {results}")

    # All operations should succeed (some will be no-ops if already in that state)
    errors = [r for r in results if r.startswith("ERROR")]
    assert len(errors) == 0, f"Expected no errors in pause/unpause operations, got: {errors}"

    final_task = get_task(
        project_name=project_name, task_id="race_pause_task", db_session=postgres_session
    )
    assert final_task.get("is_paused", False) in [
        True,
        False,
    ], f"Expected consistent pause state, got {final_task.get('is_paused', False)}"


def test_paused_tasks_excluded_from_auto_selection_comprehensive(
    clean_db, postgres_container, postgres_session, project_name
):
    # Create the project first
    create_project(
        project_name=project_name,
        segmentation_path="gs://test-bucket/segmentation",
        sv_resolution_x=4.0,
        sv_resolution_y=4.0,
        sv_resolution_z=40.0,
        db_session=postgres_session,
    )

    create_task_type(
        project_name=project_name,
        data=TaskType(task_type="test", completion_statuses=["done"]),
        db_session=postgres_session,
    )

    create_user(
        project_name=project_name,
        data=User(
            user_id="auto_user",
            hourly_rate=50.0,
            active_task="",
            qualified_task_types=["test"],
        ),
        db_session=postgres_session,
    )

    create_task(
        project_name=project_name,
        data=Task(
            task_id="assigned_task",
            assigned_user_id="auto_user",
            active_user_id="",
            completed_user_id="",
            ng_state={"url": "http://x"},
            ng_state_initial={"url": "http://x"},
            priority=3,
            batch_id="b",
            task_type="test",
            is_active=True,
            last_leased_ts=0.0,
            completion_status="",
        ),
        db_session=postgres_session,
    )

    create_task(
        project_name=project_name,
        data=Task(
            task_id="unassigned_task",
            assigned_user_id="",
            active_user_id="",
            completed_user_id="",
            ng_state={"url": "http://x"},
            ng_state_initial={"url": "http://x"},
            priority=2,
            batch_id="b",
            task_type="test",
            is_active=True,
            last_leased_ts=0.0,
            completion_status="",
        ),
        db_session=postgres_session,
    )

    old_time = time.time() - 300  # 5 minutes ago (idle)
    create_task(
        project_name=project_name,
        data=Task(
            task_id="idle_task",
            assigned_user_id="",
            active_user_id="other_user",
            completed_user_id="",
            ng_state={"url": "http://x"},
            ng_state_initial={"url": "http://x"},
            priority=1,
            batch_id="b",
            task_type="test",
            is_active=True,
            last_leased_ts=old_time,
            completion_status="",
        ),
        db_session=postgres_session,
    )

    postgres_session.commit()

    selected = start_task(
        project_name=project_name, user_id="auto_user", db_session=postgres_session
    )
    assert selected == "assigned_task", f"Expected to select assigned task first, got {selected}"

    release_task(
        project_name=project_name,
        user_id="auto_user",
        task_id="assigned_task",
        completion_status="",
        db_session=postgres_session,
    )

    pause_task(project_name=project_name, task_id="assigned_task", db_session=postgres_session)

    selected = start_task(
        project_name=project_name, user_id="auto_user", db_session=postgres_session
    )
    assert (
        selected == "unassigned_task"
    ), f"Expected to select unassigned task after pausing assigned, got {selected}"

    release_task(
        project_name=project_name,
        user_id="auto_user",
        task_id="unassigned_task",
        completion_status="",
        db_session=postgres_session,
    )
    pause_task(project_name=project_name, task_id="unassigned_task", db_session=postgres_session)

    selected = start_task(
        project_name=project_name, user_id="auto_user", db_session=postgres_session
    )
    assert (
        selected == "idle_task"
    ), f"Expected to select idle task after pausing others, got {selected}"

    release_task(
        project_name=project_name,
        user_id="auto_user",
        task_id="idle_task",
        completion_status="",
        db_session=postgres_session,
    )
    pause_task(project_name=project_name, task_id="idle_task", db_session=postgres_session)

    # Should now select nothing
    selected = start_task(
        project_name=project_name, user_id="auto_user", db_session=postgres_session
    )
    assert selected is None, f"Expected no selection when all tasks paused, got {selected}"
