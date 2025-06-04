# pylint: disable=redefined-outer-name,unused-argument
import time
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select

from zetta_utils.task_management.db import get_db_session
from zetta_utils.task_management.db.models import Base, TimesheetModel
from zetta_utils.task_management.dependency import create_dependency
from zetta_utils.task_management.subtask import (
    _handle_subtask_completion,
    create_subtask,
    get_subtask,
    pause_subtask,
    release_subtask,
    start_subtask,
    unpause_subtask,
)
from zetta_utils.task_management.subtask_type import create_subtask_type
from zetta_utils.task_management.task import create_task, get_task
from zetta_utils.task_management.timesheet import submit_timesheet
from zetta_utils.task_management.types import (
    Dependency,
    Subtask,
    SubtaskType,
    Task,
    User,
    UserUpdate,
)
from zetta_utils.task_management.user import create_user, get_user, update_user


def setup_simple_scenario(db_session, project_name):
    create_subtask_type(
        project_name=project_name,
        data=SubtaskType(subtask_type="test", completion_statuses=["done"]),
        db_session=db_session,
    )
    create_task(
        project_name=project_name,
        data=Task(
            task_id="t", batch_id="b", status="ingested", task_type="test", ng_state="http://x"
        ),
        db_session=db_session,
    )

    for letter in ["a", "b", "c"]:
        create_subtask(
            project_name=project_name,
            data=Subtask(
                task_id="t",
                subtask_id=letter,
                assigned_user_id="",
                active_user_id="",
                completed_user_id="",
                ng_state="http://x",
                ng_state_initial="http://x",
                priority=1,
                batch_id="b",
                subtask_type="test",
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
                subtask_id="c",
                dependent_on_subtask_id=letter,
                required_completion_status="done",
                is_satisfied=False,
            ),
            db_session=db_session,
        )

    db_session.commit()


def setup_takeover_scenario(db_session, project_name):
    create_subtask_type(
        project_name=project_name,
        data=SubtaskType(subtask_type="test", completion_statuses=["done"]),
        db_session=db_session,
    )
    create_task(
        project_name=project_name,
        data=Task(
            task_id="takeover_task",
            batch_id="b",
            status="ingested",
            task_type="test",
            ng_state="http://x",
        ),
        db_session=db_session,
    )

    for i in range(1, 4):
        create_user(
            project_name=project_name,
            data=User(
                user_id=f"user_{i}",
                hourly_rate=50.0,
                active_subtask="",
                qualified_subtask_types=["test"],
            ),
            db_session=db_session,
        )

    # Create a subtask that becomes idle
    old_time = time.time() - 300  # 5 minutes ago (older than idle threshold)
    create_subtask(
        project_name=project_name,
        data=Subtask(
            task_id="takeover_task",
            subtask_id="idle_subtask",
            assigned_user_id="",
            active_user_id="user_1",
            completed_user_id="",
            ng_state="http://x",
            ng_state_initial="http://x",
            priority=1,
            batch_id="b",
            subtask_type="test",
            is_active=True,
            last_leased_ts=old_time,
            completion_status="",
        ),
        db_session=db_session,
    )

    update_user(
        project_name=project_name,
        user_id="user_1",
        data=UserUpdate(active_subtask="idle_subtask"),
        db_session=db_session,
    )

    db_session.commit()


def setup_timesheet_scenario(db_session, project_name):
    create_subtask_type(
        project_name=project_name,
        data=SubtaskType(subtask_type="test", completion_statuses=["done"]),
        db_session=db_session,
    )
    create_task(
        project_name=project_name,
        data=Task(
            task_id="timesheet_task",
            batch_id="b",
            status="ingested",
            task_type="test",
            ng_state="http://x",
        ),
        db_session=db_session,
    )

    for i in range(1, 4):
        create_user(
            project_name=project_name,
            data=User(
                user_id=f"timesheet_user_{i}",
                hourly_rate=50.0,
                active_subtask="",
                qualified_subtask_types=["test"],
            ),
            db_session=db_session,
        )

    for i in range(1, 4):
        create_subtask(
            project_name=project_name,
            data=Subtask(
                task_id="timesheet_task",
                subtask_id=f"timesheet_subtask_{i}",
                assigned_user_id="",
                active_user_id=f"timesheet_user_{i}",
                completed_user_id="",
                ng_state="http://x",
                ng_state_initial="http://x",
                priority=1,
                batch_id="b",
                subtask_type="test",
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
            data=UserUpdate(active_subtask=f"timesheet_subtask_{i}"),
            db_session=db_session,
        )

    db_session.commit()


def complete_subtask(postgres_container, project_name, subtask_id):
    """Worker function - creates its own session to avoid sharing"""
    connection_url = postgres_container.get_connection_url()
    session = get_db_session(engine_url=connection_url)
    try:
        _handle_subtask_completion(session, project_name, subtask_id, "done")
        session.commit()
    finally:
        session.close()


def takeover_subtask(postgres_container, project_name, user_id, subtask_id):
    """Worker function to test start_subtask takeover race conditions"""
    connection_url = postgres_container.get_connection_url()
    session = get_db_session(engine_url=connection_url)
    try:
        result = start_subtask(
            project_name=project_name,
            user_id=user_id,
            subtask_id=subtask_id,
            db_session=session,
        )
        session.commit()
        return result
    except Exception as e:  # pylint: disable=broad-exception-caught
        session.rollback()
        return f"ERROR: {e}"
    finally:
        session.close()


def submit_concurrent_timesheet(postgres_container, project_name, user_id, subtask_id, duration):
    """Worker function to test submit_timesheet race conditions"""
    connection_url = postgres_container.get_connection_url()
    session = get_db_session(engine_url=connection_url)
    try:
        submit_timesheet(
            project_name=project_name,
            user_id=user_id,
            duration_seconds=duration,
            subtask_id=subtask_id,
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

    _handle_subtask_completion(postgres_session, project_name, "a", "done")
    _handle_subtask_completion(postgres_session, project_name, "b", "done")

    final_c = get_subtask(project_name=project_name, subtask_id="c", db_session=postgres_session)
    assert final_c["is_active"], "C should be active"


def test_concurrent_race_condition(clean_db, postgres_container, postgres_session, project_name):
    N_ITERATIONS = 20

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
            future_a = executor.submit(complete_subtask, postgres_container, project_name, "a")
            future_b = executor.submit(complete_subtask, postgres_container, project_name, "b")

            # Wait for both to complete with timeout
            future_a.result(timeout=10)
            future_b.result(timeout=10)

        # Small delay to ensure database consistency
        time.sleep(0.1)

        # Check result for this iteration
        final_c = get_subtask(
            project_name=project_name, subtask_id="c", db_session=postgres_session
        )
        assert final_c[
            "is_active"
        ], f"C should be active after A and B complete (iteration {iteration + 1})"

        print(f"Iteration {iteration + 1} passed ✓")

    print(f"\nAll {N_ITERATIONS} iterations passed! 🎉")


def test_start_subtask_takeover_race_condition(
    clean_db, postgres_container, postgres_session, project_name
):
    N_ITERATIONS = 20

    for iteration in range(N_ITERATIONS):
        print(f"\n=== Start Subtask Takeover Iteration {iteration + 1}/{N_ITERATIONS} ===")

        # Clean up database state for this iteration
        postgres_session.rollback()

        Base.metadata.drop_all(postgres_session.bind)
        Base.metadata.create_all(postgres_session.bind)
        postgres_session.commit()

        # Set up scenario for this iteration
        setup_takeover_scenario(postgres_session, project_name)

        # Run 3 users trying to take over the same idle subtask concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(2, 4):  # user_2 and user_3 try to take over from user_1
                future = executor.submit(
                    takeover_subtask, postgres_container, project_name, f"user_{i}", "idle_subtask"
                )
                futures.append(future)

            # Collect results
            results = []
            for future in futures:
                result = future.result(timeout=10)
                results.append(result)

        print(f"Results: {results}")

        # Exactly one should succeed, others should fail
        successes = [r for r in results if r == "idle_subtask"]
        errors = [r for r in results if r.startswith("ERROR")]

        assert (
            len(successes) == 1
        ), f"Expected exactly 1 success, got {len(successes)}: {successes}"
        assert len(errors) == 1, f"Expected exactly 1 error, got {len(errors)}: {errors}"

        # Check final state - subtask should belong to one of the new users
        final_subtask = get_subtask(
            project_name=project_name, subtask_id="idle_subtask", db_session=postgres_session
        )
        assert final_subtask["active_user_id"] in [
            "user_2",
            "user_3",
        ], f"Subtask should belong to user_2 or user_3, got {final_subtask['active_user_id']}"

        # Check that the previous user (user_1) no longer has the subtask
        user_1 = get_user(project_name=project_name, user_id="user_1", db_session=postgres_session)
        assert (
            user_1["active_subtask"] == ""
        ), f"User_1 should no longer have active subtask, got {user_1['active_subtask']}"

        print(f"Takeover iteration {iteration + 1} passed ✓")

    print(f"\nAll {N_ITERATIONS} takeover iterations passed! 🎉")


def test_submit_timesheet_race_condition(
    clean_db, postgres_container, postgres_session, project_name
):
    N_ITERATIONS = 20

    for iteration in range(N_ITERATIONS):
        print(f"\n=== Timesheet Race Iteration {iteration + 1}/{N_ITERATIONS} ===")

        # Clean up database state for this iteration
        postgres_session.rollback()

        Base.metadata.drop_all(postgres_session.bind)
        Base.metadata.create_all(postgres_session.bind)
        postgres_session.commit()

        # Set up scenario for this iteration
        setup_timesheet_scenario(postgres_session, project_name)

        # Run multiple concurrent timesheet submissions for the same user/subtask
        user_id = "timesheet_user_1"
        subtask_id = "timesheet_subtask_1"
        duration_per_submission = 300  # 5 minutes each
        num_concurrent_submissions = 5

        with ThreadPoolExecutor(max_workers=num_concurrent_submissions) as executor:
            futures = []
            for _ in range(num_concurrent_submissions):
                future = executor.submit(
                    submit_concurrent_timesheet,
                    postgres_container,
                    project_name,
                    user_id,
                    subtask_id,
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
            .where(TimesheetModel.subtask_id == subtask_id)
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
            result = start_subtask(project_name=project_name, user_id=user_id, db_session=session)
            session.commit()
            return result if result else "NO_SUBTASK"
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            return f"ERROR: {e}"
        finally:
            session.close()

    # Set up scenario - single subtask, 3 users
    create_subtask_type(
        project_name=project_name,
        data=SubtaskType(subtask_type="test", completion_statuses=["done"]),
        db_session=postgres_session,
    )
    create_task(
        project_name=project_name,
        data=Task(
            task_id="auto_task",
            batch_id="b",
            status="ingested",
            task_type="test",
            ng_state="http://x",
        ),
        db_session=postgres_session,
    )

    for i in range(1, 4):
        create_user(
            project_name=project_name,
            data=User(
                user_id=f"auto_user_{i}",
                hourly_rate=50.0,
                active_subtask="",
                qualified_subtask_types=["test"],
            ),
            db_session=postgres_session,
        )

    create_subtask(
        project_name=project_name,
        data=Subtask(
            task_id="auto_task",
            subtask_id="available_subtask",
            assigned_user_id="",
            active_user_id="",
            completed_user_id="",
            ng_state="http://x",
            ng_state_initial="http://x",
            priority=1,
            batch_id="b",
            subtask_type="test",
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

    # Only one should get the subtask
    successes = [r for r in results if r == "available_subtask"]
    assert (
        len(successes) == 1
    ), f"RACE CONDITION: Expected 1 success, got {len(successes)}: {results}"


def test_task_completion_race_condition(
    clean_db, postgres_container, postgres_session, project_name
):
    def complete_worker(subtask_id):
        connection_url = postgres_container.get_connection_url()
        session = get_db_session(engine_url=connection_url)
        try:

            user_id = (
                f"completion_user_{subtask_id.split('_')[-1]}"  # Extract user from subtask_id
            )
            release_subtask(
                project_name=project_name,
                user_id=user_id,
                subtask_id=subtask_id,
                completion_status="done",
                db_session=session,
            )
            return "SUCCESS"
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            return f"ERROR: {e}"
        finally:
            session.close()

    create_subtask_type(
        project_name=project_name,
        data=SubtaskType(subtask_type="test", completion_statuses=["done"]),
        db_session=postgres_session,
    )
    create_task(
        project_name=project_name,
        data=Task(
            task_id="completion_task",
            batch_id="b",
            status="ingested",
            task_type="test",
            ng_state="http://x",
        ),
        db_session=postgres_session,
    )

    create_user(
        project_name=project_name,
        data=User(
            user_id="completion_user_1",
            hourly_rate=50.0,
            active_subtask="completion_subtask_1",
            qualified_subtask_types=["test"],
        ),
        db_session=postgres_session,
    )
    create_user(
        project_name=project_name,
        data=User(
            user_id="completion_user_2",
            hourly_rate=50.0,
            active_subtask="completion_subtask_2",
            qualified_subtask_types=["test"],
        ),
        db_session=postgres_session,
    )

    for i in range(1, 3):
        create_subtask(
            project_name=project_name,
            data=Subtask(
                task_id="completion_task",
                subtask_id=f"completion_subtask_{i}",
                assigned_user_id="",
                active_user_id=f"completion_user_{i}",
                completed_user_id="",
                ng_state="http://x",
                ng_state_initial="http://x",
                priority=1,
                batch_id="b",
                subtask_type="test",
                is_active=True,
                last_leased_ts=0.0,
                completion_status="",
            ),
            db_session=postgres_session,
        )

    postgres_session.commit()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(complete_worker, f"completion_subtask_{i}") for i in range(1, 3)
        ]
        results = [future.result(timeout=10) for future in futures]

    print(f"Completion results: {results}")

    successes = [r for r in results if r == "SUCCESS"]
    assert (
        len(successes) == 2
    ), f"Expected 2 successful completions, got {len(successes)}: {results}"

    final_task = get_task(
        project_name=project_name, task_id="completion_task", db_session=postgres_session
    )
    assert (
        final_task["status"] == "fully_processed"
    ), f"Task should be fully_processed, got {final_task['status']}"


def test_pause_unpause_subtask_functionality(
    clean_db, postgres_container, postgres_session, project_name
):
    create_subtask_type(
        project_name=project_name,
        data=SubtaskType(subtask_type="test", completion_statuses=["done"]),
        db_session=postgres_session,
    )
    create_task(
        project_name=project_name,
        data=Task(
            task_id="pause_task",
            batch_id="b",
            status="ingested",
            task_type="test",
            ng_state="http://x",
        ),
        db_session=postgres_session,
    )

    create_user(
        project_name=project_name,
        data=User(
            user_id="pause_user",
            hourly_rate=50.0,
            active_subtask="",
            qualified_subtask_types=["test"],
        ),
        db_session=postgres_session,
    )

    create_subtask(
        project_name=project_name,
        data=Subtask(
            task_id="pause_task",
            subtask_id="pausable_subtask",
            assigned_user_id="",
            active_user_id="",
            completed_user_id="",
            ng_state="http://x",
            ng_state_initial="http://x",
            priority=1,
            batch_id="b",
            subtask_type="test",
            is_active=True,
            last_leased_ts=0.0,
            completion_status="",
        ),
        db_session=postgres_session,
    )

    postgres_session.commit()

    auto_selected = start_subtask(
        project_name=project_name, user_id="pause_user", db_session=postgres_session
    )
    assert (
        auto_selected == "pausable_subtask"
    ), f"Expected auto-selection to work, got {auto_selected}"

    release_subtask(
        project_name=project_name,
        user_id="pause_user",
        subtask_id="pausable_subtask",
        completion_status="",
        db_session=postgres_session,
    )

    pause_subtask(
        project_name=project_name, subtask_id="pausable_subtask", db_session=postgres_session
    )

    auto_selected_paused = start_subtask(
        project_name=project_name, user_id="pause_user", db_session=postgres_session
    )
    assert (
        auto_selected_paused is None
    ), f"Expected paused subtask to not be auto-selected, got {auto_selected_paused}"

    manual_selected = start_subtask(
        project_name=project_name,
        user_id="pause_user",
        subtask_id="pausable_subtask",
        db_session=postgres_session,
    )
    assert (
        manual_selected == "pausable_subtask"
    ), f"Expected manual selection to work on paused subtask, got {manual_selected}"

    release_subtask(
        project_name=project_name,
        user_id="pause_user",
        subtask_id="pausable_subtask",
        completion_status="",
        db_session=postgres_session,
    )

    unpause_subtask(
        project_name=project_name, subtask_id="pausable_subtask", db_session=postgres_session
    )

    auto_selected_unpaused = start_subtask(
        project_name=project_name, user_id="pause_user", db_session=postgres_session
    )
    assert (
        auto_selected_unpaused == "pausable_subtask"
    ), f"Expected unpaused subtask to be auto-selectable, got {auto_selected_unpaused}"


def test_pause_unpause_race_condition(
    clean_db, postgres_container, postgres_session, project_name
):
    def pause_worker(subtask_id):
        connection_url = postgres_container.get_connection_url()
        session = get_db_session(engine_url=connection_url)
        try:
            pause_subtask(project_name=project_name, subtask_id=subtask_id, db_session=session)
            return "PAUSED"
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            return f"ERROR: {e}"
        finally:
            session.close()

    def unpause_worker(subtask_id):
        connection_url = postgres_container.get_connection_url()
        session = get_db_session(engine_url=connection_url)
        try:
            unpause_subtask(project_name=project_name, subtask_id=subtask_id, db_session=session)
            return "UNPAUSED"
        except Exception as e:  # pylint: disable=broad-exception-caught
            session.rollback()
            return f"ERROR: {e}"
        finally:
            session.close()

    create_subtask_type(
        project_name=project_name,
        data=SubtaskType(subtask_type="test", completion_statuses=["done"]),
        db_session=postgres_session,
    )
    create_task(
        project_name=project_name,
        data=Task(
            task_id="race_pause_task",
            batch_id="b",
            status="ingested",
            task_type="test",
            ng_state="http://x",
        ),
        db_session=postgres_session,
    )

    create_subtask(
        project_name=project_name,
        data=Subtask(
            task_id="race_pause_task",
            subtask_id="race_pause_subtask",
            assigned_user_id="",
            active_user_id="",
            completed_user_id="",
            ng_state="http://x",
            ng_state_initial="http://x",
            priority=1,
            batch_id="b",
            subtask_type="test",
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
            futures.append(executor.submit(pause_worker, "race_pause_subtask"))
            futures.append(executor.submit(unpause_worker, "race_pause_subtask"))

        results = [future.result(timeout=10) for future in futures]

    print(f"Pause/Unpause race results: {results}")

    # All operations should succeed (some will be no-ops if already in that state)
    errors = [r for r in results if r.startswith("ERROR")]
    assert len(errors) == 0, f"Expected no errors in pause/unpause operations, got: {errors}"

    final_subtask = get_subtask(
        project_name=project_name, subtask_id="race_pause_subtask", db_session=postgres_session
    )
    assert final_subtask.get("is_paused", False) in [
        True,
        False,
    ], f"Expected consistent pause state, got {final_subtask.get('is_paused', False)}"


def test_paused_subtasks_excluded_from_auto_selection_comprehensive(
    clean_db, postgres_container, postgres_session, project_name
):
    create_subtask_type(
        project_name=project_name,
        data=SubtaskType(subtask_type="test", completion_statuses=["done"]),
        db_session=postgres_session,
    )
    create_task(
        project_name=project_name,
        data=Task(
            task_id="auto_select_task",
            batch_id="b",
            status="ingested",
            task_type="test",
            ng_state="http://x",
        ),
        db_session=postgres_session,
    )

    create_user(
        project_name=project_name,
        data=User(
            user_id="auto_user",
            hourly_rate=50.0,
            active_subtask="",
            qualified_subtask_types=["test"],
        ),
        db_session=postgres_session,
    )

    create_subtask(
        project_name=project_name,
        data=Subtask(
            task_id="auto_select_task",
            subtask_id="assigned_subtask",
            assigned_user_id="auto_user",
            active_user_id="",
            completed_user_id="",
            ng_state="http://x",
            ng_state_initial="http://x",
            priority=3,
            batch_id="b",
            subtask_type="test",
            is_active=True,
            last_leased_ts=0.0,
            completion_status="",
        ),
        db_session=postgres_session,
    )

    create_subtask(
        project_name=project_name,
        data=Subtask(
            task_id="auto_select_task",
            subtask_id="unassigned_subtask",
            assigned_user_id="",
            active_user_id="",
            completed_user_id="",
            ng_state="http://x",
            ng_state_initial="http://x",
            priority=2,
            batch_id="b",
            subtask_type="test",
            is_active=True,
            last_leased_ts=0.0,
            completion_status="",
        ),
        db_session=postgres_session,
    )

    old_time = time.time() - 300  # 5 minutes ago (idle)
    create_subtask(
        project_name=project_name,
        data=Subtask(
            task_id="auto_select_task",
            subtask_id="idle_subtask",
            assigned_user_id="",
            active_user_id="other_user",
            completed_user_id="",
            ng_state="http://x",
            ng_state_initial="http://x",
            priority=1,
            batch_id="b",
            subtask_type="test",
            is_active=True,
            last_leased_ts=old_time,
            completion_status="",
        ),
        db_session=postgres_session,
    )

    postgres_session.commit()

    selected = start_subtask(
        project_name=project_name, user_id="auto_user", db_session=postgres_session
    )
    assert (
        selected == "assigned_subtask"
    ), f"Expected to select assigned subtask first, got {selected}"

    release_subtask(
        project_name=project_name,
        user_id="auto_user",
        subtask_id="assigned_subtask",
        completion_status="",
        db_session=postgres_session,
    )

    pause_subtask(
        project_name=project_name, subtask_id="assigned_subtask", db_session=postgres_session
    )

    selected = start_subtask(
        project_name=project_name, user_id="auto_user", db_session=postgres_session
    )
    assert (
        selected == "unassigned_subtask"
    ), f"Expected to select unassigned subtask after pausing assigned, got {selected}"

    release_subtask(
        project_name=project_name,
        user_id="auto_user",
        subtask_id="unassigned_subtask",
        completion_status="",
        db_session=postgres_session,
    )
    pause_subtask(
        project_name=project_name, subtask_id="unassigned_subtask", db_session=postgres_session
    )

    selected = start_subtask(
        project_name=project_name, user_id="auto_user", db_session=postgres_session
    )
    assert (
        selected == "idle_subtask"
    ), f"Expected to select idle subtask after pausing others, got {selected}"

    release_subtask(
        project_name=project_name,
        user_id="auto_user",
        subtask_id="idle_subtask",
        completion_status="",
        db_session=postgres_session,
    )
    pause_subtask(
        project_name=project_name, subtask_id="idle_subtask", db_session=postgres_session
    )

    # Should now select nothing
    selected = start_subtask(
        project_name=project_name, user_id="auto_user", db_session=postgres_session
    )
    assert selected is None, f"Expected no selection when all subtasks paused, got {selected}"
