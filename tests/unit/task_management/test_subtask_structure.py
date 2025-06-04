# pylint: disable=redefined-outer-name,unused-argument
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select

from zetta_utils.task_management.db import get_db_session
from zetta_utils.task_management.db.models import Base, DependencyModel, SubtaskModel
from zetta_utils.task_management.subtask_structure import create_subtask_structure
from zetta_utils.task_management.subtask_type import create_subtask_type
from zetta_utils.task_management.task import create_task, get_task
from zetta_utils.task_management.types import SubtaskType, Task


def setup_structure_test_scenario(db_session, project_name):
    """Set up scenario for testing create_subtask_structure race conditions"""
    # Create required subtask types
    for subtask_type in [
        "segmentation_proofread",
        "segmentation_verify",
        "segmentation_proofread_expert",
    ]:
        create_subtask_type(
            project_name=project_name,
            data=SubtaskType(subtask_type=subtask_type, completion_statuses=["done", "need_help"]),
            db_session=db_session,
        )

    # Create tasks for concurrent structure creation
    for i in range(1, 4):
        create_task(
            project_name=project_name,
            data=Task(
                task_id=f"structure_task_{i}",
                batch_id=f"batch_{i}",
                status="pending",
                task_type="segmentation",
                ng_state="http://example.com/structure",
            ),
            db_session=db_session,
        )

    db_session.commit()


def create_structure_worker(postgres_container, project_name, task_id, structure_name):
    """Worker function to test create_subtask_structure race conditions"""
    connection_url = postgres_container.get_connection_url()
    session = get_db_session(engine_url=connection_url)
    try:
        result = create_subtask_structure(
            project_name=project_name,
            task_id=task_id,
            subtask_structure=structure_name,
            subtask_structure_kwargs={},
            priority=1,
            db_session=session,
        )
        session.commit()
        return "SUCCESS" if result else "FAILED"
    except Exception as e:  # pylint: disable=broad-exception-caught
        session.rollback()
        return f"ERROR: {e}"
    finally:
        session.close()


def test_create_subtask_structure_concurrent_creation(
    clean_db, postgres_container, postgres_session, project_name
):
    """
    Test that concurrent creation of subtask structures works correctly without race conditions.
    Each structure should be created atomically as a complete unit.
    """
    # pylint: disable=too-many-locals
    N_ITERATIONS = 2  # Reduced for simplicity

    for iteration in range(N_ITERATIONS):
        print(f"\n=== Structure Creation Iteration {iteration + 1}/{N_ITERATIONS} ===")

        # Clean up database state for this iteration
        postgres_session.rollback()

        Base.metadata.drop_all(postgres_session.bind)
        Base.metadata.create_all(postgres_session.bind)
        postgres_session.commit()

        # Set up scenario for this iteration
        setup_structure_test_scenario(postgres_session, project_name)

        # Run concurrent subtask structure creation
        structure_name = "segmentation_proofread_1pass"
        task_ids = ["structure_task_1", "structure_task_2", "structure_task_3"]

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for task_id in task_ids:
                future = executor.submit(
                    create_structure_worker,
                    postgres_container,
                    project_name,
                    task_id,
                    structure_name,
                )
                futures.append((task_id, future))

            # Collect results
            results = {}
            for task_id, future in futures:
                result = future.result(timeout=15)
                results[task_id] = result

        print(f"Structure creation results: {results}")

        # All should succeed
        successes = [r for r in results.values() if r == "SUCCESS"]
        errors = [r for r in results.values() if r.startswith("ERROR")]

        assert len(successes) == 3, f"Expected 3 successes, got {len(successes)}: {results}"
        assert len(errors) == 0, f"Expected 0 errors, got {len(errors)}: {results}"

        # Check that each task has the complete structure created atomically
        for task_id in task_ids:
            # Check task status was updated
            task = get_task(
                project_name=project_name, task_id=task_id, db_session=postgres_session
            )
            assert (
                task["status"] == "ingested"
            ), f"Task {task_id} should have status 'ingested', got {task['status']}"

            # Check that all expected subtasks were created
            subtask_query = (
                select(SubtaskModel)
                .where(SubtaskModel.project_name == project_name)
                .where(SubtaskModel.task_id == task_id)
            )
            subtasks = postgres_session.execute(subtask_query).scalars().all()

            # Should have exactly 3 subtasks for this structure
            assert (
                len(subtasks) == 3
            ), f"Task {task_id} should have 3 subtasks, got {len(subtasks)}"

            # Check that all subtask types are correct
            subtask_types = [s.subtask_type for s in subtasks]
            expected_types = [
                "segmentation_proofread",
                "segmentation_verify",
                "segmentation_proofread_expert",
            ]
            assert set(subtask_types) == set(expected_types), (
                f"Task {task_id} subtask types mismatch: "
                f"got {subtask_types}, expected {expected_types}"
            )

            # Check that dependencies were created
            dep_query = select(DependencyModel).where(DependencyModel.project_name == project_name)
            dependencies = postgres_session.execute(dep_query).scalars().all()

            # Should have dependencies for each task (2 per task)
            task_dependencies = [d for d in dependencies if task_id in d.subtask_id]
            assert (
                len(task_dependencies) == 2
            ), f"Task {task_id} should have 2 dependencies, got {len(task_dependencies)}"

            print(f"Task {task_id} structure verified ✓")

        print(f"Structure creation iteration {iteration + 1} passed ✓")

    print(f"\nAll {N_ITERATIONS} structure creation iterations passed! 🎉")
