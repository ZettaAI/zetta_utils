# pylint: disable=redefined-outer-name,unused-argument
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import select

from zetta_utils.task_management.db import get_db_session
from zetta_utils.task_management.db.models import Base, DependencyModel, TaskModel
from zetta_utils.task_management.job import create_job, get_job
from zetta_utils.task_management.task_structure import create_task_structure
from zetta_utils.task_management.task_type import create_task_type
from zetta_utils.task_management.types import Job, TaskType


def setup_structure_test_scenario(db_session, project_name):
    """Set up scenario for testing create_task_structure race conditions"""
    # Create required task types
    for task_type in [
        "segmentation_proofread",
        "segmentation_verify",
        "segmentation_proofread_expert",
    ]:
        create_task_type(
            project_name=project_name,
            data=TaskType(task_type=task_type, completion_statuses=["done", "need_help"]),
            db_session=db_session,
        )

    # Create jobs for concurrent structure creation
    for i in range(1, 4):
        create_job(
            project_name=project_name,
            data=Job(
                job_id=f"structure_job_{i}",
                batch_id=f"batch_{i}",
                status="pending",
                job_type="segmentation",
                ng_state={"url": "http://example.com/structure"},
            ),
            db_session=db_session,
        )

    db_session.commit()


def create_structure_worker(postgres_container, project_name, job_id, structure_name):
    """Worker function to test create_task_structure race conditions"""
    connection_url = postgres_container.get_connection_url()
    session = get_db_session(engine_url=connection_url)
    try:
        result = create_task_structure(
            project_name=project_name,
            job_id=job_id,
            task_structure=structure_name,
            task_structure_kwargs={},
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


def test_create_task_structure_nonexistent_structure(
    clean_db, postgres_session, project_name
):
    """Test that creating a task structure with a nonexistent structure name raises ValueError"""
    # Create required task types first
    create_task_type(
        project_name=project_name,
        data=TaskType(task_type="segmentation_proofread", completion_statuses=["done", "need_help"]),
        db_session=postgres_session,
    )
    
    # Create a job
    create_job(
        project_name=project_name,
        data=Job(
            job_id="test_job",
            batch_id="test_batch",
            status="pending",
            job_type="segmentation",
            ng_state={"url": "http://example.com/test"},
        ),
        db_session=postgres_session,
    )
    
    # Try to create task structure with nonexistent structure name
    with pytest.raises(RuntimeError, match="Failed to create task structure: Task structure 'nonexistent_structure' is not registered"):
        create_task_structure(
            project_name=project_name,
            job_id="test_job",
            task_structure="nonexistent_structure",
            task_structure_kwargs={},
            priority=1,
            db_session=postgres_session,
        )


def test_create_task_structure_two_path(
    clean_db, postgres_session, project_name
):
    """Test creating a task structure with validation layer path"""
    # Create required task types first
    for task_type in [
        "segmentation_proofread",
        "segmentation_verify", 
        "segmentation_proofread_expert",
    ]:
        create_task_type(
            project_name=project_name,
            data=TaskType(task_type=task_type, completion_statuses=["done", "need_help"]),
            db_session=postgres_session,
        )
    
    # Create a job
    create_job(
        project_name=project_name,
        data=Job(
            job_id="test_job_two_path",
            batch_id="test_batch",
            status="pending",
            job_type="segmentation",
            ng_state={"layers": [{"source": "test", "type": "image", "name": "original"}]},
        ),
        db_session=postgres_session,
    )
    
    # Create task structure with validation layer path
    result = create_task_structure(
        project_name=project_name,
        job_id="test_job_two_path",
        task_structure="segmentation_proofread_two_path",
        task_structure_kwargs={"validation_layer_path": "gs://test-bucket/validation"},
        priority=1,
        db_session=postgres_session,
    )
    
    assert result is True
    
    # Check that job status was updated
    job = get_job(project_name=project_name, job_id="test_job_two_path", db_session=postgres_session)
    assert job["status"] == "ingested"
    
    # Check that all expected tasks were created
    task_query = (
        select(TaskModel)
        .where(TaskModel.project_name == project_name)
        .where(TaskModel.job_id == "test_job_two_path")
    )
    tasks = postgres_session.execute(task_query).scalars().all()
    
    # Should have exactly 3 tasks for this structure
    assert len(tasks) == 3
    
    # Check that all task types are correct
    task_types = [s.task_type for s in tasks]
    expected_types = [
        "segmentation_proofread",
        "segmentation_verify",
        "segmentation_proofread_expert",
    ]
    assert set(task_types) == set(expected_types)
    
    # Check that the verify task has the validation layer in ng_state
    verify_task = next(t for t in tasks if t.task_type == "segmentation_verify")
    ng_state = verify_task.ng_state
    assert "layers" in ng_state
    validation_layer = next(layer for layer in ng_state["layers"] if layer["name"] == "validation")
    assert validation_layer["source"] == "gs://test-bucket/validation"
    assert validation_layer["type"] == "segmentation"


def test_create_task_structure_concurrent_creation(
    clean_db, postgres_container, postgres_session, project_name
):
    """
    Test that concurrent creation of task structures works correctly without race conditions.
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

        # Run concurrent task structure creation
        structure_name = "segmentation_proofread_1pass"
        job_ids = ["structure_job_1", "structure_job_2", "structure_job_3"]

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for job_id in job_ids:
                future = executor.submit(
                    create_structure_worker,
                    postgres_container,
                    project_name,
                    job_id,
                    structure_name,
                )
                futures.append((job_id, future))

            # Collect results
            results = {}
            for job_id, future in futures:
                result = future.result(timeout=15)
                results[job_id] = result

        print(f"Structure creation results: {results}")

        # All should succeed
        successes = [r for r in results.values() if r == "SUCCESS"]
        errors = [r for r in results.values() if r.startswith("ERROR")]

        assert len(successes) == 3, f"Expected 3 successes, got {len(successes)}: {results}"
        assert len(errors) == 0, f"Expected 0 errors, got {len(errors)}: {results}"

        # Check that each job has the complete structure created atomically
        for job_id in job_ids:
            # Check job status was updated
            job = get_job(project_name=project_name, job_id=job_id, db_session=postgres_session)
            assert (
                job["status"] == "ingested"
            ), f"Job {job_id} should have status 'ingested', got {job['status']}"

            # Check that all expected tasks were created
            task_query = (
                select(TaskModel)
                .where(TaskModel.project_name == project_name)
                .where(TaskModel.job_id == job_id)
            )
            tasks = postgres_session.execute(task_query).scalars().all()

            # Should have exactly 3 tasks for this structure
            assert len(tasks) == 3, f"Job {job_id} should have 3 tasks, got {len(tasks)}"

            # Check that all task types are correct
            task_types = [s.task_type for s in tasks]
            expected_types = [
                "segmentation_proofread",
                "segmentation_verify",
                "segmentation_proofread_expert",
            ]
            assert set(task_types) == set(expected_types), (
                f"Job {job_id} task types mismatch: "
                f"got {task_types}, expected {expected_types}"
            )

            # Check that dependencies were created
            dep_query = select(DependencyModel).where(DependencyModel.project_name == project_name)
            dependencies = postgres_session.execute(dep_query).scalars().all()

            # Should have dependencies for each job (2 per job)
            job_dependencies = [d for d in dependencies if job_id in d.task_id]
            assert (
                len(job_dependencies) == 2
            ), f"Job {job_id} should have 2 dependencies, got {len(job_dependencies)}"

            print(f"Job {job_id} structure verified ✓")

        print(f"Structure creation iteration {iteration + 1} passed ✓")

    print(f"\nAll {N_ITERATIONS} structure creation iterations passed! 🎉")
