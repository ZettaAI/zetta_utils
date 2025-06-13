# pylint: disable=redefined-outer-name,unused-argument
import pytest

from zetta_utils.task_management.dependency import (
    create_dependency,
    get_dependencies_depending_on_task,
    get_dependencies_for_task,
    get_dependency,
    get_unsatisfied_dependencies_for_task,
    update_dependency,
)
from zetta_utils.task_management.task import (
    create_task,
    get_task,
    release_task,
    start_task,
)
from zetta_utils.task_management.task_type import create_task_type
from zetta_utils.task_management.types import (
    Dependency,
    DependencyUpdate,
    Task,
    TaskType,
)


@pytest.fixture
def sample_tasks() -> list[Task]:
    return [
        Task(
            **{
                "job_id": "job_1",
                "task_id": "task_1",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "ng_state": {"url": "http://example.com"},
                "ng_state_initial": {"url": "http://example.com"},
                "priority": 1,
                "batch_id": "batch_1",
                "task_type": "segmentation_proofread",
                "is_active": True,
                "last_leased_ts": 0.0,
                "completion_status": "",
            }
        ),
        Task(
            **{
                "job_id": "job_1",
                "task_id": "task_2",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "ng_state": {"url": "http://example.com"},
                "ng_state_initial": {"url": "http://example.com"},
                "priority": 1,
                "batch_id": "batch_1",
                "task_type": "segmentation_proofread",
                "is_active": False,  # Initially inactive until dependency is satisfied
                "last_leased_ts": 0.0,
                "completion_status": "",
            }
        ),
    ]


@pytest.fixture
def existing_tasks(
    project_name, existing_task_type, sample_tasks, db_session, job_factory
):
    # Create the job first using factory
    job_factory("job_1")

    # Create the tasks
    for task in sample_tasks:
        create_task(db_session=db_session, project_name=project_name, data=task)

    # Yield the data for the test to use
    yield sample_tasks


@pytest.fixture
def sample_dependency() -> Dependency:
    return Dependency(
        **{
            "dependency_id": "dep_1",
            "task_id": "task_2",
            "dependent_on_task_id": "task_1",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )


@pytest.fixture
def existing_dependency(project_name, existing_tasks, sample_dependency, db_session):
    create_dependency(db_session=db_session, project_name=project_name, data=sample_dependency)
    yield sample_dependency


def test_get_dependency_success(clean_db, existing_dependency, project_name, db_session):
    """Test retrieving a dependency that exists"""
    result = get_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_1"
    )
    assert result == existing_dependency


def test_get_dependency_not_found(clean_db, project_name, db_session):
    """Test retrieving a dependency that doesn't exist"""
    with pytest.raises(KeyError, match="Dependency dep_1 not found"):
        get_dependency(db_session=db_session, project_name=project_name, dependency_id="dep_1")


def test_create_dependency_success(clean_db, existing_tasks, project_name, db_session):
    """Test creating a new dependency"""
    dependency_data: Dependency = {
        "dependency_id": "dep_1",
        "task_id": "task_2",
        "dependent_on_task_id": "task_1",
        "required_completion_status": "done",
        "is_satisfied": False,
    }
    result = create_dependency(
        db_session=db_session, project_name=project_name, data=dependency_data
    )
    assert result == dependency_data["dependency_id"]

    # Check that the dependency was created in database
    created_dep = get_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_1"
    )
    assert created_dep["is_satisfied"] is False


def test_create_dependency_nonexistent_task(
    project_name, clean_db, existing_task_type, db_session
):
    """Test that creating a dependency with a nonexistent task raises an error"""
    # Create a dependency with a nonexistent task
    dependency_data = Dependency(
        **{
            "dependency_id": "dep_nonexistent",
            "task_id": "nonexistent_task",
            "dependent_on_task_id": "another_nonexistent_task",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )

    # This should raise a ValueError with a specific message
    with pytest.raises(ValueError, match="Task nonexistent_task not found"):
        create_dependency(db_session=db_session, project_name=project_name, data=dependency_data)


def test_update_dependency_success(clean_db, existing_dependency, project_name, db_session):
    """Test updating a dependency"""
    update_data = DependencyUpdate(required_completion_status="not_done")
    result = update_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_1", data=update_data
    )
    assert result is True

    # Check that the dependency was updated in database
    updated_dep = get_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_1"
    )
    assert updated_dep["required_completion_status"] == "not_done"


def test_update_dependency_not_found(clean_db, project_name, db_session):
    """Test updating a dependency that doesn't exist"""
    update_data = DependencyUpdate(required_completion_status="not_done")
    with pytest.raises(KeyError, match="Dependency dep_1 not found"):
        update_dependency(
            db_session=db_session,
            project_name=project_name,
            dependency_id="dep_1",
            data=update_data,
        )


def test_dependent_task_activation(
    clean_db, existing_dependency, existing_user, project_name, db_session
):
    """Test that completing a task activates dependent tasks"""
    # First verify task_2 is inactive
    task_before = get_task(
        db_session=db_session, project_name=project_name, task_id="task_2"
    )
    assert task_before["is_active"] is False

    # Start and complete task_1
    start_task(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
    )
    release_task(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        task_id="task_1",
        completion_status="done",
    )

    # Verify task_2 was activated
    task_after = get_task(
        db_session=db_session, project_name=project_name, task_id="task_2"
    )
    assert task_after["is_active"] is True


def test_create_dependency_duplicate(clean_db, existing_dependency, project_name, db_session):
    """Test creating a dependency that already exists"""
    duplicate_dependency = Dependency(
        **{
            "dependency_id": "dep_1",
            "task_id": "task_2",
            "dependent_on_task_id": "task_1",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )
    with pytest.raises(ValueError, match="Dependency dep_1 already exists"):
        create_dependency(
            db_session=db_session, project_name=project_name, data=duplicate_dependency
        )


def test_create_dependency_same_task(clean_db, existing_tasks, project_name, db_session):
    """Test creating a dependency where a task depends on itself"""
    invalid_dependency = Dependency(
        **{
            "dependency_id": "dep_1",
            "task_id": "task_1",
            "dependent_on_task_id": "task_1",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )
    with pytest.raises(ValueError, match="Task cannot depend on itself"):
        create_dependency(
            db_session=db_session, project_name=project_name, data=invalid_dependency
        )


def test_create_dependency_nonexistent_dependent_on_task(clean_db, project_name, db_session):
    """Test creating a dependency with a nonexistent dependent-on task"""
    # First create the task type
    task_type = TaskType(
        **{"task_type": "segmentation_proofread", "completion_statuses": ["done", "need_help"]}
    )
    create_task_type(db_session=db_session, project_name=project_name, data=task_type)

    # Then create the dependent task
    task_data = Task(
        **{
            "job_id": "job_1",
            "task_id": "task_1",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": {"url": "http://example.com"},
            "ng_state_initial": {"url": "http://example.com"},
            "priority": 1,
            "batch_id": "batch_1",
            "task_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    create_task(db_session=db_session, project_name=project_name, data=task_data)

    # Now create the dependency with a nonexistent dependent-on task
    dependency_data = Dependency(
        **{
            "dependency_id": "dep_1",
            "task_id": "task_1",
            "dependent_on_task_id": "nonexistent_task",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )
    with pytest.raises(ValueError, match="Task nonexistent_task not found"):
        create_dependency(db_session=db_session, project_name=project_name, data=dependency_data)


def test_dependency_satisfaction_complex_conditions(
    project_name, clean_db, existing_task_type, db_session, job_factory, task_factory
):
    """Test complex dependency satisfaction conditions"""
    # Create job and tasks using factory fixtures
    job_factory("job_1")

    task_factory("job_1", "task_complex_1", is_active=True)
    task_factory("job_1", "task_complex_2", is_active=True)
    task_factory("job_1", "task_complex_3", is_active=False)

    # Create dependencies
    dependency1 = Dependency(
        **{
            "dependency_id": "dep_complex_1",
            "task_id": "task_complex_3",
            "dependent_on_task_id": "task_complex_1",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )
    dependency2 = Dependency(
        **{
            "dependency_id": "dep_complex_2",
            "task_id": "task_complex_3",
            "dependent_on_task_id": "task_complex_2",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )

    create_dependency(db_session=db_session, project_name=project_name, data=dependency1)
    create_dependency(db_session=db_session, project_name=project_name, data=dependency2)


def test_check_dependencies_satisfied_wrong_status(
    clean_db,
    project_name,
    existing_user,
    existing_task_type,
    db_session,
    job_factory,
    task_factory,
):
    """Test that dependencies are not satisfied when completion status doesn't match"""
    # Create job and tasks using factory fixtures
    job_factory("job_1")

    task_factory("job_1", "task_status_1", is_active=True)
    task_factory("job_1", "task_status_2", is_active=False)
    task_factory("job_1", "task_status_3", is_active=False)

    # Create dependencies
    dependency1 = Dependency(
        **{
            "dependency_id": "dep_status_1",
            "task_id": "task_status_2",
            "dependent_on_task_id": "task_status_1",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )
    dependency2 = Dependency(
        **{
            "dependency_id": "dep_status_2",
            "task_id": "task_status_3",
            "dependent_on_task_id": "task_status_1",
            "required_completion_status": "need_help",
            "is_satisfied": False,
        }
    )
    create_dependency(db_session=db_session, project_name=project_name, data=dependency1)
    create_dependency(db_session=db_session, project_name=project_name, data=dependency2)

    # Start and complete task1 with "need_help" status
    start_task(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        task_id="task_status_1",
    )
    release_task(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        task_id="task_status_1",
        completion_status="need_help",
    )

    # Verify dependency1 (requiring "done") is not satisfied
    dep1_after = get_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_status_1"
    )
    assert not dep1_after["is_satisfied"]

    # Verify dependency2 (requiring "need_help") is satisfied
    dep2_after = get_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_status_2"
    )
    assert dep2_after["is_satisfied"]

    # Now complete with "done" status
    start_task(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        task_id="task_status_1",
    )
    release_task(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        task_id="task_status_1",
        completion_status="done",
    )

    # Verify dependency1 (requiring "done") is now satisfied
    dep1_final = get_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_status_1"
    )
    assert dep1_final["is_satisfied"]


def test_update_dependency_is_satisfied(clean_db, existing_dependency, project_name, db_session):
    """Test updating a dependency's is_satisfied field - covers line 127"""
    # Update is_satisfied field
    update_data = DependencyUpdate(is_satisfied=True)
    result = update_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_1", data=update_data
    )
    assert result is True

    # Verify the dependency was updated
    updated_dep = get_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_1"
    )
    assert updated_dep["is_satisfied"] is True


def test_update_dependency_task_id(
    clean_db, existing_dependency, project_name, existing_tasks, db_session
):
    """Test updating a dependency's task_id field - covers line 131"""
    # Update task_id field
    update_data = DependencyUpdate(task_id="task_1")
    result = update_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_1", data=update_data
    )
    assert result is True

    # Verify the dependency was updated
    updated_dep = get_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_1"
    )
    assert updated_dep["task_id"] == "task_1"


def test_update_dependency_dependent_on_task_id(
    clean_db, existing_dependency, project_name, existing_tasks, db_session
):
    """Test updating a dependency's dependent_on_task_id field - covers line 133"""
    # Update dependent_on_task_id field
    update_data = DependencyUpdate(dependent_on_task_id="task_2")
    result = update_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_1", data=update_data
    )
    assert result is True

    # Verify the dependency was updated
    updated_dep = get_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_1"
    )
    assert updated_dep["dependent_on_task_id"] == "task_2"


def test_get_dependencies_for_task(clean_db, existing_dependency, project_name, db_session):
    """Test getting all dependencies for a specific task - covers lines 161-168"""
    # Get dependencies for task_2 (which has dependency dep_1)
    dependencies = get_dependencies_for_task(
        db_session=db_session, project_name=project_name, task_id="task_2"
    )

    # Should return one dependency
    assert len(dependencies) == 1
    assert dependencies[0]["dependency_id"] == "dep_1"
    assert dependencies[0]["task_id"] == "task_2"
    assert dependencies[0]["dependent_on_task_id"] == "task_1"

    # Test with a task that has no dependencies
    no_deps = get_dependencies_for_task(
        db_session=db_session, project_name=project_name, task_id="task_1"
    )
    assert len(no_deps) == 0


def test_get_dependencies_depending_on_task(
    clean_db, existing_dependency, project_name, db_session
):
    """Test getting all dependencies that depend on a specific task - covers lines 170-176"""
    # Get dependencies that depend on task_1 (dep_1 depends on task_1)
    depending_deps = get_dependencies_depending_on_task(
        db_session=db_session, project_name=project_name, dependent_on_task_id="task_1"
    )

    # Should return one dependency (dep_1)
    assert len(depending_deps) == 1
    assert depending_deps[0]["dependency_id"] == "dep_1"
    assert depending_deps[0]["task_id"] == "task_2"
    assert depending_deps[0]["dependent_on_task_id"] == "task_1"

    # Test with a task that nothing depends on
    no_depending = get_dependencies_depending_on_task(
        db_session=db_session, project_name=project_name, dependent_on_task_id="task_2"
    )
    assert len(no_depending) == 0


def test_get_unsatisfied_dependencies_for_task(
    clean_db, existing_dependency, project_name, db_session
):
    """Test getting unsatisfied dependencies for a specific task - covers lines 190-197"""
    # Initially, dep_1 should be unsatisfied
    unsatisfied_deps = get_unsatisfied_dependencies_for_task(
        db_session=db_session, project_name=project_name, task_id="task_2"
    )
    assert len(unsatisfied_deps) == 1
    assert unsatisfied_deps[0]["dependency_id"] == "dep_1"
    assert unsatisfied_deps[0]["is_satisfied"] is False

    # Now satisfy the dependency
    update_data = DependencyUpdate(is_satisfied=True)
    update_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_1", data=update_data
    )

    # Should now return no unsatisfied dependencies
    no_unsatisfied = get_unsatisfied_dependencies_for_task(
        db_session=db_session, project_name=project_name, task_id="task_2"
    )
    assert len(no_unsatisfied) == 0

    # Test with a task that has no dependencies
    no_deps = get_unsatisfied_dependencies_for_task(
        db_session=db_session, project_name=project_name, task_id="task_1"
    )
    assert len(no_deps) == 0
