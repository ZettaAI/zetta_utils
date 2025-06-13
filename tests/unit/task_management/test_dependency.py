# pylint: disable=redefined-outer-name,unused-argument
import pytest

from zetta_utils.task_management.dependency import (
    create_dependency,
    get_dependencies_depending_on_subtask,
    get_dependencies_for_subtask,
    get_dependency,
    get_unsatisfied_dependencies_for_subtask,
    update_dependency,
)
from zetta_utils.task_management.subtask import (
    create_subtask,
    get_subtask,
    release_subtask,
    start_subtask,
)
from zetta_utils.task_management.subtask_type import create_subtask_type
from zetta_utils.task_management.types import (
    Dependency,
    DependencyUpdate,
    Subtask,
    SubtaskType,
)


@pytest.fixture
def sample_subtasks() -> list[Subtask]:
    return [
        Subtask(
            **{
                "job_id": "job_1",
                "subtask_id": "subtask_1",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "ng_state": "http://example.com",
                "ng_state_initial": "http://example.com",
                "priority": 1,
                "batch_id": "batch_1",
                "subtask_type": "segmentation_proofread",
                "is_active": True,
                "last_leased_ts": 0.0,
                "completion_status": "",
            }
        ),
        Subtask(
            **{
                "job_id": "job_1",
                "subtask_id": "subtask_2",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "ng_state": "http://example.com",
                "ng_state_initial": "http://example.com",
                "priority": 1,
                "batch_id": "batch_1",
                "subtask_type": "segmentation_proofread",
                "is_active": False,  # Initially inactive until dependency is satisfied
                "last_leased_ts": 0.0,
                "completion_status": "",
            }
        ),
    ]


@pytest.fixture
def existing_subtasks(
    project_name, existing_subtask_type, sample_subtasks, db_session, job_factory
):
    # Create the job first using factory
    job_factory("job_1")

    # Create the subtasks
    for subtask in sample_subtasks:
        create_subtask(db_session=db_session, project_name=project_name, data=subtask)

    # Yield the data for the test to use
    yield sample_subtasks


@pytest.fixture
def sample_dependency() -> Dependency:
    return Dependency(
        **{
            "dependency_id": "dep_1",
            "subtask_id": "subtask_2",
            "dependent_on_subtask_id": "subtask_1",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )


@pytest.fixture
def existing_dependency(project_name, existing_subtasks, sample_dependency, db_session):
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


def test_create_dependency_success(clean_db, existing_subtasks, project_name, db_session):
    """Test creating a new dependency"""
    dependency_data: Dependency = {
        "dependency_id": "dep_1",
        "subtask_id": "subtask_2",
        "dependent_on_subtask_id": "subtask_1",
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


def test_create_dependency_nonexistent_subtask(
    project_name, clean_db, existing_subtask_type, db_session
):
    """Test that creating a dependency with a nonexistent subtask raises an error"""
    # Create a dependency with a nonexistent subtask
    dependency_data = Dependency(
        **{
            "dependency_id": "dep_nonexistent",
            "subtask_id": "nonexistent_subtask",
            "dependent_on_subtask_id": "another_nonexistent_subtask",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )

    # This should raise a ValueError with a specific message
    with pytest.raises(ValueError, match="Subtask nonexistent_subtask not found"):
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


def test_dependent_subtask_activation(
    clean_db, existing_dependency, existing_user, project_name, db_session
):
    """Test that completing a subtask activates dependent subtasks"""
    # First verify subtask_2 is inactive
    subtask_before = get_subtask(
        db_session=db_session, project_name=project_name, subtask_id="subtask_2"
    )
    assert subtask_before["is_active"] is False

    # Start and complete subtask_1
    start_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
    )
    release_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_1",
        completion_status="done",
    )

    # Verify subtask_2 was activated
    subtask_after = get_subtask(
        db_session=db_session, project_name=project_name, subtask_id="subtask_2"
    )
    assert subtask_after["is_active"] is True


def test_create_dependency_duplicate(clean_db, existing_dependency, project_name, db_session):
    """Test creating a dependency that already exists"""
    duplicate_dependency = Dependency(
        **{
            "dependency_id": "dep_1",
            "subtask_id": "subtask_2",
            "dependent_on_subtask_id": "subtask_1",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )
    with pytest.raises(ValueError, match="Dependency dep_1 already exists"):
        create_dependency(
            db_session=db_session, project_name=project_name, data=duplicate_dependency
        )


def test_create_dependency_same_subtask(clean_db, existing_subtasks, project_name, db_session):
    """Test creating a dependency where a subtask depends on itself"""
    invalid_dependency = Dependency(
        **{
            "dependency_id": "dep_1",
            "subtask_id": "subtask_1",
            "dependent_on_subtask_id": "subtask_1",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )
    with pytest.raises(ValueError, match="Subtask cannot depend on itself"):
        create_dependency(
            db_session=db_session, project_name=project_name, data=invalid_dependency
        )


def test_create_dependency_nonexistent_dependent_on_subtask(clean_db, project_name, db_session):
    """Test creating a dependency with a nonexistent dependent-on subtask"""
    # First create the subtask type
    subtask_type = SubtaskType(
        **{"subtask_type": "segmentation_proofread", "completion_statuses": ["done", "need_help"]}
    )
    create_subtask_type(db_session=db_session, project_name=project_name, data=subtask_type)

    # Then create the dependent subtask
    subtask_data = Subtask(
        **{
            "job_id": "job_1",
            "subtask_id": "subtask_1",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com",
            "ng_state_initial": "http://example.com",
            "priority": 1,
            "batch_id": "batch_1",
            "subtask_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    create_subtask(db_session=db_session, project_name=project_name, data=subtask_data)

    # Now create the dependency with a nonexistent dependent-on subtask
    dependency_data = Dependency(
        **{
            "dependency_id": "dep_1",
            "subtask_id": "subtask_1",
            "dependent_on_subtask_id": "nonexistent_subtask",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )
    with pytest.raises(ValueError, match="Subtask nonexistent_subtask not found"):
        create_dependency(db_session=db_session, project_name=project_name, data=dependency_data)


def test_dependency_satisfaction_complex_conditions(
    project_name, clean_db, existing_subtask_type, db_session, job_factory, subtask_factory
):
    """Test complex dependency satisfaction conditions"""
    # Create job and subtasks using factory fixtures
    job_factory("job_1")

    subtask_factory("job_1", "subtask_complex_1", is_active=True)
    subtask_factory("job_1", "subtask_complex_2", is_active=True)
    subtask_factory("job_1", "subtask_complex_3", is_active=False)

    # Create dependencies
    dependency1 = Dependency(
        **{
            "dependency_id": "dep_complex_1",
            "subtask_id": "subtask_complex_3",
            "dependent_on_subtask_id": "subtask_complex_1",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )
    dependency2 = Dependency(
        **{
            "dependency_id": "dep_complex_2",
            "subtask_id": "subtask_complex_3",
            "dependent_on_subtask_id": "subtask_complex_2",
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
    existing_subtask_type,
    db_session,
    job_factory,
    subtask_factory,
):
    """Test that dependencies are not satisfied when completion status doesn't match"""
    # Create job and subtasks using factory fixtures
    job_factory("job_1")

    subtask_factory("job_1", "subtask_status_1", is_active=True)
    subtask_factory("job_1", "subtask_status_2", is_active=False)
    subtask_factory("job_1", "subtask_status_3", is_active=False)

    # Create dependencies
    dependency1 = Dependency(
        **{
            "dependency_id": "dep_status_1",
            "subtask_id": "subtask_status_2",
            "dependent_on_subtask_id": "subtask_status_1",
            "required_completion_status": "done",
            "is_satisfied": False,
        }
    )
    dependency2 = Dependency(
        **{
            "dependency_id": "dep_status_2",
            "subtask_id": "subtask_status_3",
            "dependent_on_subtask_id": "subtask_status_1",
            "required_completion_status": "need_help",
            "is_satisfied": False,
        }
    )
    create_dependency(db_session=db_session, project_name=project_name, data=dependency1)
    create_dependency(db_session=db_session, project_name=project_name, data=dependency2)

    # Start and complete subtask1 with "need_help" status
    start_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_status_1",
    )
    release_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_status_1",
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
    start_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_status_1",
    )
    release_subtask(
        db_session=db_session,
        project_name=project_name,
        user_id="test_user",
        subtask_id="subtask_status_1",
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


def test_update_dependency_subtask_id(
    clean_db, existing_dependency, project_name, existing_subtasks, db_session
):
    """Test updating a dependency's subtask_id field - covers line 131"""
    # Update subtask_id field
    update_data = DependencyUpdate(subtask_id="subtask_1")
    result = update_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_1", data=update_data
    )
    assert result is True

    # Verify the dependency was updated
    updated_dep = get_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_1"
    )
    assert updated_dep["subtask_id"] == "subtask_1"


def test_update_dependency_dependent_on_subtask_id(
    clean_db, existing_dependency, project_name, existing_subtasks, db_session
):
    """Test updating a dependency's dependent_on_subtask_id field - covers line 133"""
    # Update dependent_on_subtask_id field
    update_data = DependencyUpdate(dependent_on_subtask_id="subtask_2")
    result = update_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_1", data=update_data
    )
    assert result is True

    # Verify the dependency was updated
    updated_dep = get_dependency(
        db_session=db_session, project_name=project_name, dependency_id="dep_1"
    )
    assert updated_dep["dependent_on_subtask_id"] == "subtask_2"


def test_get_dependencies_for_subtask(clean_db, existing_dependency, project_name, db_session):
    """Test getting all dependencies for a specific subtask - covers lines 161-168"""
    # Get dependencies for subtask_2 (which has dependency dep_1)
    dependencies = get_dependencies_for_subtask(
        db_session=db_session, project_name=project_name, subtask_id="subtask_2"
    )

    # Should return one dependency
    assert len(dependencies) == 1
    assert dependencies[0]["dependency_id"] == "dep_1"
    assert dependencies[0]["subtask_id"] == "subtask_2"
    assert dependencies[0]["dependent_on_subtask_id"] == "subtask_1"

    # Test with a subtask that has no dependencies
    no_deps = get_dependencies_for_subtask(
        db_session=db_session, project_name=project_name, subtask_id="subtask_1"
    )
    assert len(no_deps) == 0


def test_get_dependencies_depending_on_subtask(
    clean_db, existing_dependency, project_name, db_session
):
    """Test getting all dependencies that depend on a specific subtask - covers lines 170-176"""
    # Get dependencies that depend on subtask_1 (dep_1 depends on subtask_1)
    depending_deps = get_dependencies_depending_on_subtask(
        db_session=db_session, project_name=project_name, dependent_on_subtask_id="subtask_1"
    )

    # Should return one dependency (dep_1)
    assert len(depending_deps) == 1
    assert depending_deps[0]["dependency_id"] == "dep_1"
    assert depending_deps[0]["subtask_id"] == "subtask_2"
    assert depending_deps[0]["dependent_on_subtask_id"] == "subtask_1"

    # Test with a subtask that nothing depends on
    no_depending = get_dependencies_depending_on_subtask(
        db_session=db_session, project_name=project_name, dependent_on_subtask_id="subtask_2"
    )
    assert len(no_depending) == 0


def test_get_unsatisfied_dependencies_for_subtask(
    clean_db, existing_dependency, project_name, db_session
):
    """Test getting unsatisfied dependencies for a specific subtask - covers lines 190-197"""
    # Initially, dep_1 should be unsatisfied
    unsatisfied_deps = get_unsatisfied_dependencies_for_subtask(
        db_session=db_session, project_name=project_name, subtask_id="subtask_2"
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
    no_unsatisfied = get_unsatisfied_dependencies_for_subtask(
        db_session=db_session, project_name=project_name, subtask_id="subtask_2"
    )
    assert len(no_unsatisfied) == 0

    # Test with a subtask that has no dependencies
    no_deps = get_unsatisfied_dependencies_for_subtask(
        db_session=db_session, project_name=project_name, subtask_id="subtask_1"
    )
    assert len(no_deps) == 0
