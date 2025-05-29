# pylint: disable=redefined-outer-name,unused-argument
import pytest

from zetta_utils.task_management.dependency import (
    create_dependency,
    get_dependency,
    update_dependency,
)
from zetta_utils.task_management.project import get_collection
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
                "task_id": "task_1",
                "subtask_id": "subtask_1",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "ng_state": "http://example.com",
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
                "task_id": "task_1",
                "subtask_id": "subtask_2",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "ng_state": "http://example.com",
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
def existing_subtasks(project_name, existing_subtask_type, sample_subtasks):
    # Create the subtasks
    for subtask in sample_subtasks:
        create_subtask(project_name, subtask)

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
def existing_dependency(project_name, existing_subtasks, sample_dependency):
    create_dependency(project_name, sample_dependency)
    yield sample_dependency


def test_get_dependency_success(clean_collections, existing_dependency, project_name):
    """Test retrieving a dependency that exists"""
    result = get_dependency(project_name, "dep_1")
    assert result == existing_dependency


def test_get_dependency_not_found(clean_collections, project_name):
    """Test retrieving a dependency that doesn't exist"""
    with pytest.raises(KeyError, match="Dependency dep_1 not found"):
        get_dependency(project_name, "dep_1")


def test_create_dependency_success(clean_collections, existing_subtasks, project_name):
    """Test creating a new dependency"""
    dependency_data: Dependency = {
        "dependency_id": "dep_1",
        "subtask_id": "subtask_2",
        "dependent_on_subtask_id": "subtask_1",
        "required_completion_status": "done",
        "is_satisfied": False,
    }
    result = create_dependency(project_name, dependency_data)
    assert result == dependency_data["dependency_id"]

    # Check that the dependency was created in Firestore
    doc = get_collection(project_name, "dependencies").document("dep_1").get()
    assert doc.exists

    # Check that is_satisfied was set to False by default
    assert doc.to_dict()["is_satisfied"] is False


def test_create_dependency_nonexistent_subtask(
    project_name, clean_collections, existing_subtask_type
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
        create_dependency(project_name, dependency_data)


def test_update_dependency_success(clean_collections, existing_dependency, project_name):
    """Test updating a dependency"""
    update_data = DependencyUpdate(**{"required_completion_status": "not_done"})
    result = update_dependency(project_name, "dep_1", update_data)
    assert result is True

    # Check that the dependency was updated in Firestore
    doc = get_collection(project_name, "dependencies").document("dep_1").get()
    assert doc.to_dict()["required_completion_status"] == "not_done"


def test_update_dependency_not_found(clean_collections, project_name):
    """Test updating a dependency that doesn't exist"""
    update_data = DependencyUpdate(**{"required_completion_status": "not_done"})
    with pytest.raises(KeyError, match="Dependency dep_1 not found"):
        update_dependency(project_name, "dep_1", update_data)


def test_dependent_subtask_activation(
    clean_collections, existing_dependency, existing_user, project_name
):
    """Test that completing a subtask activates dependent subtasks"""
    # First verify subtask_2 is inactive
    subtask_before = get_subtask(project_name, "subtask_2")
    assert subtask_before["is_active"] is False

    # Start and complete subtask_1
    start_subtask(project_name, "test_user_1", "subtask_1")
    release_subtask(project_name, "test_user_1", "subtask_1", "done")

    # Verify subtask_2 was activated
    subtask_after = get_subtask(project_name, "subtask_2")
    assert subtask_after["is_active"] is True


def test_create_dependency_duplicate(clean_collections, existing_dependency, project_name):
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
        create_dependency(project_name, duplicate_dependency)


def test_create_dependency_same_subtask(clean_collections, existing_subtasks, project_name):
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
        create_dependency(project_name, invalid_dependency)


def test_create_dependency_nonexistent_dependent_on_subtask(clean_collections, project_name):
    """Test creating a dependency with a nonexistent dependent-on subtask"""
    # First create the subtask type
    subtask_type = SubtaskType(
        **{"subtask_type": "segmentation_proofread", "completion_statuses": ["done", "need_help"]}
    )
    create_subtask_type(subtask_type)

    # Then create the dependent subtask
    subtask_data = Subtask(
        **{
            "task_id": "task_1",
            "subtask_id": "subtask_1",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com",
            "priority": 1,
            "batch_id": "batch_1",
            "subtask_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    create_subtask(project_name, subtask_data)

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
        create_dependency(project_name, dependency_data)


def test_dependency_satisfaction_complex_conditions(
    project_name, clean_collections, existing_subtask_type
):
    """Test complex dependency satisfaction conditions"""
    # Create three subtasks
    subtask1 = Subtask(
        **{
            "task_id": "task_1",
            "subtask_id": "subtask_complex_1",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com",
            "priority": 1,
            "batch_id": "batch_1",
            "subtask_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    subtask2 = Subtask(
        **{
            "task_id": "task_1",
            "subtask_id": "subtask_complex_2",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com",
            "priority": 2,
            "batch_id": "batch_1",
            "subtask_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    subtask3 = Subtask(
        **{
            "task_id": "task_1",
            "subtask_id": "subtask_complex_3",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com",
            "priority": 3,
            "batch_id": "batch_1",
            "subtask_type": "segmentation_proofread",
            "is_active": False,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )

    create_subtask(project_name, subtask1)
    create_subtask(project_name, subtask2)
    create_subtask(project_name, subtask3)

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

    create_dependency(project_name, dependency1)
    create_dependency(project_name, dependency2)


def test_check_dependencies_satisfied_wrong_status(
    clean_collections, project_name, existing_user, existing_subtask_type
):
    """Test that dependencies are not satisfied when completion status doesn't match"""
    # Create all three subtasks and dependencies
    subtask1 = Subtask(
        **{
            "task_id": "task_1",
            "subtask_id": "subtask_status_1",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com",
            "priority": 1,
            "batch_id": "batch_1",
            "subtask_type": "segmentation_proofread",
            "is_active": True,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    subtask2 = Subtask(
        **{
            "task_id": "task_1",
            "subtask_id": "subtask_status_2",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com",
            "priority": 2,
            "batch_id": "batch_1",
            "subtask_type": "segmentation_proofread",
            "is_active": False,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    subtask3 = Subtask(
        **{
            "task_id": "task_1",
            "subtask_id": "subtask_status_3",
            "assigned_user_id": "",
            "active_user_id": "",
            "completed_user_id": "",
            "ng_state": "http://example.com",
            "priority": 3,
            "batch_id": "batch_1",
            "subtask_type": "segmentation_proofread",
            "is_active": False,
            "last_leased_ts": 0.0,
            "completion_status": "",
        }
    )
    create_subtask(project_name, subtask1)
    create_subtask(project_name, subtask2)
    create_subtask(project_name, subtask3)

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
    create_dependency(project_name, dependency1)
    create_dependency(project_name, dependency2)

    # Start and complete subtask1 with "need_help" status
    start_subtask(project_name, "test_user_1", "subtask_status_1")
    release_subtask(project_name, "test_user_1", "subtask_status_1", "need_help")

    # Verify dependency1 (requiring "done") is not satisfied
    dep1_after = get_dependency(project_name, "dep_status_1")
    assert not dep1_after["is_satisfied"]

    # Verify dependency2 (requiring "need_help") is satisfied
    dep2_after = get_dependency(project_name, "dep_status_2")
    assert dep2_after["is_satisfied"]

    # Now complete with "done" status
    start_subtask(project_name, "test_user_1", "subtask_status_1")
    release_subtask(project_name, "test_user_1", "subtask_status_1", "done")

    # Verify dependency1 (requiring "done") is now satisfied
    dep1_final = get_dependency(project_name, "dep_status_1")
    assert dep1_final["is_satisfied"]
