# pylint: disable=redefined-outer-name,unused-argument
import pytest
from google.cloud import firestore

from zetta_utils.task_management.ingestion import ingest_batch, ingest_task
from zetta_utils.task_management.subtask import create_subtask
from zetta_utils.task_management.subtask_type import create_subtask_type
from zetta_utils.task_management.task import create_task, get_task, update_task
from zetta_utils.task_management.types import Subtask, SubtaskType, Task


@pytest.fixture
def project_name_ingestion() -> str:
    return "test_project_ingestion"


@pytest.fixture(autouse=True)
def clean_collections(firestore_emulator, project_name_ingestion):
    client = firestore.Client()
    collections = [
        f"{project_name_ingestion}_tasks",
        f"{project_name_ingestion}_subtasks",
        "subtask_types",
    ]
    for coll in collections:
        for doc in client.collection(coll).list_documents():
            doc.delete()
    yield
    for coll in collections:
        for doc in client.collection(coll).list_documents():
            doc.delete()


@pytest.fixture
def sample_task() -> Task:
    return Task(
        **{
            "task_id": "task_1",
            "batch_id": "batch_1",
            "status": "pending_ingestion",
            "task_type": "segmentation",
            "link": "http://example.com/task_1",
        }
    )


@pytest.fixture
def existing_task(firestore_emulator, project_name_ingestion, sample_task):
    create_task(project_name_ingestion, sample_task)
    yield sample_task


@pytest.fixture
def sample_subtask_type() -> SubtaskType:
    return SubtaskType(
        **{
            "subtask_type": "segmentation_proofread",
            "completion_statuses": ["done", "need_help"],
            "description": "Proofread segmentation",
        }
    )


@pytest.fixture
def existing_subtask_type(firestore_emulator, sample_subtask_type):
    create_subtask_type(sample_subtask_type)
    yield sample_subtask_type


@pytest.fixture
def sample_subtasks(existing_subtask_type) -> list[Subtask]:
    return [
        Subtask(
            **{
                "task_id": f"task_{i}",
                "subtask_id": f"subtask_{i}",
                "assigned_user_id": "",
                "active_user_id": "",
                "completed_user_id": "",
                "link": f"http://example.com/{i}",
                "priority": i,
                "batch_id": "batch_1",
                "subtask_type": existing_subtask_type["subtask_type"],
                "is_active": True,
                "last_leased_ts": 0.0,
                "completion_status": "",
            }
        )
        for i in range(1, 4)
    ]


@pytest.fixture
def existing_subtasks(firestore_emulator, project_name_ingestion, sample_subtasks):
    for subtask in sample_subtasks:
        create_subtask(project_name_ingestion, subtask)
    yield sample_subtasks


def test_ingest_task(
    firestore_emulator, project_name_ingestion, existing_task, existing_subtask_type
):
    """Test ingesting a task"""
    result = ingest_task(
        project_name=project_name_ingestion,
        task_id="task_1",
        subtask_structure="segmentation_proofread_simple",
        priority=2,
    )
    assert result is True

    task = get_task(project_name_ingestion, "task_1")
    assert task["status"] == "ingested"


def test_ingest_task_already_ingested(
    firestore_emulator, project_name_ingestion, existing_task, existing_subtask_type
):
    """Test that ingesting an already ingested task returns False"""
    ingest_task(
        project_name=project_name_ingestion,
        task_id="task_1",
        subtask_structure="segmentation_proofread_simple",
        priority=2,
    )

    result = ingest_task(
        project_name=project_name_ingestion,
        task_id="task_1",
        subtask_structure="segmentation_proofread_simple",
        priority=2,
    )
    assert result is False


def test_ingest_task_re_ingest(
    firestore_emulator, project_name_ingestion, existing_task, existing_subtask_type
):
    """Test re-ingesting a task"""
    ingest_task(
        project_name=project_name_ingestion,
        task_id="task_1",
        subtask_structure="segmentation_proofread_simple",
        priority=2,
    )

    result = ingest_task(
        project_name=project_name_ingestion,
        task_id="task_1",
        subtask_structure="segmentation_proofread_simple",
        re_ingest="not_processed",
        priority=2,
    )
    assert result is True


def test_ingest_batch(firestore_emulator, project_name_ingestion, existing_subtask_type):
    """Test ingesting a batch of tasks"""
    tasks = [
        Task(
            **{
                "task_id": f"task_{i}",
                "batch_id": "batch_1",
                "status": "pending_ingestion",
                "task_type": "segmentation",
                "link": f"http://example.com/task_{i}",
            }
        )
        for i in range(1, 4)
    ]

    for task in tasks:
        create_task(project_name_ingestion, task)

    result = ingest_batch(project_name_ingestion, "batch_1", "segmentation_proofread_simple")
    assert result is True

    for i in range(1, 4):
        task = get_task(project_name_ingestion, f"task_{i}")
        assert task["status"] == "ingested"


def test_ingest_batch_re_ingest(firestore_emulator, project_name_ingestion, existing_subtask_type):
    tasks = [
        Task(
            **{
                "task_id": f"task_{i}",
                "batch_id": "batch_1",
                "status": "pending_ingestion",
                "task_type": "segmentation",
                "link": f"http://example.com/task_{i}",
            }
        )
        for i in range(1, 4)
    ]

    for task in tasks:
        create_task(project_name_ingestion, task)

    ingest_batch(project_name_ingestion, "batch_1", "segmentation_proofread_simple")

    update_task(project_name_ingestion, "task_3", {"status": "fully_processed"})

    result = ingest_batch(
        project_name_ingestion,
        "batch_1",
        "segmentation_proofread_simple",
        re_ingest="not_processed",
    )
    assert result is True
    # Check that task_1 and task_2 were re-ingested (they were in 'ingested' state)
    task_1 = get_task(project_name_ingestion, "task_1")
    task_2 = get_task(project_name_ingestion, "task_2")
    assert task_1["status"] == "ingested"
    assert task_2["status"] == "ingested"

    # Check that task_3 was not re-ingested (it was in 'fully_processed' state)
    task_3 = get_task(project_name_ingestion, "task_3")
    assert task_3["status"] == "fully_processed"

    # Test re-ingesting all tasks including fully_processed
    result = ingest_batch(
        project_name_ingestion,
        "batch_1",
        "segmentation_proofread_simple",
        re_ingest="all",
    )
    assert result is True

    # Verify all tasks were re-ingested including task_3
    task_3 = get_task(project_name_ingestion, "task_3")
    assert task_3["status"] == "ingested"


def test_ingest_task_nonexistent(firestore_emulator, project_name_ingestion):
    """Test that ingesting a nonexistent task raises KeyError"""
    with pytest.raises(KeyError, match="Tasks not found: nonexistent_task"):
        ingest_task(
            project_name=project_name_ingestion,
            task_id="nonexistent_task",
            subtask_structure="segmentation_proofread_simple",
            priority=2,
        )


def test_ingest_task_fully_processed_no_reingest(
    firestore_emulator, project_name_ingestion, existing_subtask_type
):
    """Test that ingesting a fully processed task without re_ingest returns False"""
    task_data = Task(
        **{
            "task_id": "fully_processed_task",
            "batch_id": "batch_1",
            "status": "fully_processed",
            "task_type": "segmentation",
            "link": "http://example.com/fully_processed_task",
        }
    )
    create_task(project_name_ingestion, task_data)

    result = ingest_task(
        project_name=project_name_ingestion,
        task_id="fully_processed_task",
        subtask_structure="segmentation_proofread_simple",
        priority=2,
    )
    assert result is False

    result = ingest_task(
        project_name=project_name_ingestion,
        task_id="fully_processed_task",
        subtask_structure="segmentation_proofread_simple",
        re_ingest="not_processed",
        priority=2,
    )
    assert result is False

    result = ingest_task(
        project_name=project_name_ingestion,
        task_id="fully_processed_task",
        subtask_structure="segmentation_proofread_simple",
        re_ingest="all",
        priority=2,
    )
    assert result is True


def test_ingest_batch_no_tasks(firestore_emulator, project_name_ingestion):
    """Test that ingesting a batch with no tasks returns False"""
    result = ingest_batch(
        project_name=project_name_ingestion,
        batch_id="nonexistent_batch",
        subtask_structure="segmentation_proofread_simple",
        priority=2,
    )
    assert result is False


def test_ingest_batch_no_matching_tasks(firestore_emulator, project_name_ingestion):
    """Test that ingesting a batch with no matching tasks returns False"""
    for i in range(1, 4):
        task_data = Task(
            **{
                "task_id": f"task_{i}",
                "batch_id": "batch_1",
                "status": "pending_ingestion",
                "task_type": "segmentation",
                "link": f"http://example.com/task_{i}",
            }
        )
        create_task(project_name_ingestion, task_data)

    result = ingest_batch(
        project_name=project_name_ingestion,
        batch_id="batch_2",
        subtask_structure="segmentation_proofread_simple",
        priority=2,
    )
    assert result is False
